import os.path
from abc import ABC
from copy import deepcopy
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata

from dl.compress.HyperProvider import IntervalProvider
from dl.compress.compress_util import arrays_normalization, calculate_average_value
from dl.model.ModelExt import Extender
from dl.wrapper.Wrapper import VWrapper
from env.running_env import *
from env.static_env import rank_limit
from utils.objectIO import pickle_mkdir_save, pickle_load


class HRank(ABC):
    ERROR_MESS1 = "Generate mask must after calling get_rank()."

    def __init__(self, wrapper: VWrapper) -> None:
        self.rank_list = []
        self.info_flow_list = []
        self.degrees = []

        self.prune_mask = None
        self.feature_result: torch.Tensor = torch.tensor(0.)
        self.total: torch.Tensor = torch.tensor(0.)

        self.wrapper = wrapper

        model_percept = Extender(self.wrapper.access_model())
        self.reg_layers = model_percept.feature_map_layers()
        self.flow_layers_params = model_percept.flow_layers_parameters()
        self.prune_layers = model_percept.prune_layers()
        self.prune_layers_params = model_percept.prune_layer_parameters()
        self.accumulate = [0 for _ in range(len(self.flow_layers_params))]

    # get feature map of certain layer via hook
    def get_feature_hook(self, module, input, output):
        imgs = output.shape[0]
        channels = output.shape[1]
        ranks = torch.tensor([torch.linalg.matrix_rank(output[i, j, :, :]).item()
                              for i in range(imgs) for j in range(channels)])
        ranks = ranks.view(imgs, -1).float()

        # aggregation channel rank of all imgs
        ranks = ranks.sum(0)

        self.feature_result = self.feature_result * self.total + ranks
        self.total = self.total + imgs
        self.feature_result = self.feature_result / self.total

    def drive_hook(self, sub_module: nn.Module, random: bool = False):
        handler = sub_module.register_forward_hook(self.get_feature_hook)
        self.notify_feed_run(random)
        handler.remove()
        self.rank_list.append(self.feature_result.numpy())
        self.cache_flash()

    def cache_flash(self):
        self.feature_result: torch.Tensor = torch.tensor(0.)
        self.total: torch.Tensor = torch.tensor(0.)

    def rank_flash(self):
        self.rank_list.clear()
        self.info_flow_list.clear()
        self.degrees.clear()
        self.accumulate = [0 for _ in range(len(self.flow_layers_params))]
        self.cache_flash()

    def notify_feed_run(self, random: bool):
        if random:
            self.wrapper.random_run(rank_limit)
        else:
            self.wrapper.step_run(rank_limit)

    def notify_test(self, loader: tdata.dataloader):
        pass

    def get_rank(self, random: bool = False, store: bool = True) -> int:
        self.rank_flash()
        if os.path.isfile(args.rank_path) and not random:
            self.deserialize_rank(args.rank_path)
        else:
            for cov_layer in self.reg_layers:
                self.drive_hook(cov_layer, random)

        # for cov_layer in self.reg_layers:
        #     self.drive_hook(cov_layer, random)

        path_id = -1
        if store:
            path, path_id = file_repo.new_rank('Norm_Rank')
            self.save_rank(path)
        global_logger.info("Rank init finished======================>")
        return path_id

    def get_rank_simp(self, random: bool = True):
        self.rank_flash()
        cov_layer = self.reg_layers[0]
        self.drive_hook(cov_layer, random)

    def rank_plus(self, info_norm: int = 1, backward: int = 1) -> int:
        for params in self.flow_layers_params:
            filters = params.shape[0]
            channels = params.shape[1]
            if info_norm == 1:
                degree = torch.tensor([torch.linalg.matrix_rank(params[i, j, :, :]).item()
                                       for i in range(filters) for j in range(channels)])
            elif info_norm == 2:
                degree = torch.tensor([torch.sum(torch.abs(params[i, j, :, :])).item()
                                       for i in range(filters) for j in range(channels)])
            else:
                degree = 0
                global_logger.info('Illegal info_norm manner.')
                exit(1)
            # 之前得到的degree已经展成一维了，需要重新转换成二维
            degree = degree.reshape(params.size()[0:2])
            self.degrees.append(degree.numpy())
            self.info_flow_list.append(torch.sum(degree, dim=0).numpy())
            global_logger.info(f"Finish {params.size()} weight......")
        self.info_flow_list = arrays_normalization(self.info_flow_list,
                                                   calculate_average_value(self.rank_list))
        self.rank_aggregation(backward)
        path, path_id = file_repo.new_rank('Rank_Plus')
        self.save_rank(path)
        global_logger.info("Rank plus finished======================>")
        return path_id

    # rank_list:list[ndarray]
    # info_flow_list:list[ndarray]
    def rank_aggregation(self, backward: int, en_alpha: float = 0.7, en_shrink: float = 0.5):
        tail = len(self.rank_list) - 1
        if backward == 1:
            for index in range(tail):
                self.rank_list[index] += en_alpha * self.info_flow_list[index]
                en_alpha *= en_shrink
        elif backward == 2:
            self.accumulate[tail] = self.info_flow_list[tail]
            for index in range(tail-1, -1, -1):
                self.accumulate[index] += en_alpha * np.matmul(np.transpose(self.info_flow_list[index]),
                                                               self.info_flow_list[index+1])
                self.rank_list[index] += en_alpha * self.accumulate[index]
        else:
            global_logger.info('Illegal backward manner.')
            exit(1)

    def save_rank(self, path: str):
        pickle_mkdir_save(self.rank_list, path)

    def deserialize_rank(self, path: str):
        self.rank_list = pickle_load(path)

    def mask_prune(self, compress_rate: list):
        assert len(self.rank_list) != 0, self.ERROR_MESS1
        use_bn = len(self.prune_layers) >= 2 * len(self.reg_layers)
        param_index = 0
        for rank, rate in zip(self.rank_list, compress_rate):
            param = self.prune_layers_params[param_index]
            param_index += 1
            f, c, w, h = param.size()
            pruned_num = int(rate * f)
            ind = np.argsort(rank)[pruned_num:]
            zeros = torch.zeros(f, 1, 1, 1)
            for i in range(len(ind)):
                zeros[ind[i], 0, 0, 0] = 1.
            zeros = self.wrapper.sync_tensor(zeros)
            param.data = param.data * zeros

            if use_bn:
                param = self.prune_layers_params[param_index]
                param_index += 1
                param.data = param.data * torch.squeeze(zeros)
        global_logger.info("Prune finished======================>")

    def structure_prune(self, pruning_rate: List[float]):
        pass

    def warm_up(self, epochs: int, batch_limit: int):
        for i in range(epochs):
            self.wrapper.step_run(batch_limit=batch_limit, train=True)

        # # exp code
        # global_container.flash(f"{args.exp_name}_acc")
        # # exp code

        global_logger.info("Warm up finished======================>")

    def is_prune(self):
        pass

    # step = self.rounds // 10
    # if auto_inter:
    #     self.rate_provider = RateProvider(pruning_rate=self.rate,
    #                                       total_round=self.rounds,
    #                                       interval=step)
    #     self.inter_provider = IntervalProvider()
