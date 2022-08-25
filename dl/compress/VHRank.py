import os.path
from abc import ABC
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
        self.prune_mask = None
        self.feature_result: torch.Tensor = torch.tensor(0.)
        self.total: torch.Tensor = torch.tensor(0.)

        self.wrapper = wrapper

        model_percept = Extender(self.wrapper.access_model())
        self.reg_layers = model_percept.feature_map_layers()
        self.flow_layers_params = model_percept.flow_layers_parameters()
        self.prune_layers = model_percept.prune_layers()
        self.prune_layers_params = model_percept.prune_layer_parameters()

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
        self.cache_flash()

    def notify_feed_run(self, random: bool):
        if random:
            self.wrapper.random_run(rank_limit)
        else:
            self.wrapper.step_run(rank_limit)

    def notify_test(self, loader: tdata.dataloader):
        pass

    def get_rank(self, random: bool = False, store: bool = True) -> int:

        if os.path.isfile(args.rank_path) and not random:
            self.deserialize_rank(args.rank_path)
        else:
            for cov_layer in self.reg_layers:
                self.drive_hook(cov_layer, random)

        path_id = -1
        if store:
            path, path_id = file_repo.new_rank('Norm_Rank')
            self.save_rank(path)
        global_logger.info("Rank init finished======================>")
        return path_id

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
            accumulate = 0
            for index in range(tail, -1, -1):
                self.rank_list[index] += en_alpha * self.info_flow_list[index]

                accumulate += en_alpha * self.info_flow_list[index]
                self.rank_list[index] += accumulate
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

        # exp code
        self.wrapper.container.store(f"{args.exp_name}_acc")
        # exp code

        global_logger.info("Warm up finished======================>")


class VGG16HRank(HRank):
    def __init__(self, wrapper: VWrapper) -> None:
        super().__init__(wrapper)

    def get_rank_plus(self):
        pass

    def structure_prune(self, pruning_rate: List[float]):
        pass

    # def load_params(self, rank_dict: OrderedDict = None):
    #     last_select_index = None  # Conv index selected in the previous layer
    #     if rank_dict is None:
    #         if not self.rank_dict:
    #             self.deserialize_rank()
    #         iter_ranks = iter(self.rank_dict.values())
    #     else:
    #         iter_ranks = iter(rank_dict.values())
    #     osd = self.checkpoint.state_dict()
    #
    #     for name, module in self.checkpoint.named_modules():
    #         if self.wrapper.device.GPUs:
    #             name = name.replace('module.', '')
    #
    #         if isinstance(module, nn.Conv2d):
    #             # self.cp_model_sd[name + '.bias'] = osd[name + '.bias']
    #             ori_weight = osd[name + '.weight']
    #             cur_weight = self.cp_model_sd[name + '.weight']
    #             ori_filter_num = ori_weight.size(0)
    #             cur_filter_num = cur_weight.size(0)
    #
    #             if ori_filter_num != cur_filter_num:
    #
    #                 rank = next(iter_ranks)
    #                 # preserved filter index based on rank
    #                 select_index = np.argsort(rank)[ori_filter_num - cur_filter_num:]
    #
    #                 # traverse list in increase order(not necessary step)
    #                 select_index.sort()
    #
    #                 if last_select_index is not None:
    #                     for index_i, i in enumerate(select_index):
    #                         for index_j, j in enumerate(last_select_index):
    #                             self.cp_model_sd[name + '.weight'][index_i][index_j] = \
    #                                 osd[name + '.weight'][i][j]
    #                 else:
    #                     for index_i, i in enumerate(select_index):
    #                         self.cp_model_sd[name + '.weight'][index_i] = \
    #                             osd[name + '.weight'][i]
    #
    #                 last_select_index = select_index
    #
    #             elif last_select_index is not None:
    #                 for i in range(ori_filter_num):
    #                     for index_j, j in enumerate(last_select_index):
    #                         self.cp_model_sd[name + '.weight'][i][index_j] = \
    #                             osd[name + '.weight'][i][j]
    #
    #             # retain origin channel
    #             else:
    #                 self.cp_model_sd[name + '.weight'] = ori_weight
    #                 last_select_index = None
    #
    #         # elif isinstance(module, nn.Linear):
    #         #     self.cp_model_sd[name + '.weight'] = osd[name + '.weight']
    #         #     self.cp_model_sd[name + '.bias'] = osd[name + '.bias']
    #
    #     self.cp_model.load_state_dict(self.cp_model_sd)
    #     torch.save(self.cp_model.state_dict(), 'test.pt')
    #
    # def init_cp_model(self, pruning_rate: List[float]):
    #     self.cp_model = model_util.vgg_16_bn(pruning_rate)
    #     model_util.initialize(self.cp_model)
    #     self.cp_model_sd = self.cp_model.state_dict()


# class ResNet56HRank(HRank):
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.relu_cfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]
#
#     def get_rank(self, random: bool = False):
#         hook_cnt = 0
#         cov_layer = wrapper.model.relu
#         self.drive_hook(cov_layer, hook_cnt)
#         hook_cnt += 1
#
#         # ResNet56 per block
#         for i in range(3):
#             # eval()!!!
#             block = eval('self.wrapper.wrapper.layer%d' % (i + 1))
#             for j in range(9):
#                 for _relu in range(2):
#                     if _relu == 0:
#                         cov_layer = block[j].relu1
#                     else:
#                         cov_layer = block[j].relu2
#                     self.drive_hook(cov_layer, loader, hook_cnt)
#                     hook_cnt += 1
#         file_repo.reset_rank_index()
#
#     def deserialize_rank(self):
#         file_repo.reset_rank_index()
#
#     def init_cp_model(self, pruning_rate: List[float]):
#         pass
#
#     def load_params(self):
#         pass


class ResNet50HRank(HRank):
    pass
