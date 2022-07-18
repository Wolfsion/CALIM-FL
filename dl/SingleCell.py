import math
from typing import Iterator, Any
import torch
import torch.nn as nn
import torch.utils.data as tdata

from dl.compress.VHRank import HRank
from dl.model.model_util import create_model, pre_train_model
from dl.wrapper.Wrapper import VWrapper
from env.running_env import args, file_repo
from dl.data.dataProvider import get_data_loader
from env.running_env import global_logger
from env.static_env import wu_epoch, wu_batch, vgg16_candidate_rate


class SingleCell:
    def __init__(self, dataloader: tdata.dataloader = None, prune: bool = False):
        self.model = None
        self.dataloader = dataloader
        self.test_dataloader = None

        # wrapper: VWrapper
        self.wrapper: VWrapper

        # prune_ext: HRank
        self.prune_ext: HRank

        # 训练数据个数
        self.latest_feed_amount = 0

        # 当前训练的批次
        self.train_epoch = 0

        self.init_model_dataloader()
        self.init_wrapper()
        if prune:
            self.init_prune()

    # init
    def init_model_dataloader(self):
        self.model = create_model(args.model)
        if args.pre_train:
            pre_train_model(self.model, file_repo.model_path)
        if self.dataloader is None:
            self.dataloader = get_data_loader(args.dataset, data_type="train",
                                              batch_size=args.batch_size, shuffle=True)
            self.test_dataloader = get_data_loader(args.dataset, data_type="test",
                                                   batch_size=args.batch_size, shuffle=True)

    def init_wrapper(self):
        self.wrapper = VWrapper(self.model, self.dataloader,
                                args.optim, args.scheduler, args.loss_func)
        self.wrapper.init_device(args.use_gpu, args.gpu_ids)
        self.wrapper.init_optim(args.learning_rate, args.momentum, args.weight_decay)
        self.wrapper.init_scheduler_loss(args.step_size, args.gamma)

    def init_prune(self):
        self.prune_ext = HRank(self.wrapper)

    # checkpoint
    def sync_model(self):
        if self.model.state_dict() != self.wrapper.model.state_dict():
            self.wrapper.model.load_state_dict(self.model.state_dict())
        if self.model.state_dict() != self.prune_ext.model.state_dict():
            self.prune_ext.model.load_state_dict(self.model.state_dict())

    def access_model(self) -> nn.Module:
        return self.wrapper.access_model()

    def modify_model(self, model: nn.Module):
        self.model = model
        self.sync_model()

    def run_model(self, train: bool = False,
                  pre_params: Iterator = None,
                  batch_limit: int = 0) -> int:
        self.latest_feed_amount = 0
        self.train_epoch += args.local_epoch
        for i in range(args.local_epoch):
            global_logger.info(f"Train epoch:{i+1}======>")

            if batch_limit == 0:
                _, loss, total = self.wrapper.step_run(args.batch_limit, train, pre_params)
            else:
                _, loss, total = self.wrapper.step_run(batch_limit, train, pre_params)
            self.latest_feed_amount += total
            self.wrapper.show_lr()
        return loss

    def test_performance(self):
        self.wrapper.valid_performance(self.test_dataloader)

    def decay_lr(self, epoch: int):
        self.wrapper.adjust_lr(math.pow(args.gamma, epoch))

    def show_lr(self):
        self.wrapper.show_lr()

    def prune_model(self):
        self.prune_ext.get_rank()
        self.prune_ext.rank_plus()
        # self.prune_ext.rank_plus2()
        self.prune_ext.mask_prune(vgg16_candidate_rate)
        self.prune_ext.warm_up(wu_epoch, wu_batch)
