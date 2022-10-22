import math
from copy import deepcopy
from typing import Iterator
import torch.nn as nn
import torch.utils.data as tdata

from dl.compress.HyperProvider import IntervalProvider, RateProvider
from dl.compress.VHRank import HRank
from dl.model.model_util import create_model
from dl.wrapper.ExitDriver import ExitManager
from dl.wrapper.Wrapper import VWrapper
from env.running_env import args, file_repo, global_container
from dl.data.dataProvider import get_data_loader
from env.running_env import global_logger
from env.static_env import wu_epoch, wu_batch, vgg16_candidate_rate


class SingleCell:
    ERROR_MESS1 = "Test must specify batch_limit."

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
        self.train_epoch = 1

        self.init_model_dataloader()
        self.init_wrapper()
        if prune:
            self.prune_ext = HRank(self.wrapper)
            self.hyper_inter = IntervalProvider()
            self.hyper_rate = RateProvider(args.prune_rate, args.federal_round, args.check_inter)

        self.exit_manager = ExitManager(self.wrapper)

    # init
    def init_model_dataloader(self):
        self.model = create_model(args.model, num_classes=args.num_classes)
        if self.dataloader is None:
            self.dataloader = get_data_loader(args.dataset, data_type="train",
                                              batch_size=args.batch_size, shuffle=True)
            self.test_dataloader = get_data_loader(args.dataset, data_type="test",
                                                   batch_size=args.batch_size, shuffle=True)

    def init_wrapper(self):
        self.wrapper = VWrapper(self.model, self.dataloader,
                                args.optim, args.scheduler, args.loss_func)
        self.wrapper.init_device(args.use_gpu, args.gpu_ids)
        self.wrapper.init_optim(args.learning_rate, args.momentum, args.weight_decay, args.nesterov)
        total_epoch = args.local_epoch * args.federal_round * args.active_workers if args.federal \
            else args.local_epoch
        self.wrapper.init_scheduler_loss(args.step_size, args.gamma, total_epoch, args.warm_steps, args.min_lr)
        if args.pre_train:
            self.wrapper.load_checkpoint(file_repo.model_path)

    # checkpoint
    def sync_model(self):
        if self.model.state_dict() != self.wrapper.model.state_dict():
            self.wrapper.model.load_state_dict(self.model.state_dict())
        # if self.model.state_dict() != self.prune_ext.model.state_dict():
        #     self.prune_ext.model.load_state_dict(self.model.state_dict())

    def access_model(self) -> nn.Module:
        return self.wrapper.access_model()

    def modify_model(self, model: nn.Module):
        self.model = model
        self.sync_model()

    def run_model(self, train: bool = False,
                  pre_params: Iterator = None,
                  batch_limit: int = 0) -> int:
        sum_loss = 0.0
        self.latest_feed_amount = 0

        if train:
            for i in range(args.local_epoch):
                global_logger.info(f"******The current train epoch: {self.train_epoch+i}******")
                if batch_limit == 0:
                    cort, total, loss = self.wrapper.step_run(args.batch_limit, train, pre_params)
                else:
                    cort, total, loss = self.wrapper.step_run(batch_limit, train, pre_params)
                sum_loss += loss
                self.latest_feed_amount += total
                self.wrapper.show_lr()
            self.train_epoch += args.local_epoch
            return sum_loss / args.local_epoch
        else:
            assert batch_limit != 0, self.ERROR_MESS1
            cort, total, loss = self.wrapper.step_run(batch_limit, train=False)
            global_container.flash(f'{args.exp_name}-test_acc', cort/total*100)
            return loss

    def test_performance(self):
        self.wrapper.valid_performance(self.test_dataloader)

    def decay_lr(self, epoch: int):
        self.wrapper.adjust_lr(math.pow(args.gamma, epoch))

    def show_lr(self):
        self.wrapper.show_lr()

    def prune_process(self, random: bool, plus: bool, federal: bool = args.federal):
        path_id = self.prune_ext.get_rank(random=random)
        args.rank_norm_path = file_repo.fetch_path(path_id)
        if plus:
            path_id = self.prune_ext.rank_plus(info_norm=args.info_norm, backward=args.backward)
            args.rank_plus_path = file_repo.fetch_path(path_id)

        if args.is_prune and not args.rank_plus:
            self.prune_ext.mask_prune(args.prune_rate)
        else:
            self.prune_ext.mask_prune(self.hyper_rate.get_curt_rate())
        if not federal:
            self.prune_ext.warm_up(wu_epoch, wu_batch)

    def prune_model(self, plus: bool = True, random: bool = False, auto_inter: bool = False):
        if auto_inter:
            self.prune_ext.get_rank_simp(random=random)
            self.hyper_inter.push_container(deepcopy(self.prune_ext.rank_list))
            if self.hyper_inter.is_timing():
                global_logger.info(f"Will prune in this round.")
                self.prune_process(random=random, plus=plus)
            else:
                global_logger.info(f"Do not prune in this round.")
        else:
            global_logger.info(f"Will prune in this round.")
            self.prune_process(random=random, plus=plus)

    def exit_proc(self, check: bool = False, one_key: str = None):
        if check:
            self.exit_manager.checkpoint_freeze()
        self.exit_manager.config_freeze()
        self.exit_manager.running_freeze(one_key)
