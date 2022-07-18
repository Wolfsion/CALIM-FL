from copy import deepcopy
from typing import List, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from thop import profile
from torch.optim import lr_scheduler
from torch.nn.functional import binary_cross_entropy_with_logits
from timeit import default_timer as timer

from env.static_env import *
from env.running_env import *
from dl.wrapper import DeviceManager
from env.support_config import *
from dl.model import model_util
from dl.wrapper.optimizer import SGD_PruneFL
from utils.VContainer import VContainer


def error_mess(class_name: str, param: str) -> str:
    return f"Create an instance of the {class_name} need necessary {param} parameter."


# 参数自适应选择 kwargs*
# 为空 给出默认配置
class VWrapper:
    ERROR_MESS1 = "Model not support."
    ERROR_MESS2 = "Optimizer not support."
    ERROR_MESS3 = "Scheduler not support."
    ERROR_MESS4 = "Loss function not support."
    ERROR_MESS5 = "Checkpoint do not find model_key attribute."

    def __init__(self, model: nn.Module, dataloader: tdata.dataloader,
                 optimizer: VOptimizer, scheduler: VScheduler, loss: VLossFunc):
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.loss_type = loss

        self.device = None
        self.loss_func = None
        self.lr_scheduler = None
        self.optimizer = None

        self.model = model
        self.loader = dataloader

        self.last_acc = 0.0
        self.curt_batch = 0
        self.container = VContainer()

    def default_config(self):
        pass

    def init_device(self, use_gpu: bool, gpu_ids: List):
        self.device = DeviceManager.VDevice(use_gpu, gpu_ids)
        self.model = self.device.bind_model(self.model)

    def init_optim(self, learning_rate: float, momentum: float, weight_decay: float):
        if self.optimizer_type == VOptimizer.SGD:

            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=momentum, weight_decay=weight_decay)

        elif self.optimizer_type == VOptimizer.SGD_PFL:

            self.optimizer = SGD_PruneFL.SGD_PFL(self.model.parameters(), lr=INIT_LR)

        else:
            assert False, self.ERROR_MESS2

    def init_scheduler_loss(self, step_size: int, gamma: float):
        if self.scheduler_type == VScheduler.StepLR:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            assert False, self.ERROR_MESS3

        if self.loss_type == VLossFunc.Cross_Entropy:
            self.loss_func = binary_cross_entropy_with_logits
        else:
            assert False, self.ERROR_MESS4

    def running_scale(self, dataloader: tdata.dataloader = None):
        if dataloader is None:
            dataloader = self.loader
        inputs, label = next(iter(dataloader))
        size = inputs.size()
        return size

    def curt_state_info(self):
        pass

    # def feed_random_run(self):
    #     with torch.no_grad():
    #         for batch_idx in range(rank_limit):
    #             global_logger.info('using random data...')
    #             inputs = torch.randn(self.random_batch_size, 3, 32, 32)
    #             targets = torch.randn(self.random_batch_size, self.random_labels)
    #             self.wrapper.step(inputs, targets)

    # 当需要投喂测试集时，传入dataloader
    # 实现random数据投喂
    def step_run(self, batch_limit: int, train: bool = False,
                 pre_params: Iterator = None, loader: tdata.dataloader = None,
                 random: bool = False) -> (int, float, int):
        if train:
            self.model.train()
        else:
            self.model.eval()

        test_loss = 0
        correct = 0
        total = 0
        process = "Train" if train else "Test"

        curt_loader = loader if loader is not None else self.loader

        for batch_idx, (inputs, targets) in enumerate(curt_loader):
            if batch_idx > batch_limit:
                break

            inputs, labels = self.device.on_tensor(inputs, targets)
            pred = self.model(inputs)
            loss = self.loss_func(pred, labels)

            if train:
                if pre_params is not None:
                    # fedprox
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), pre_params):
                        proximal_term += (w - w_t).norm(2)
                    loss += (args.mu / 2) * proximal_term
                    # fedprox

                self.optim_step(loss)

            _, predicted = pred.max(1)
            _, targets = labels.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
            total += targets.size(0)

            global_logger.info('%s:batch_idx:%d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (process, batch_idx, test_loss / (batch_idx + 1),
                                  100. * correct / total, correct, total))

            # exp code
            self.container.flash(args.exp_name, 100. * correct / total)
            # exp code

            self.curt_batch += 1

        return correct, test_loss, total

    def optim_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_last_lr(self):
        if self.lr_scheduler is None:
            return self.optimizer.defaults["lr"]
        else:
            return self.lr_scheduler.get_last_lr()[0]

    def access_model(self) -> nn.Module:
        return self.device.access_model()

    def adjust_lr(self, factor: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor

    def show_lr(self):
        for param_group in self.optimizer.param_groups:
            global_logger.info(f"The current learning rate: {param_group['lr']}======>")

    def sync_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return next(self.device.on_tensor(tensor))

    # # to finish
    # def save_checkpoint(self, file_path: str):
    #     # exp_const_config = {"exp_name": CIFAR10_NAME, "batch_size": CLIENT_BATCH_SIZE,
    #     #                     "num_local_updates": NUM_LOCAL_UPDATES, "init_lr": INIT_LR,
    #     #                     "lrhl": LR_HALF_LIFE}
    #     exp_const_config = {"exp_name": CIFAR10_NAME, "state_dict": self.device.freeze_model()}
    #     # args_config = vars(self.args)
    #     # configs = exp_const_config.copy()
    #     # configs.update(args_config)
    #     model_util.mkdir_save(exp_const_config, file_path)
    #
    # # !
    # def load_checkpoint(self, path: str, model_key: str = 'state_dict'):
    #     checkpoint = torch.load(path, map_location=torch.device('cpu'))
    #     assert model_key in checkpoint.keys(), self.ERROR_MESS1
    #     self.device.load_model(checkpoint[model_key])

    def valid_performance(self, loader: tdata.dataloader):
        inputs = torch.rand(*self.running_scale())
        cpu_model = deepcopy(self.device.access_model()).cpu()
        flops, params = profile(cpu_model, inputs=(inputs,))
        time_start = timer()
        correct, test_loss, total = self.step_run(valid_limit, loader=loader)
        time_cost = timer() - time_start
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        global_logger.info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (test_loss / valid_limit, 100. * correct / total, correct, total))

        global_logger.info('Time cost: %.3f | FLOPs: %d | Params: %d'
                           % (time_cost, flops, params))

        global_logger.info('Total params: %d | Trainable params: %d'
                           % (total_params, total_trainable_params))
