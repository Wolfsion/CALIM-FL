import gc
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

from torch.optim.lr_scheduler import ReduceLROnPlateau

from dl.wrapper.optimizer.WarmUpCosinLR import WarmUPCosineLR
from dl.wrapper.optimizer.WarmUpStepLR import WarmUPStepLR
from env.static_env import *
from env.running_env import *
from dl.wrapper import DeviceManager
from env.support_config import *
from dl.model import model_util
from dl.wrapper.optimizer import SGD_PruneFL
from utils.VContainer import VContainer
from utils.objectIO import pickle_mkdir_save, pickle_load


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

        self.latest_acc = 0.0
        self.latest_loss = 0.0
        self.curt_batch = 0
        self.curt_epoch = 0

        self.seed = 2022

    def default_config(self):
        pass

    def init_device(self, use_gpu: bool, gpu_ids: List):
        self.device = DeviceManager.VDevice(use_gpu, gpu_ids)
        self.model = self.device.bind_model(self.model)

    def init_optim(self, learning_rate: float, momentum: float,
                   weight_decay: float, nesterov: bool):
        if self.optimizer_type == VOptimizer.SGD:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        elif self.optimizer_type == VOptimizer.SGD_PFL:
            self.optimizer = SGD_PruneFL.SGD_PFL(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_type == VOptimizer.ADAM:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay)
        else:
            assert False, self.ERROR_MESS2

    def init_scheduler_loss(self, step_size: int, gamma: float, T_max: int, warm_up_steps: int, min_lr: float):
        if self.scheduler_type == VScheduler.StepLR:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif self.scheduler_type == VScheduler.CosineAnnealingLR:
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif self.scheduler_type == VScheduler.WarmUPCosineLR:
            self.lr_scheduler = WarmUPCosineLR(self.optimizer, warm_up_steps, T_max, lr_min=min_lr)
        elif self.scheduler_type == VScheduler.ReduceLROnPlateau:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        elif self.scheduler_type == VScheduler.WarmUPStepLR:
            self.lr_scheduler = WarmUPStepLR(self.optimizer, step_size=step_size, gamma=gamma)
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
        data_size = inputs.size()
        label_size = label.size()
        return data_size, label_size

    # def curt_state_info(self):
    #     global_logger.info("Not support.")

    def random_run(self, batch_limit: int):
        torch.manual_seed(self.seed)
        with torch.no_grad():
            global_logger.info('Using random data.======>')
            data_size, label_size = self.running_scale()
            for batch_idx in range(batch_limit):
                inputs = torch.randn(data_size)
                targets = torch.randn(label_size)
                inputs, labels = self.device.on_tensor(inputs, targets)
                pred = self.model(inputs)
        self.seed += 1

    # 当需要投喂测试集时，传入dataloader
    # 实现random数据投喂
    def step_run(self, batch_limit: int, train: bool = False,
                 pre_params: Iterator = None, loader: tdata.dataloader = None) -> (int, float, int):
        if train:
            self.model.train()
        else:
            self.model.eval()

        train_loss = 0
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
            train_loss += loss.item()
            total += targets.size(0)

            self.latest_acc = 100. * correct / total
            self.latest_loss = train_loss / (batch_idx + 1)

            if batch_idx % print_interval == 0:
                global_logger.info('%s:batch_idx:%d | Loss: %.6f | Acc: %.3f%% (%d/%d)'
                                   % (process, batch_idx, self.latest_loss, self.latest_acc, correct, total))
            self.curt_batch += 1

        if train:
            gc.collect()
            torch.cuda.empty_cache()
            self.curt_epoch += 1
            global_container.flash(f"{args.exp_name}_acc", self.latest_acc)
            self.scheduler_step()

        return correct, total, self.latest_loss

    def optim_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def scheduler_step(self):
        if self.scheduler_type == VScheduler.ReduceLROnPlateau:
            self.lr_scheduler.step(metrics=self.latest_loss)
        else:
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
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        global_logger.info(f"The current learning rate: {lr}======>")

    def sync_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return next(self.device.on_tensor(tensor))

    def save_checkpoint(self, file_path: str):
        exp_checkpoint = {"exp_name": CIFAR10_NAME, "state_dict": self.device.freeze_model(),
                          "batch_size": args.batch_size, "last_epoch": self.curt_epoch,
                          "init_lr": args.learning_rate}
        pickle_mkdir_save(exp_checkpoint, file_path)

    def load_checkpoint(self, path: str, model_key: str = 'state_dict'):
        if path.find('.pt') == -1:
            checkpoint = pickle_load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        assert model_key in checkpoint.keys(), self.ERROR_MESS5
        self.device.load_model(checkpoint[model_key])

    def valid_performance(self, loader: tdata.dataloader):
        # inputs = torch.rand(*(self.running_scale()[0]))
        # cpu_model = deepcopy(self.device.access_model()).cpu()
        # flops, params = profile(cpu_model, inputs=(inputs,))
        inputs = torch.rand(*(self.running_scale()[0])).cuda()
        gpu_model = deepcopy(self.device.access_model()).cuda()
        flops, params = profile(gpu_model, inputs=(inputs,))

        time_start = timer()
        correct, total, test_loss = self.step_run(valid_limit, loader=loader)
        time_cost = timer() - time_start
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        global_logger.info('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (test_loss, 100. * correct / total, correct, total))

        global_logger.info('Time cost: %.6f | FLOPs: %d | Params: %d'
                           % (time_cost, flops, params))

        global_logger.info('Total params: %d | Trainable params: %d'
                           % (total_params, total_trainable_params))
