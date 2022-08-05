from ruamel.yaml import YAML
from copy import deepcopy

from env.args_request import DEFAULT_ARGS
from env.support_config import *


def model_str2enum(value: str) -> VModel:
    if value == 'vgg16':
        ret = VModel.VGG16
    elif value == 'resnet56':
        ret = VModel.ResNet56
    elif value == 'resnet110':
        ret = VModel.ResNet110
    elif value == 'mobilenetV2':
        ret = VModel.MobileNetV2
    else:
        ret = VModel.UPPER
    return ret


def dataset_str2enum(value: str) -> VDataSet:
    if value == 'cifar10':
        ret = VDataSet.CIFAR10
    elif value == 'cifar100':
        ret = VDataSet.CIFAR100
    else:
        ret = VDataSet.UPPER
    return ret


def optim_str2enum(value: str) -> VOptimizer:
    if value == 'sgd':
        ret = VOptimizer.SGD
    elif value == 'sgd_pfl':
        ret = VOptimizer.SGD_PFL
    elif value == 'adam':
        ret = VOptimizer.ADAM
    else:
        ret = VOptimizer.UPPER
    return ret


def scheduler_str2enum(value: str) -> VScheduler:
    if value == 'step_lr':
        ret = VScheduler.StepLR
    elif value == 'cosine_lr':
        ret = VScheduler.CosineAnnealingLR
    elif value == 'warmup_cos_lr':
        ret = VScheduler.WarmUPCosineLR
    else:
        ret = VScheduler.UPPER
    return ret


def loss_str2enum(value: str) -> VLossFunc:
    if value == 'cross_entropy':
        ret = VLossFunc.Cross_Entropy
    else:
        ret = VLossFunc.UPPER
    return ret


class ArgRepo:
    ERROR_MESS1 = "The yaml file lacks necessary parameters."

    def __init__(self, yml_path: str):
        self.r_yaml = YAML(typ="safe")
        self.yml_path = yml_path
        self.init_attr_placeholder()
        self.runtime_attr_placeholder()

    def init_attr_placeholder(self):
        self.exp_name = None
        self.model = None
        self.pre_train = None
        self.use_gpu = None
        self.gpu_ids = None
        self.dataset = None
        self.batch_size = None
        self.optim = None
        self.nesterov = None
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.loss_func = None
        self.scheduler = None
        self.step_size = None
        self.gamma = None
        self.warm_steps = None
        self.non_iid = None
        self.workers = None
        self.active_workers = None
        self.federal_round = None
        self.local_epoch = None
        self.batch_limit = None
        self.coff = None
        self.test_batch_limit = None

        self.info_norm = None
        self.backward = None

    def runtime_attr_placeholder(self):
        self.rank_norm_path = None
        self.rank_plus_path = None
        self.num_classes = None

    def activate(self, strict: bool = False):
        options = self.parse_args()
        if strict:
            assert self.is_legal(options), self.ERROR_MESS1
        self.mount_args(options)

    def parse_args(self) -> dict:
        with open(self.yml_path, 'r') as f:
            args = deepcopy(DEFAULT_ARGS)
            args.update(dict(self.r_yaml.load(f)))
        return args

    def is_legal(self, options: dict) -> bool:
        pass

    def mount_args(self, options: dict):
        for k, v in options.items():
            if k == 'model':
                setattr(self, k, model_str2enum(v))
            elif k == 'dataset':
                setattr(self, k, dataset_str2enum(v))
            elif k == 'optim':
                setattr(self, k, optim_str2enum(v))
            elif k == 'scheduler':
                setattr(self, k, scheduler_str2enum(v))
            elif k == 'loss_func':
                setattr(self, k, loss_str2enum(v))
            else:
                setattr(self, k, v)

        if self.dataset == VDataSet.CIFAR10:
            setattr(self, 'num_classes', 10)
        elif self.dataset == VDataSet.CIFAR100:
            setattr(self, 'num_classes', 100)
        else:
            print("The dataset is not supported.")
            exit(1)
