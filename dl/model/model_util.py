import os
import pickle
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

from dl.model.mobilenet import MobileNetV2
from dl.model.resnet import ResNet, BasicBlock
from dl.model.vgg import VGG16, VGG11
from env.running_env import global_logger
from env.static_env import *
from env.support_config import VModel


def initialize(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model


def create_model(model: VModel, compress_rate=ORIGIN_CP_RATE, num_classes=10) -> nn.Module:
    if model == VModel.VGG11:
        return initialize(VGG11(compress_rate=compress_rate))
    elif model == VModel.VGG16:
        return initialize(VGG16(compress_rate=compress_rate))
    elif model == VModel.ResNet56:
        return initialize(ResNet(BasicBlock, 56, compress_rate=compress_rate, num_classes=100))
    elif model == VModel.ResNet110:
        return initialize(ResNet(BasicBlock, 56, compress_rate=compress_rate, num_classes=100))
    elif model == VModel.MobileNetV2:
        return initialize(MobileNetV2(compress_rate=compress_rate, width_mult=1))


def load_model_params(load_model: nn.Module, source_model: nn.Module):
    for old_params, new_params in zip(load_model.parameters(), source_model.parameters()):
        old_params.data = new_params.data.clone()


def load_params(model: nn.Module, params: OrderedDict):
    for k, v in model.named_parameters():
        v.data = params[k].data.clone()


def model_device(model: nn.Module):
    curt = str(next(model.parameters()).device)
    if (len(curt) > CPU_STR_LEN):
        return GPU
    else:
        return CPU


def dict_diff(dict1: dict, dict2: dict):
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        if k1 != k2:
            global_logger.info('Key beq:dict1_key:', k1)
            global_logger.info('Key beq:dict2_key:', k2)
        else:
            if not v1.equal(v2):
                global_logger.info(f"The value of key:{k1} is not equal.")


def pre_train_model(model_obj: nn.Module, path_pt: str):
    checkpoint = torch.load(path_pt, map_location=torch.device('cpu'))
    model_key = 'state_dict'
    model_obj.load_state_dict(checkpoint[model_key])
