from enum import Enum, unique

import numpy as np
import pandas as pd
import torch
from scipy import spatial
from sklearn.metrics import pairwise_distances
from torch import nn, optim

from dl.data.dataProvider import get_data_loaders, get_data_loader
from dl.data.samplers import dataset_user_indices
from dl.model.ModelExt import Extender
from dl.model.model_util import create_model
from env import yaml2args
from env.running_env import args
import os

from env.support_config import VModel
from utils.Cleaner import FileCleaner
from utils.PathManager import FileType
from utils.Visualizer import HRankBoard
from utils.objectIO import touch_file, remove_file, pickle_mkdir_save, str_save


def path_name():
    path = r'res/images/test.png'
    base, file = os.path.split(path)
    name, postfix = os.path.splitext(file)
    print(f"path:{base}, file:{file}")
    print(f"name:{name}, postfix:{postfix}")
    repath = os.path.join(base, name)
    refile = os.path.join(repath, file)
    print(f"refile:{refile}")
    print(f"{name}{postfix}")


def test_args():
    args = yaml2args.ArgRepo(r'share/cifar10-vgg16.yml')
    args.activate()
    print("here")


def clear_file():
    cl = FileCleaner(remain_days=7)
    cl.clear_files()


def test_lis(new: str):
    lis = ['a', 'v', 'b']
    for item in lis:
        if item == new:
            print("Exist")
            return
    print("Not Exist")


def test_resnet():
    # clear_file()
    from torchsummary import summary
    model = create_model(VModel.MobileNetV2)
    # model = create_model(VModel.ResNet110)
    summary(model, input_size=(3, 32, 32), batch_size=-1, device="cpu")

    ext = Extender(model)

    layers = []
    pre_module = None
    for name, module in model.named_modules():
        # print(name, '---', module)
        if isinstance(module, nn.Conv2d) and pre_module is not None:
            layers.append(pre_module)
        if len(list(module.modules())) == 1:
            pre_module = module

    fl = ext.feature_map_layers()
    print('+++', len(fl))
    print(fl)

    ori_layers = []
    params = model.named_parameters()
    index = 0

    # for cov_id in range(1, 56):
    #     for name, param in params:
    #         if index == (cov_id - 1) * 3:
    #             print(name)
    #             ori_layers.append(param)
    #         elif (cov_id - 1)*3 < index < cov_id*3:
    #             print('+++', name)
    #             ori_layers.append(param)
    #         index += 1
    # print(index)

    compress_rate = [0.] + [0.18] * 29
    stage_repeat = [9, 9, 9]
    stage_out_channel = [16] + [16] * 9 + [32] * 9 + [64] * 9

    stage_oup_cprate = []
    stage_oup_cprate += [compress_rate[0]]
    for i in range(len(stage_repeat) - 1):
        stage_oup_cprate += [compress_rate[i + 1]] * stage_repeat[i]
    stage_oup_cprate += [0.] * stage_repeat[-1]
    mid_cprate = compress_rate[len(stage_repeat):]

    print(len(mid_cprate))
    print(len(stage_oup_cprate))

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1 - mid_cprate[i - 1]))]

    print(overall_channel)
    print(mid_channel)

    param = torch.randn(5, 3, 3, 3)
    a = torch.randn(5, 3)
    degree = a.reshape(param.size()[0:2])
    print(param.size()[0:2])
    b = torch.sum(a, dim=0)

    print(b.size())


if __name__ == "__main__":
    from dl.compress.test_unit import main
    main()
    print("----------------------")
