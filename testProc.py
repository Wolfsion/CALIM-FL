from enum import Enum, unique

import pandas as pd
import torch

from dl.data.dataProvider import get_data_loaders, get_data_loader
from dl.data.samplers import cifar10_user_indices
from env import yaml2args
from env.running_env import args
import os

from utils.PathManager import FileType
from utils.objectIO import touch_file


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


def dataset():
    user_dict = cifar10_user_indices(args.workers)
    workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
                                       users_indices=user_dict, num_workers=0, pin_memory=False)
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    loaders = list(workers_loaders.values())
    print("here")




if __name__ == "__main__":
    from utils.test_unit import main
    main()
    print("----------------------")
