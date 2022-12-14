import random
from typing import List

import torch
from abc import ABC, abstractmethod
from torch.utils.data import Sampler
from fedlab.utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner

from env.running_env import global_logger
from env.static_env import *
from dl.data.dataProvider import get_data
from env.support_config import VDataSet


def dataset_user_indices(dataset_type: VDataSet, num_slices, non_iid: str, seed: int = 2022):
    dataset = get_data(dataset_type, data_type="train")
    if dataset_type == VDataSet.CIFAR10:
        if non_iid == 'hetero':
            dir_part = CIFAR10Partitioner(dataset.targets, num_slices,
                                          balance=None, partition="dirichlet",
                                          dir_alpha=0.3, seed=seed)
        else:
            dir_part = CIFAR10Partitioner(dataset.targets, num_slices,
                                          balance=None, partition="shards",
                                          num_shards=200, seed=seed)
    elif dataset_type == VDataSet.CIFAR100:
        if non_iid == 'hetero':
            dir_part = CIFAR100Partitioner(dataset.targets, num_slices,
                                           balance=None, partition="dirichlet",
                                           dir_alpha=0.3, seed=seed)
        else:
            dir_part = CIFAR100Partitioner(dataset.targets, num_slices,
                                           balance=None, partition="shards",
                                           num_shards=200, seed=seed)
    else:
        global_logger.error("Not supported non_iid type.")
        dir_part = None
        exit(1)
    return dir_part.client_dict


class NSampler(Sampler):
    def __init__(self, dataset: VDataSet, indices: []):
        self.dataset = dataset
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class CF10NormSamplerPool:
    def __init__(self, num_slices: int, seed: int = 1):
        self.clients = num_slices
        cifar10 = get_data(VDataSet.CIFAR10.name, data_type="train")
        balance_iid_part = CIFAR10Partitioner(cifar10.targets, num_slices, balance=True,
                                              partition="iid", seed=seed)
        tmp_dict = balance_iid_part.client_dict
        self.samplers = [NSampler(VDataSet.CIFAR10, tmp_dict[i]) for i in range(num_slices)]

    def get_sampler(self, index: int) -> NSampler:
        assert index < self.clients, self.ERROR_MESS2
        return self.samplers[index]


class LSampler(Sampler, ABC):
    ERROR_MESS1 = "The dataset is not supported."

    def __init__(self, datatype: VDataSet, num_slices, num_round, data_per_client,
                 client_selection, client_per_round=None):
        self.indices = []
        self.users_indices = dict()
        self.getIndices(datatype, num_slices, num_round, data_per_client,
                        client_selection, client_per_round)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    @abstractmethod
    def getIndices(self, datatype, num_slices, num_round, data_per_client,
                   client_selection, client_per_round):
        pass


# create indices list, reorder data
class IIDSampler(LSampler):
    def getIndices(self, datatype, num_slices, num_round, data_per_client, client_selection, client_per_round):
        if datatype == VDataSet.CIFAR10.value:
            total_num = CIFAR10_NUM_TRAIN_DATA
        else:
            total_num = 0

        rand_perm = torch.randperm(total_num).tolist()
        len_slice = total_num // num_slices
        tmp_indices = []

        for i in range(num_slices):
            tmp_indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        range_partition = list(range(num_slices))
        new_list_ind = [[] for _ in range(num_slices)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_slices

        list_pos = [0] * num_slices
        for _ in range(num_round):
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
            else:
                selected_client_idx = range_partition

        for client_idx in selected_client_idx:
            ind = tmp_indices[client_idx]
            pos = list_pos[client_idx]
            while len(new_list_ind[client_idx]) < pos + data_per_client:
                random.shuffle(ind)
                new_list_ind[client_idx].extend(ind)
            self.indices.extend(new_list_ind[client_idx][pos:pos + data_per_client])
            list_pos[client_idx] = pos + data_per_client


class CF10NIIDSampler(LSampler):
    ERROR_MESS1 = "The idx_selected is null."

    def __init__(self, num_slices, max_num_round, data_per_client, client_selection: bool,
                 client_per_round=None, seed=1, datatype=VDataSet.CIFAR10):
        self.seed = seed
        self.idx_selected = []
        super().__init__(datatype, num_slices, max_num_round, data_per_client,
                         client_selection, client_per_round)

    def getIndices(self, datatype, num_slices, num_round, data_per_client, client_selection, client_per_round):
        assert datatype == VDataSet.CIFAR10, "must be CIFAR10"
        cifar10 = get_data(VDataSet.CIFAR10, data_type="train")
        hetero_dir_part = CIFAR10Partitioner(cifar10.targets, num_slices,
                                             balance=None, partition="dirichlet",
                                             dir_alpha=0.3, seed=self.seed)
        self.users_indices = hetero_dir_part.client_dict
        range_partition = list(range(num_slices))
        new_list_ind = [[] for _ in range(num_slices)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_slices

        list_pos = [0] * num_slices
        for _ in range(num_round):
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
                self.idx_selected.append(selected_client_idx)
            else:
                selected_client_idx = range_partition

            for client_idx in selected_client_idx:
                ind = self.users_indices[client_idx]
                pos = list_pos[client_idx]
                while len(new_list_ind[client_idx]) < pos + data_per_client:
                    new_list_ind[client_idx].extend(ind)
                self.indices.extend(new_list_ind[client_idx][pos:pos + data_per_client])
                list_pos[client_idx] = pos + data_per_client

    def curt_selected(self) -> List[List[int]]:
        assert len(self.idx_selected) != 0, self.ERROR_MESS1
        return self.idx_selected
