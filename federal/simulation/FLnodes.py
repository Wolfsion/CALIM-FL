import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Iterator
import torch.utils.data as tdata

from dl.SingleCell import SingleCell
from dl.compress.VHRank import HRank


# FedProx Nodes
from federal.aggregation.FedAvg import FedAvg


class FLMaster(ABC):
    def __init__(self, workers_num: int, schedule_num: int, local_epoch: int, master_cell: SingleCell):
        self.workers = workers_num
        self.plan = schedule_num
        self.pace = local_epoch * schedule_num // 2
        self.cell = master_cell
        self.merge = FedAvg(master_cell.access_model().state_dict())

    def schedule_strategy(self) -> List:
        indices = random.sample(range(0, self.workers), self.plan)
        return indices

    @abstractmethod
    def union_run(self, rounds: int):
        pass

    @abstractmethod
    def info_aggregation(self):
        pass

    @abstractmethod
    def global_performance(self):
        pass

    @abstractmethod
    def info_sync(self):
        pass

    @abstractmethod
    def drive_workers(self):
        pass


class FLWorker(ABC):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        self.id = worker_id
        self.cell = worker_cell

    @abstractmethod
    def local_train(self, global_parameters: Iterator):
        pass
