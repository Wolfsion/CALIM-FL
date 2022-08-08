from typing import Iterator

from dl.SingleCell import SingleCell
from env.running_env import global_logger
from federal.simulation.FLnodes import FLWorker


# CIFAR VGG
class CVWorker(FLWorker):
    def __init__(self, worker_id: int, worker_cell: SingleCell):
        super().__init__(worker_id, worker_cell)

    def local_train(self, global_params: Iterator):
        global_logger.info(f'Train from {self.id}')
        self.cell.run_model(train=True, pre_params=global_params)

