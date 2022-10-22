# CIFAR VGG
import os

import torch
import torch.utils.data as tdata

from dl.SingleCell import SingleCell
from env.running_env import global_logger, args, file_repo
from federal.simulation.FLnodes import FLMaster
from federal.simulation.Worker import CVWorker
from utils.objectIO import pickle_mkdir_save
import dl.compress.compress_util as com_util


class CVMaster(FLMaster):
    remain_rounds = 10
    
    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict):
        master_cell = SingleCell(loader, True)
        super().__init__(workers, activists, local_epoch, master_cell)

        if workers_loaders is not None:
            workers_cells = [SingleCell(loader, True) for loader in list(workers_loaders.values())]
            self.workers_nodes = [CVWorker(index, cell) for index, cell in enumerate(workers_cells)]
        else:
            global_logger.error("Not Support Code.")
            exit(1)

        self.curt_selected = []
        self.check_inter = 9999

        self.pre_dict = self.cell.access_model().state_dict()
        self.curt_dict = self.cell.access_model().state_dict()
        self.pre_loss = 9999
        self.curt_loss = 0
        self.curt_epoch = 0

        self.first_prune = True

        self.des_size = []

    def prune_init(self, rounds: int, rate: list, inter: int):
        self.rounds = rounds
        self.curt_round = 1
        self.rate = rate
        self.check_inter = inter

    def union_run(self, random_data: bool, auto_inter: bool):
        for i in range(self.rounds):
            global_logger.info(f"======Federal Round: {i+1}======")
            self.schedule_strategy()
            self.info_sync()
            self.drive_workers()
            self.info_aggregation()
            self.weight_redo()

            if args.is_prune and args.rank_plus and i != self.rounds-1:
                self.master_prune(random_data, True)

            if args.is_prune and not args.rank_plus and i != self.rounds-1:
                if self.rounds - i < self.remain_rounds and self.first_prune:
                    self.check_inter = 1
                    self.master_prune(random_data, False)
                    self.first_prune = False

            self.serialize_size()
            self.curt_round = self.curt_round + 1

            # self.curt_epoch += args.active_workers * args.local_epoch

        path, _ = file_repo.new_seq('model_weight_size')
        pickle_mkdir_save(self.des_size, path)

        # self.cell.prune_ext.warm_up(wu_epoch, wu_batch)
        global_logger.info(f"Federal train finished======>")
        self.global_performance_detail()

    def schedule_strategy(self):
        self.curt_selected = super().schedule_strategy()

    def info_aggregation(self):
        workers_dict = []
        for index in self.curt_selected:
            workers_dict.append(self.workers_nodes[index].cell.access_model().state_dict())
        self.merge.merge_dict(workers_dict)
        for index in self.curt_selected:
            self.workers_nodes[index].cell.decay_lr(self.pace)

    def info_sync(self):
        workers_cells = []
        for index in self.curt_selected:
            workers_cells.append(self.workers_nodes[index].cell)
        self.merge.all_sync(workers_cells, 0)

    def drive_workers(self):
        for index in self.curt_selected:
            if args.fedprox:
                torch.cuda.empty_cache()
                self.workers_nodes[index].local_train(self.cell.access_model().parameters())
            else:
                torch.cuda.empty_cache()
                self.workers_nodes[index].local_train()

    def global_performance(self):
        self.curt_loss = self.cell.run_model(batch_limit=5)

    def global_performance_detail(self):
        self.cell.test_performance()

    def weight_redo(self):
        self.merge.weight_redo(self.cell)

    def master_prune(self, random_data: bool, rank_plus: bool = True):
        if self.curt_round % self.check_inter == 0:
            self.cell.prune_model(random=random_data, auto_inter=False, plus=rank_plus)
        else:
            global_logger.info(f"Do not prune in round:{self.curt_round}.")

    def serialize_size(self, coo_path: str = "coo"):
        model_dict = self.cell.access_model().cpu().state_dict()
        coo_dict = com_util.dict_coo_express(model_dict)
        pickle_mkdir_save(coo_dict, coo_path)
        self.des_size.append(os.stat(coo_path).st_size / (1024 * 1024))
        if args.use_gpu:
            self.cell.access_model().cuda()
