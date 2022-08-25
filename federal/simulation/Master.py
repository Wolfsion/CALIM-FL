# CIFAR VGG

import torch.utils.data as tdata

from dl.SingleCell import SingleCell
from env.running_env import global_logger
from federal.simulation.FLnodes import FLMaster
from federal.simulation.Worker import CVWorker


class CVMaster(FLMaster):
    def __init__(self, workers: int, activists: int, local_epoch: int,
                 loader: tdata.dataloader, workers_loaders: dict):
        master_cell = SingleCell(loader)
        super().__init__(workers, activists, local_epoch, master_cell)
        if workers_loaders is not None:
            workers_cells = [SingleCell(loader) for loader in list(workers_loaders.values())]
            self.workers_nodes = [CVWorker(index, cell) for index, cell in enumerate(workers_cells)]

        self.curt_selected = []

        self.pre_dict = self.cell.access_model().state_dict()
        self.curt_dict = self.cell.access_model().state_dict()
        self.pre_loss = 9999
        self.curt_loss = 0
        self.curt_epoch = 0

    def union_run(self, rounds: int):
        for i in range(rounds):
            global_logger.info(f"======Federal Round: {i+1}======")
            self.schedule_strategy()
            self.info_sync()
            self.drive_workers()
            self.info_aggregation()
            self.weight_redo()
            # self.curt_epoch += args.active_workers * args.local_epoch

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

    # def info_aggregation(self):
    #     # 计算总权重
    #     total_federal_weight = 0
    #     for index in self.curt_selected:
    #         total_federal_weight += self.workers_nodes[index].cell.latest_feed_amount
    #
    #     # 初始化全0参数矩阵
    #     params_dict = OrderedDict()
    #     for k, v in self.workers_nodes[0].cell.access_model().named_parameters():
    #         params_dict[k] = torch.zeros_like(v.data)
    #
    #     # 加和参数
    #     for index in self.curt_selected:
    #         for k, v in self.workers_nodes[index].cell.access_model().named_parameters():
    #             # params_dict[k] += v.data * (self.workers_nodes[index].cell.latest_feed_amount /
    #             #                             total_federal_weight)
    #             params_dict[k] += v.data * (1 / args.active_workers)
    #
    #     # 更新全局模型参数
    #     load_params(self.cell.access_model(), params_dict)
    #
    #     for index in self.curt_selected:
    #         self.workers_nodes[index].cell.decay_lr(args.active_workers * args.local_epoch)

    # def info_sync(self):
    #     for index in self.curt_selected:
    #         load_model_params(self.workers_nodes[index].cell.access_model(), self.cell.access_model())
    #
    #         self.workers_nodes[index].cell.show_lr()

    def info_sync(self):
        workers_cells = []
        for index in self.curt_selected:
            workers_cells.append(self.workers_nodes[index].cell)
        self.merge.all_sync(workers_cells, 0)

    def drive_workers(self):
        for index in self.curt_selected:
            self.workers_nodes[index].local_train(self.cell.access_model().parameters())

    def global_performance(self):
        self.curt_loss = self.cell.run_model(batch_limit=5)

    def global_performance_detail(self):
        self.cell.test_performance()

    # def weight_redo(self):
    #     if self.curt_loss > args.coff * self.pre_loss:
    #         self.cell.access_model().load_state_dict(self.pre_dict)
    #     else:
    #         self.pre_dict = self.cell.access_model().state_dict()

    def weight_redo(self):
        self.merge.weight_redo(self.cell)
