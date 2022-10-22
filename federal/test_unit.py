import time
from copy import deepcopy

import torch.cuda

from dl.data.dataProvider import get_data_loaders, get_data_loader
from dl.data.samplers import dataset_user_indices
from env.running_env import args
from federal.simulation.Master import CVMaster
from utils.Visualizer import HRankBoard


def simulation_federal_process(one_key: str = None):
    # protocol implementation
    torch.cuda.empty_cache()
    user_dict = dataset_user_indices(args.dataset, args.workers, args.non_iid)
    workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
                                       users_indices=user_dict, num_workers=0, pin_memory=False)
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    master_node = CVMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                           loader=test_loader, workers_loaders=workers_loaders)

    master_node.prune_init(args.federal_round, args.prune_rate, args.check_inter)
    master_node.union_run(args.random_data, args.auto_inter)
    # protocol implementation
    master_node.cell.exit_proc(one_key=one_key)


def test_random():
    args.curt_base = True
    args.random_data = False
    simulation_federal_process(f'{args.exp_name}-test_acc')
    args.curt_base = False
    args.random_data = True
    simulation_federal_process(f'{args.exp_name}-test_acc')

    board = HRankBoard()
    board.simp_acc_compare_img(baseline=args.running_base_path, random=args.running_plus_path)


def test_fedavg():
    args.fedprox = False
    args.random_data = True

    # args.curt_base = True
    #
    # args.is_prune = False
    # args.rank_plus = False
    #
    # simulation_federal_process(f'{args.exp_name}-test_acc')
    #
    # torch.cuda.empty_cache()
    # args.curt_base = False
    # args.curt_final = False
    #
    # args.is_prune = True
    # args.rank_plus = False
    #
    # simulation_federal_process(f'{args.exp_name}-test_acc')

    torch.cuda.empty_cache()
    args.curt_base = False
    args.curt_final = True

    args.is_prune = True
    args.rank_plus = True

    simulation_federal_process(f'{args.exp_name}-test_acc')

    board = HRankBoard()
    board.simp_acc_compare_img(Fedavg=args.running_base_path,
                               HRank=args.running_plus_path,
                               VRankFL=args.running_final_path)


def test_fedprox():
    args.fedprox = True
    args.random_data = True

    args.curt_base = True
    args.is_prune = False
    args.rank_plus = False
    simulation_federal_process(f'{args.exp_name}-test_acc')

    torch.cuda.empty_cache()
    args.curt_base = False
    args.curt_final = False

    args.is_prune = True
    args.rank_plus = False

    simulation_federal_process(f'{args.exp_name}-test_acc')

    torch.cuda.empty_cache()
    args.curt_base = False
    args.curt_final = True
    args.is_prune = True
    args.rank_plus = True

    simulation_federal_process(f'{args.exp_name}-test_acc')

    board = HRankBoard()
    board.simp_acc_compare_img(Fedprox=args.running_base_path,
                               HRank=args.running_plus_path,
                               VRankFL=args.running_final_path)


def main():
    if args.random_test:
        test_random()
    elif args.fedavg_test:
        test_fedavg()
    elif args.fedprox_test:
        test_fedprox()
    else:
        simulation_federal_process()
