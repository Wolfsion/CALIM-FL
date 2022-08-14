from dl.data.dataProvider import get_data_loaders, get_data_loader
from dl.data.samplers import dataset_user_indices
from env.running_env import args
from federal.simulation.Master import CVMaster


def simulation_federal_process():
    # protocol implementation
    user_dict = dataset_user_indices(args.dataset, args.workers)
    workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
                                       users_indices=user_dict, num_workers=0, pin_memory=False)
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    master_node = CVMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                           loader=test_loader, workers_loaders=workers_loaders)
    master_node.union_run(args.federal_round)
    # protocol implementation


def main():
    simulation_federal_process()
