from dl.data.samplers import CF10NormSamplerPool, dataset_user_indices
from dl.data.dataProvider import get_data_loader, DataLoader, get_data_loaders
from env.running_env import args
from env.static_env import CIFAR10_NAME


def loader_pool(num_slices: int, batch_size: int) -> [DataLoader]:
    sampler_pool = CF10NormSamplerPool(num_slices)
    loader_list = [get_data_loader(CIFAR10_NAME, data_type="train",
                                   batch_size=batch_size, shuffle=False,
                                   sampler=sampler_pool.get_sampler(i),
                                   num_workers=8, pin_memory=False) for i in range(num_slices)]
    return loader_list


def test_loaders():
    user_dict = dataset_user_indices(args.dataset, args.workers)
    workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
                                       users_indices=user_dict, num_workers=0, pin_memory=False)
    test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
    loaders = list(workers_loaders.values())
    print("here")


# def dataset():
#     user_dict = dataset_user_indices(args.dataset, args.workers)
#     workers_loaders = get_data_loaders(args.dataset, data_type="train", batch_size=args.batch_size,
#                                        users_indices=user_dict, num_workers=0, pin_memory=False)
#     test_loader = get_data_loader(args.dataset, data_type="test", batch_size=args.batch_size,
#                                   shuffle=True, num_workers=0, pin_memory=False)
#     loaders = list(workers_loaders.values())
#     print("here")
