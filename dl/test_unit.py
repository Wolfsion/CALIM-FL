from dl.SingleCell import SingleCell
from env.running_env import args
from utils.Visualizer import HRankBoard


def test_center_train():
    cell = SingleCell()
    cell.run_model(train=True)
    cell.test_performance()
    cell.exit_proc(check=False)


def test_valid():
    cell = SingleCell(prune=True)
    cell.test_performance()


# vgg16 resnet56 - cifar10
def test_prune_model():
    cell = SingleCell(prune=True)
    cell.prune_model(plus=False)
    cell.test_performance()
    cell.exit_proc()


def test_prune_model_plus():
    cell = SingleCell(prune=True)
    cell.prune_model()
    cell.test_performance()
    cell.exit_proc()

    board = HRankBoard()
    board.simp_rank_img(args.rank_norm_path)
    board.simp_rank_img(args.rank_plus_path)


def plus_ori_compare():
    args.curt_base = True
    test_prune_model()
    args.curt_base = False
    test_prune_model_plus()

    board = HRankBoard()
    board.simp_acc_compare_img(baseline=args.running_base_path, vrank=args.running_plus_path)


def test_prune_model_random():
    cell = SingleCell(prune=True)
    cell.prune_model(random=True)
    cell.test_performance()
    cell.exit_proc()


def plus_random_compare():
    args.curt_base = True
    test_prune_model_plus()
    args.curt_base = False
    test_prune_model_random()

    board = HRankBoard()
    board.simp_acc_compare_img(vrank=args.running_base_path, random=args.running_plus_path)


def test_prune_model_interval():
    pass


def init_interval_compare():
    pass


# vgg16 resnet56 resnet100 mobilenetV2 - cifar10 cifar100
def test_auto_prune():
    pass


def total_auto_line():
    pass


def hrank():
    pass


def main():
    plus_ori_compare()
