from random import random
from utils.Visualizer import HRankBoard
from env.running_env import args, file_repo


def random_list(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.randint(0, 10))
    return random_int_list


def get_lists():
    lists = [[range(100), random_list(), random_list()]]
    return lists


def test_hrank_visual():
    sim = HRankBoard()
    sim.simp_acc_compare_img()

def test_rank_img():
    board = HRankBoard()
    board.simp_rank_img(args.rank_norm_path)
    board.simp_rank_img(args.rank_plus_path)

def main():
    test_rank_img()

