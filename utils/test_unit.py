from random import random

from env.running_env import file_repo
from utils.PathManager import FileType
# from utils.DataExtractor import Extractor
# from utils.Visualizer import VisBoard
from utils.Visualizer import HRankBoard
from utils.objectIO import pickle_mkdir_save


def random_list(length=100):
    random_int_list = []
    for i in range(length):
        random_int_list.append(random.randint(0, 10))
    return random_int_list


def get_lists():
    lists = [[range(100), random_list(), random_list()]]
    return lists


# def test_sns():
#     repo = Extractor()
#     vis = VisBoard(repo)
#     vis.single_var_dist("F", "k")
#     #vis.double_vars_dist("F", "kr")
#     #vis.double_vars_regression("F", "kr")

def test_enum():
    print(FileType.LOG_TYPE)
    print(type(FileType.LOG_TYPE))

def test_path_manager():
    lis = [1, 2, 3]
    pickle_mkdir_save(lis, file_repo.new_rank()[0])

def test_hrank_visual():
    sim = HRankBoard()
    sim.simp_rank_img()

def main():
    test_hrank_visual()

