from typing import List, Tuple

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.DataExtractor import Extractor
from env.running_env import *
from utils.objectIO import pickle_load


def text_info(title: str, x_label: str, y_label: str):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

# axis mode:
'''
    F:FLOPs
    A:Acc
    T:Turn
    R:Rate
'''

# form mode:
'''
    h:histogram-4
    k:kernel density estimation-2
    r:rug-1
'''
class HRankBoard:
    def __init__(self):
        sns.set(style='darkgrid', color_codes=True)
        plt.figure(figsize=(20, 15))
        self.title = 'vgg16-hrank'

    def simp_rank_img(self):
        path = r"/home/xd/la/projects/HRankFL/res/milestone/vgg_16_bn/2022.07.15_22-21-23.npy"
        hrank_list = pickle_load(path)
        hr = []
        for rank in hrank_list:
            hr.extend(rank.tolist())
        layer_prefix = []
        for ind, rank in enumerate(hrank_list):
            layer_prefix.extend([f"conv{ind+1}" for _ in rank])

        df = pd.DataFrame(list(zip(hr, layer_prefix)), columns=['Rank', 'Layer'])
        sns.boxplot(x='Layer', y='Rank', data=df)
        plt.savefig(file_repo.new_img("box_plot")[0])
        plt.clf()
        sns.swarmplot(x='Layer', y='Rank', data=df)
        plt.savefig(file_repo.new_img("scatter_plot")[0])

    def simp_acc_compare_img(self):
        a = [1, 2, 3, 4, 5]
        b = [0, 3, 0, 6, 0]
        c = [4, 4, 5, 6, 7]

        hue_a = [[num, ind + 1, 'a'] for ind, num in enumerate(a)]
        hue_b = [[num, ind + 1, 'b'] for ind, num in enumerate(b)]
        hue_c = [[num, ind + 1, 'c'] for ind, num in enumerate(c)]

        hue_a.extend(hue_b)
        hue_a.extend(hue_c)

        df = pd.DataFrame(hue_a, columns=['Acc', 'Epoch', 'Class'])

        sns.set(style='darkgrid', color_codes=True)
        sns.lineplot(data=df, x="Epoch", y="Acc", hue="Class")
        plt.show()


class VisBoard:
    ERROR_MESS1 = "The axis length cannot exceed 4 characters."
    ERROR_MESS2 = "The form length cannot exceed 3 characters."
    ERROR_MESS3 = "The mode str contains not defined character."
    map_dict = {"h": 4, "k": 2, "r": 1}
    AXIS_KEYS = ['F', 'A', 'I', 'R']
    KEYS = ['FLOPs', 'Acc', 'Interval', 'Rate']
    SNS_STYLE = "darkgrid"
    PALETTE = "rainbow"

    HIST_FLAG = 0
    KDE_FLAG = 1
    RUG_FLAG = 2

    def __init__(self):
        self.io = Extractor()
        # figSize and font
        sns.set_style(self.SNS_STYLE)
        # sns.set_palette(self.PALETTE)

    def check_axis_form(self, axis: str, form: str):
        assert len(axis) < 5, self.ERROR_MESS1
        assert all([ch in self.AXIS_KEYS for ch in axis]), self.ERROR_MESS3

        assert len(form) < 4, self.ERROR_MESS2
        assert all([ch in self.map_dict.keys() for ch in form]), self.ERROR_MESS3

    def map_bools(self, form: str) -> Tuple[bool, bool, bool]:
        ret = 0
        for ch in form:
            ret += self.map_dict[ch]
        hist_flag = False if ret < 4 else True
        kde_flag = False if ret // 2 % 2 == 0 else True
        rug_flag = False if ret % 2 == 0 else True
        return hist_flag, kde_flag, rug_flag

    def map_list(self, mode: str) -> List[int]:
        indices = []
        for ch in mode:
            indices.append(self.AXIS_KEYS.index(ch))
        return indices

    def init_graph_contex(self, axis: str, form: str) -> Tuple[pd.DataFrame, List, Tuple]:
        self.check_axis_form(axis, form)
        graph_type = self.map_bools(form)
        data_key = self.map_list(axis)
        data_frame = self.io.map_vars(data_key)
        return data_frame, data_key, graph_type

    def end_graph_contex(self, indices: List[int]):
        if len(indices) == 1:
            title = f"{self.KEYS[indices[0]]} Graph"
            text_info(title, self.KEYS[indices[0]], "frequency")
        else:
            title = f"{self.KEYS[indices[0]]}~{self.KEYS[indices[1]]} Graph"
            text_info(title, self.KEYS[indices[0]], self.KEYS[indices[1]])
        plt.savefig(file_repo.img(title))

    # 直方图和密度图必须二选一，rug可以按需选择
    def single_var_dist(self, axis: str, form: str = 'k'):
        df, indices, mode = self.init_graph_contex(axis, form)
        SINGLE = 0
        if mode[self.HIST_FLAG]:
            sns.displot(df, x=self.KEYS[indices[SINGLE]],
                        kde=mode[self.KDE_FLAG],
                        rug=mode[self.RUG_FLAG],
                        hue="class")
        else:
            sns.displot(df, x=self.KEYS[indices[SINGLE]],
                        kind="kde",
                        rug=[self.RUG_FLAG],
                        hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_dist"))

    # def single_var_sequence(self, axis: str, form: str = 'k'):
    #     df, indices, mode = self.init_graph_contex(axis, form)
    #     SINGLE = 0
    #     sns.lineplot(x=df.index,
    #                  y=df[self.KEYS[indices[SINGLE]]],
    #                  ci=None)
    #     plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_sequencer_chart"))

    def single_var_sequence(self, axis: str, form: str = 'k'):
        # df, indices, mode = self.init_graph_contex(axis, form)
        # SINGLE = 0
        # sns.lineplot(data=df, x=self.KEYS[indices[0]],
        #              y=self.KEYS[indices[1]],
        #              ci=None, hue="Interval")
        # text_info('Top-Acc1 Retrain', 'Retrain batch', 'Acc/%')
        # plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_sequencer_chart"))

        df, indices, mode = self.init_graph_contex(axis, form)
        SINGLE = 0
        sns.lineplot(x=df.index,
                     y=df[self.KEYS[indices[SINGLE]]],
                     ci=None)
        text_info('pairwise_distances', 'Round', 'Magnitude')
        plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_sequencer_chart"))

    def double_vars_relation(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        DOUBLE_X = 0
        DOUBLE_Y = 1
        sns.lineplot(data=df, x=self.KEYS[indices[DOUBLE_X]],
                     y=self.KEYS[indices[DOUBLE_Y]],
                     ci=None, hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[DOUBLE_X]]}~{self.KEYS[indices[DOUBLE_Y]]}_relation"))

    def double_vars_dist(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        DOUBLE_X = 0
        DOUBLE_Y = 1
        sns.jointplot(data=df, x=self.KEYS[indices[DOUBLE_X]], y=self.KEYS[indices[DOUBLE_Y]], hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[DOUBLE_X]]}~{self.KEYS[indices[DOUBLE_Y]]}_dist"))

    def double_vars_regression(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        DOUBLE_X = 0
        DOUBLE_Y = 1
        sns.regplot(data=df, x=self.KEYS[indices[DOUBLE_X]], y=self.KEYS[indices[DOUBLE_Y]], hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[DOUBLE_X]]}~{self.KEYS[indices[DOUBLE_Y]]}_dist"))

    def multi_vars_regression(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        sns.lmplot()
