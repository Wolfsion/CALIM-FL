import re

import numpy as np
import torch

zero_value = 0.001


# Calculate memory size of the give tensor - units:B
def get_size(tensor: torch.Tensor) -> int:
    nums_element = tensor.nelement()
    regex = r"(\d+)$"
    test_str = str(tensor.dtype)
    match = next(re.finditer(regex, test_str, re.MULTILINE))
    bits = int(match.group())
    return bits * nums_element // 8


# 仅支持纬度数据
# only support single dimension
def max_min_normalization(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x) + zero_value)


# only support single dimension
def mean_var_normalization(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + zero_value)


def arrays_normalization(arrays: list, scales: list, way=mean_var_normalization) -> list:
    assert len(scales) == len(arrays), 'The length of arrays must be equal to the length of scale.'
    norm_list = []
    for x, coff in zip(arrays, scales):
        norm_list.append(coff*way(x))
    return norm_list


def calculate_average_value(ndarray_list: list) -> list:
    ret_list = []
    for array in ndarray_list:
        ret_list.append(np.mean(array))
    return ret_list
