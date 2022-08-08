from typing import Union, Generator


def deepcopy_dict(ori_dict: Union[dict, Generator]):
    generator = ori_dict.items() if isinstance(ori_dict, dict) else ori_dict
    copy_dict = dict()
    for key, param in generator:
        copy_dict[key] = param.clone()
    return copy_dict


def disp_num_params(model):
    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param
        print("{} remaining: {}/{} = {}".format(layer_prefix, layer_param_in_use, layer_all_param,
                                                layer_param_in_use / layer_all_param))
    print("Total: {}/{} = {}".format(total_param_in_use, total_all_param, total_param_in_use / total_all_param))
    return total_param_in_use / total_all_param
