import torch
import torch.nn as nn


def is_pruned(module: nn.Module) -> bool:
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        return False
    else:
        return True


def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._modules.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")


class Extender:
    DICT_KEY1 = "layers"
    DICT_KEY2 = "layers_prefixes"
    DICT_KEY3 = "relu_layers"
    DICT_KEY4 = "relu_layers_prefixes"
    DICT_KEY5 = "prune_layers"
    DICT_KEY6 = "prune_layers_prefixes"

    def __init__(self, model: nn.Module):
        self.model = model
        self.masks = torch.tensor(0.)

    def collect_layers_params(self) -> dict:
        layers = []
        layers_prefixes = []
        relu_layers = [m for (k, m) in self.model.named_modules() if isinstance(m, nn.ReLU)]
        relu_layers_prefixes = [k for (k, m) in self.model.named_modules() if isinstance(m, nn.ReLU)]

        traverse_module(self.model, lambda x: len(list(x.parameters())) != 0, layers, layers_prefixes)

        prune_indices = [ly_id for ly_id, layer in enumerate(layers) if is_pruned(layer)]
        prune_layers = [layers[ly_id] for ly_id in prune_indices]
        prune_layers_prefixes = [layers_prefixes[ly_id] for ly_id in prune_indices]

        ret = {
            self.DICT_KEY1: layers,
            self.DICT_KEY2: layers_prefixes,
            self.DICT_KEY3: relu_layers,
            self.DICT_KEY4: relu_layers_prefixes,
            self.DICT_KEY5: prune_layers,
            self.DICT_KEY6: prune_layers_prefixes
        }
        return ret

    # feature_map_layers() return a list, the length of it is n.
    # flow_layers_parameters() return a list, the length of it must be n. Neglect the first conv layer params.
    # prune_layers() return a list, the length of it must be n+1 or 2n+2.
    # prune_layer_parameters() is same as prune_layers().
    # why + 1? A: we do not prune the last conv layer
    # why 2n? A: Every convolution layer has a BN layer behind it, BN layers also need to be pruned
    # n+1 show that model does not contain batch-norm layer
    # 2n+2 show that model contains batch-norm layer

    # conv pre one layer
    def feature_map_layers(self) -> list:
        layers = []
        pre_module = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and pre_module is not None:
                layers.append(pre_module)
            if len(list(module.modules())) == 1 and not isinstance(module, nn.Sequential):
                pre_module = module
        return layers

    # info_flow layer for rank_plus
    def flow_layers_parameters(self) -> list:
        layer_parameters = []
        first = True
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                if first:
                    first = False
                    continue
                else:
                    for name, params in module.named_parameters():
                        if name == 'weight':
                            layer_parameters.append(params)
        return layer_parameters

    # conv & BN
    def prune_layers(self) -> list:
        layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
            if isinstance(module, nn.BatchNorm2d):
                layers.append(module)
        return layers

    # conv & BN parameter: torch.Tensor
    def prune_layer_parameters(self) -> list:
        layers = self.prune_layers()
        layer_parameters = []
        for layer in layers:
            for name, params in layer.named_parameters():
                if name == 'weight':
                    layer_parameters.append(params)
        return layer_parameters

    def mask_compute(self):
        pass
