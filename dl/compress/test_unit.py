import torch

from dl.model.ModelExt import Extender
from dl.model.model_util import create_model
from env.static_env import *
from env.running_env import *

from dl.compress.Sparse import TopKSparse
from dl.compress.Quantization import QuantizationSGD
from dl.compress.compress_util import get_size


def mask_gen():
    model = create_model(VModel.VGG16)
    ext = Extender(model)
    prune = ext.prune_layers()
    params = ext.prune_layer_parameters()
    fm = ext.feature_map_layers()
    for pa in params:
        print(pa.size())


def sparse_and_quan():
    tensor = torch.randn(2, 2, 2, 2)
    sparser = TopKSparse(0.5)
    quaner = QuantizationSGD(16)

    print("origin tensor:", get_size(tensor))
    ct = quaner.compress(tensor)["com_tensor"]

    print(get_size(ct))

def main():
    sparse_and_quan()
