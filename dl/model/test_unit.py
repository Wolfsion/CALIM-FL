from dl.SingleCell import SingleCell
from dl.model.ModelExt import Extender
from dl.model.model_util import create_model
from env.support_config import VModel


def test_model():
    model = create_model(VModel.VGG16)
    relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
    convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    # (cov_id - 1) * 4
    params = model.named_parameters()
    cnt = 0
    for name, item in params:
        print(f"{name}:{item.size()}")
        cnt += 1
    cnt = 0
    params = model.named_parameters()
    for name, item in params:
        if cnt in relucfg:
            print(f"---{name}:{item.size()}")
        cnt += 1
    mods = model.named_modules()
    for name, item in mods:
        print(f"+++{name}:")
    print(cnt)

    for id in relucfg:
        print(model.features[id])


def test_sub_model():
    model = create_model(VModel.VGG16)
    ext = Extender(model)
    layers = ext.conv_with_layers()
    cov_idx = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]
    for idx, layer in zip(cov_idx, layers):
        cov_layer = model.features[idx]
        print(cov_layer is layer)


def test_pre_model():
    cell = SingleCell(prune=True)
    cell.test_performance()
