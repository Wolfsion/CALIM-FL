from dl.SingleCell import SingleCell
from dl.model.ModelExt import Extender
from dl.model.model_util import create_model
from env.support_config import VModel


def test_center_train():
    cell = SingleCell(prune=True)
    cell.run_model(train=True)
    cell.test_performance()


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


def test_prune_model():
    cell = SingleCell(prune=True)
    cell.prune_model()
    cell.test_performance()

# +++features.conv0:
# +++features.norm0:
# +++features.relu0:
# +++features.conv1:
# +++features.norm1:
# +++features.relu1:
# +++features.pool2:
# +++features.conv3:
# +++features.norm3:
# +++features.relu3:
# +++features.conv4:
# +++features.norm4:
# +++features.relu4:
# +++features.pool5:
# +++features.conv6:
# +++features.norm6:
# +++features.relu6:
# +++features.conv7:
# +++features.norm7:
# +++features.relu7:
# +++features.conv8:
# +++features.norm8:
# +++features.relu8:
# +++features.pool9:
# +++features.conv10:
# +++features.norm10:
# +++features.relu10:
# +++features.conv11:
# +++features.norm11:
# +++features.relu11:
# +++features.conv12:
# +++features.norm12:
# +++features.relu12:
# +++features.pool13:
# +++features.conv14:
# +++features.norm14:
# +++features.relu14:
# +++features.conv15:
# +++features.norm15:
# +++features.relu15:
# +++features.conv16:
# +++features.norm16:
# +++features.relu16:

def mask_gen():
    model = create_model(VModel.VGG16)
    ext = Extender(model)
    prune = ext.prune_layers()
    params = ext.prune_layer_parameters()
    fm = ext.feature_map_layers()
    for pa in params:
        print(pa.size())

def test_valid():
    cell = SingleCell(prune=True)
    cell.test_performance()

def main():
    test_prune_model()
