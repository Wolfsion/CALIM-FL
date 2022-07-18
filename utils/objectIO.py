import os
import pickle
import warnings


def create_path(f: str):
    dir_name = os.path.dirname(f)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)


def touch_file(f):
    create_path(f)
    fid = open(f, 'w')
    fid.close()


# load obj not only wrapper: nn.Moudle
def pickle_mkdir_save(obj, f):
    create_path(f)
    # disabling warnings from torch.Tensor's reduce function. See issue: https://github.com/pytorch/pytorch/issues/38597
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(f, "wb") as opened_f:
            pickle.dump(obj, opened_f)
            opened_f.close()


def pickle_load(f):
    with open(f, "rb") as opened_f:
        obj = pickle.load(opened_f)
    return obj


def compare_obj(obj1, obj2) -> bool:
    return obj1 is obj2


if __name__ == '__main__':
    print("Nothing.")
