from collections import OrderedDict
from env.running_env import file_repo
from utils.objectIO import pickle_mkdir_save


class VContainer:
    ERROR_MESS1 = "The element is not fit."

    def __init__(self):
        self.container = OrderedDict()

    def flash(self, key: str, element):
        if key not in self.container.keys():
            self.container[key] = []
            self.container[key].append(element)
        else:
            assert type(self.container[key][0]) == type(element), self.ERROR_MESS1
            self.container[key].append(element)

    def store(self, key: str):
        path = file_repo.new_acc(key)[0]
        pickle_mkdir_save(self.container[key], path)

    def load(self, file_path: str):
        pass

