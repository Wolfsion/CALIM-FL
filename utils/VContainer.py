from collections import OrderedDict
from env.running_env import file_repo, args
from utils.objectIO import pickle_mkdir_save


class VContainer:
    ERROR_MESS1 = "The element type is not same as the former."

    def __init__(self):
        self.container = OrderedDict()
        self.keys = []
        self.base = None

    def flash(self, key: str, element):
        if key not in self.container.keys():
            self.keys.append(key)
            self.container[key] = []
            self.container[key].append(element)
        else:
            assert type(self.container[key][0]) == type(element), self.ERROR_MESS1
            self.container[key].append(element)

    def store(self, key: str):
        path, path_id = file_repo.new_inter(key)
        pickle_mkdir_save(self.container[key], path)

        # if key.find('acc') != -1:
        if args.curt_base:
            args.running_base_path = path
        else:
            args.running_plus_path = path

    def store_all(self):
        for key in self.keys:
            self.store(key)
