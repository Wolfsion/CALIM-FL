from dl.wrapper.Wrapper import VWrapper
from env.running_env import file_repo, args
from utils.objectIO import str_save


class ExitManager:
    def __init__(self, wrapper: VWrapper):
        self.wrapper = wrapper

    @staticmethod
    def config_freeze():
        config = args.get_snapshot()
        file, file_id = file_repo.new_exp(args.exp_name)
        str_save(config, file)

    def checkpoint_freeze(self, fixed: bool = True):
        name = str(args.model).split('.')[1]
        file, file_id = file_repo.new_checkpoint(name=name, fixed=fixed)
        self.wrapper.save_checkpoint(file)

    def running_freeze(self):
        paths = "\n".join(file_repo.reg_path)
        file, _ = file_repo.new_exp(f"{args.exp_name}_paths")
        str_save(paths, file)
        self.wrapper.container.store_all()
