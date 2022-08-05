from dl.wrapper.Wrapper import VWrapper
from env.running_env import file_repo, args


class ExitManager:
    def __init__(self, wrapper: VWrapper):
        self.wrapper = wrapper

    def store_info(self):
        self.wrapper.container.store_all()
        name = str(args.model).split('.')[1]
        file, file_id = file_repo.new_checkpoint(name=name, fixed=True)
        self.wrapper.save_checkpoint(file)


def exit_process(wrapper: VWrapper):
    exit_driver = ExitManager(wrapper)
    exit_driver.store_info()

