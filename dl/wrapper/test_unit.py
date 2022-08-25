from matplotlib import pyplot as plt
from torch import optim
from torch.optim import SGD
from torchvision.models import resnet18

from dl.wrapper.ExitDriver import ExitManager
from dl.wrapper.Wrapper import VWrapper
from dl.wrapper.optimizer.WarmUpCosinLR import WarmUPCosineLR
from dl.wrapper.optimizer.WarmUpStepLR import WarmUPStepLR


def exit_process(wrapper: VWrapper):
    exit_driver = ExitManager(wrapper)
    exit_driver.checkpoint_freeze()
    exit_driver.config_freeze()
    exit_driver.running_freeze()


def show_lr(net):
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # 打印当前学习率
    print(optimizer.state_dict()['param_groups'][0]['lr'])


def test_lr():
    net = resnet18(pretrained=False)
    optimizer = SGD(net.parameters(), lr=0.1)
    for i in range(5):
        optimizer.step()
        print("Optim:", optimizer.state_dict()['param_groups'][0]['lr'])
    epoch = 200
    lr = WarmUPStepLR(optimizer)
    x = [i+1 for i in range(epoch)]
    y = []
    z = []

    for i in range(epoch):
        optimizer.step()
        lr.step()
        y.append(optimizer.state_dict()['param_groups'][0]['lr'])

    lr = WarmUPCosineLR(optimizer, T_max=epoch, lr_min=0.0)
    for i in range(epoch):
        optimizer.step()
        lr.step()
        z.append(optimizer.state_dict()['param_groups'][0]['lr'])

    print(y)
    print(z)

    # plt.plot(x, y)
    # plt.plot(x, z)
    # plt.show()
