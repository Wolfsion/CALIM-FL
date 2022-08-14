from matplotlib import pyplot as plt
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


def test_lr():
    net = resnet18(pretrained=False)
    optim = SGD(net.parameters(), lr=0.1)
    for i in range(5):
        optim.step()
        print("Optim:", optim.state_dict()['param_groups'][0]['lr'])
    epoch = 200
    lr = WarmUPStepLR(optim)
    x = [i+1 for i in range(epoch)]
    y = []
    z = []

    for i in range(epoch):
        optim.step()
        lr.step()
        y.append(optim.state_dict()['param_groups'][0]['lr'])

    lr = WarmUPCosineLR(optim, T_max=epoch, lr_min=0.0)
    for i in range(epoch):
        optim.step()
        lr.step()
        z.append(optim.state_dict()['param_groups'][0]['lr'])

    print(y)
    print(z)

    # plt.plot(x, y)
    # plt.plot(x, z)
    # plt.show()
