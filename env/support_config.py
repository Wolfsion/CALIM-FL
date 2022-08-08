from enum import Enum, unique


# dataset type
@unique
class VDataSet(Enum):
    LOWER = 0
    CIFAR10 = 1
    CIFAR100 = 2
    UPPER = 3


# Model Type
@unique
class VModel(Enum):
    LOWER = 0
    VGG11 = 1
    VGG16 = 2
    ResNet56 = 3
    ResNet110 = 4
    MobileNetV2 = 5
    UPPER = 6


# Optimizer Type
@unique
class VOptimizer(Enum):
    LOWER = 0
    SGD = 1
    SGD_PFL = 2
    ADAM = 3
    UPPER = 4


# Optimizer Type
@unique
class VScheduler(Enum):
    LOWER = 0
    StepLR = 1
    CosineAnnealingLR = 2
    WarmUPCosineLR = 3
    ReduceLROnPlateau = 4
    WarmUPStepLR = 5
    UPPER = 6


# loss func
@unique
class VLossFunc(Enum):
    LOWER = 0
    Cross_Entropy = 1
    UPPER = 2


