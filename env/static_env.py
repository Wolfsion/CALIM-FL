from enum import Enum, unique
from utils.Vlogger import VLogger

# Uniform const
CPU = -6
GPU = -66
CPU_STR_LEN = 3
INIT_LR = 0.1
LR_HALF_LIFE = 10000
ORIGIN_CP_RATE = [0.] * 100

# simulation
NUM_LOCAL_UPDATES = 5
CLIENT_BATCH_SIZE = 20
NUM_CLIENTS = 10

MAX_ROUND = 10001
MAX_DEC_DIFF = 0.3
ADJ_INTERVAL = 50
ADJ_HALF_LIFE = 10000

QUICK_LR_HALF_LIFE = 100
STEP_DECAY = 0.5 ** (1 / LR_HALF_LIFE)

# CIFAR10 const config
CIFAR10_NAME = "CIFAR10"
CIFAR10_CLASSES = 10
CIFAR10_NUM_TRAIN_DATA = 50000
CIFAR10_NUM_TEST_DATA = 10000
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# CIFAR100 const config
CIFAR100_CLASSES = 100
CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

# VGG const config

# vgg16_bn
vgg16_candidate_rate = [0.45]*7 + [0.78]*5

# resnet50
resnet50_candidate_rate = [0.1] + [0.2]*3 + [0.5]*16

# resnet56
resnet56_candidate_rate = [0.] + [0.18]*29

# mobilenetv2
mobile_candidate_rate = [0.] + [0.3]*7

# Others
MAX_HOOK_LAYER = 50
valid_limit = 5
rank_limit = 10

# Default_config
YAML_PATH = r'share/default_config.yml'

# Warm-up config
wu_epoch = 5
wu_batch = 8192

# acc_info
print_interval = 10
