###
# Default Env (refer)
###
import argparse

exp1_config = r'share/cifar10-vgg16.yml'
exp2_config = r'share/cifar10-resnet56.yml'
exp3_config = r'share/cifar100-resnet110.yml'
exp4_config = r'share/cifar100-mobilenetV2.yml'

###
# Default Env (refer)
###

###
# Custom Env (to fill)
###

# entire path
datasets_base = r'~/la/datasets'

test_config = r'share/cifar10-center.yml'

vgg16_model = r'res/checkpoint/vgg/vgg_16_bn.pt'
resnet56_model = r'res/checkpoint/resnet/resnet_56.pt'
resnet110_model = r'res/checkpoint/resnet/ResNet110.snap'
mobilenetv2_model = r'res/checkpoint/mobilenet/MobileNetV2.snap'

vgg16_ranks = r'res/milestone/vgg_16_bn/Norm_Rank---08.13.npy'
resnet56_ranks = r'res/milestone/resnet_56/none.npy'
resnet110_ranks = r'res/milestone/ResNet110/none.npy'  # Norm_Rank---08.14.npy
mobilenetv2_ranks = r'res/milestone/MobileNetV2/Norm_Rank---08.13.npy'


def auto_config(option: str):
    global test_config
    if option == 'e1':
        test_config = exp1_config
    elif option == 'e2':
        test_config = exp2_config
    elif option == 'e3':
        test_config = exp3_config
    elif option == 'e4':
        test_config = exp4_config
    else:
        print('Not support config option.')
        exit(1)

###
# Custom Env (to fill)
###
