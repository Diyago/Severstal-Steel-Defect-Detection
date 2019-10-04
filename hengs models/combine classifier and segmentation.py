import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose)
from albumentations.torch import ToTensor
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F

BatchNorm2d = nn.BatchNorm2d

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD = [0.229, 0.224, 0.225]

###############################################################################
CONVERSION = [
    'block0.0.weight', (64, 3, 7, 7), 'conv1.weight', (64, 3, 7, 7),
    'block0.1.weight', (64,), 'bn1.weight', (64,),
    'block0.1.bias', (64,), 'bn1.bias', (64,),
    'block0.1.running_mean', (64,), 'bn1.running_mean', (64,),
    'block0.1.running_var', (64,), 'bn1.running_var', (64,),
    'block1.1.conv_bn1.conv.weight', (64, 64, 3, 3), 'layer1.0.conv1.weight', (64, 64, 3, 3),
    'block1.1.conv_bn1.bn.weight', (64,), 'layer1.0.bn1.weight', (64,),
    'block1.1.conv_bn1.bn.bias', (64,), 'layer1.0.bn1.bias', (64,),
    'block1.1.conv_bn1.bn.running_mean', (64,), 'layer1.0.bn1.running_mean', (64,),
    'block1.1.conv_bn1.bn.running_var', (64,), 'layer1.0.bn1.running_var', (64,),
    'block1.1.conv_bn2.conv.weight', (64, 64, 3, 3), 'layer1.0.conv2.weight', (64, 64, 3, 3),
    'block1.1.conv_bn2.bn.weight', (64,), 'layer1.0.bn2.weight', (64,),
    'block1.1.conv_bn2.bn.bias', (64,), 'layer1.0.bn2.bias', (64,),
    'block1.1.conv_bn2.bn.running_mean', (64,), 'layer1.0.bn2.running_mean', (64,),
    'block1.1.conv_bn2.bn.running_var', (64,), 'layer1.0.bn2.running_var', (64,),
    'block1.2.conv_bn1.conv.weight', (64, 64, 3, 3), 'layer1.1.conv1.weight', (64, 64, 3, 3),
    'block1.2.conv_bn1.bn.weight', (64,), 'layer1.1.bn1.weight', (64,),
    'block1.2.conv_bn1.bn.bias', (64,), 'layer1.1.bn1.bias', (64,),
    'block1.2.conv_bn1.bn.running_mean', (64,), 'layer1.1.bn1.running_mean', (64,),
    'block1.2.conv_bn1.bn.running_var', (64,), 'layer1.1.bn1.running_var', (64,),
    'block1.2.conv_bn2.conv.weight', (64, 64, 3, 3), 'layer1.1.conv2.weight', (64, 64, 3, 3),
    'block1.2.conv_bn2.bn.weight', (64,), 'layer1.1.bn2.weight', (64,),
    'block1.2.conv_bn2.bn.bias', (64,), 'layer1.1.bn2.bias', (64,),
    'block1.2.conv_bn2.bn.running_mean', (64,), 'layer1.1.bn2.running_mean', (64,),
    'block1.2.conv_bn2.bn.running_var', (64,), 'layer1.1.bn2.running_var', (64,),
    'block1.3.conv_bn1.conv.weight', (64, 64, 3, 3), 'layer1.2.conv1.weight', (64, 64, 3, 3),
    'block1.3.conv_bn1.bn.weight', (64,), 'layer1.2.bn1.weight', (64,),
    'block1.3.conv_bn1.bn.bias', (64,), 'layer1.2.bn1.bias', (64,),
    'block1.3.conv_bn1.bn.running_mean', (64,), 'layer1.2.bn1.running_mean', (64,),
    'block1.3.conv_bn1.bn.running_var', (64,), 'layer1.2.bn1.running_var', (64,),
    'block1.3.conv_bn2.conv.weight', (64, 64, 3, 3), 'layer1.2.conv2.weight', (64, 64, 3, 3),
    'block1.3.conv_bn2.bn.weight', (64,), 'layer1.2.bn2.weight', (64,),
    'block1.3.conv_bn2.bn.bias', (64,), 'layer1.2.bn2.bias', (64,),
    'block1.3.conv_bn2.bn.running_mean', (64,), 'layer1.2.bn2.running_mean', (64,),
    'block1.3.conv_bn2.bn.running_var', (64,), 'layer1.2.bn2.running_var', (64,),
    'block2.0.conv_bn1.conv.weight', (128, 64, 3, 3), 'layer2.0.conv1.weight', (128, 64, 3, 3),
    'block2.0.conv_bn1.bn.weight', (128,), 'layer2.0.bn1.weight', (128,),
    'block2.0.conv_bn1.bn.bias', (128,), 'layer2.0.bn1.bias', (128,),
    'block2.0.conv_bn1.bn.running_mean', (128,), 'layer2.0.bn1.running_mean', (128,),
    'block2.0.conv_bn1.bn.running_var', (128,), 'layer2.0.bn1.running_var', (128,),
    'block2.0.conv_bn2.conv.weight', (128, 128, 3, 3), 'layer2.0.conv2.weight', (128, 128, 3, 3),
    'block2.0.conv_bn2.bn.weight', (128,), 'layer2.0.bn2.weight', (128,),
    'block2.0.conv_bn2.bn.bias', (128,), 'layer2.0.bn2.bias', (128,),
    'block2.0.conv_bn2.bn.running_mean', (128,), 'layer2.0.bn2.running_mean', (128,),
    'block2.0.conv_bn2.bn.running_var', (128,), 'layer2.0.bn2.running_var', (128,),
    'block2.0.shortcut.conv.weight', (128, 64, 1, 1), 'layer2.0.downsample.0.weight', (128, 64, 1, 1),
    'block2.0.shortcut.bn.weight', (128,), 'layer2.0.downsample.1.weight', (128,),
    'block2.0.shortcut.bn.bias', (128,), 'layer2.0.downsample.1.bias', (128,),
    'block2.0.shortcut.bn.running_mean', (128,), 'layer2.0.downsample.1.running_mean', (128,),
    'block2.0.shortcut.bn.running_var', (128,), 'layer2.0.downsample.1.running_var', (128,),
    'block2.1.conv_bn1.conv.weight', (128, 128, 3, 3), 'layer2.1.conv1.weight', (128, 128, 3, 3),
    'block2.1.conv_bn1.bn.weight', (128,), 'layer2.1.bn1.weight', (128,),
    'block2.1.conv_bn1.bn.bias', (128,), 'layer2.1.bn1.bias', (128,),
    'block2.1.conv_bn1.bn.running_mean', (128,), 'layer2.1.bn1.running_mean', (128,),
    'block2.1.conv_bn1.bn.running_var', (128,), 'layer2.1.bn1.running_var', (128,),
    'block2.1.conv_bn2.conv.weight', (128, 128, 3, 3), 'layer2.1.conv2.weight', (128, 128, 3, 3),
    'block2.1.conv_bn2.bn.weight', (128,), 'layer2.1.bn2.weight', (128,),
    'block2.1.conv_bn2.bn.bias', (128,), 'layer2.1.bn2.bias', (128,),
    'block2.1.conv_bn2.bn.running_mean', (128,), 'layer2.1.bn2.running_mean', (128,),
    'block2.1.conv_bn2.bn.running_var', (128,), 'layer2.1.bn2.running_var', (128,),
    'block2.2.conv_bn1.conv.weight', (128, 128, 3, 3), 'layer2.2.conv1.weight', (128, 128, 3, 3),
    'block2.2.conv_bn1.bn.weight', (128,), 'layer2.2.bn1.weight', (128,),
    'block2.2.conv_bn1.bn.bias', (128,), 'layer2.2.bn1.bias', (128,),
    'block2.2.conv_bn1.bn.running_mean', (128,), 'layer2.2.bn1.running_mean', (128,),
    'block2.2.conv_bn1.bn.running_var', (128,), 'layer2.2.bn1.running_var', (128,),
    'block2.2.conv_bn2.conv.weight', (128, 128, 3, 3), 'layer2.2.conv2.weight', (128, 128, 3, 3),
    'block2.2.conv_bn2.bn.weight', (128,), 'layer2.2.bn2.weight', (128,),
    'block2.2.conv_bn2.bn.bias', (128,), 'layer2.2.bn2.bias', (128,),
    'block2.2.conv_bn2.bn.running_mean', (128,), 'layer2.2.bn2.running_mean', (128,),
    'block2.2.conv_bn2.bn.running_var', (128,), 'layer2.2.bn2.running_var', (128,),
    'block2.3.conv_bn1.conv.weight', (128, 128, 3, 3), 'layer2.3.conv1.weight', (128, 128, 3, 3),
    'block2.3.conv_bn1.bn.weight', (128,), 'layer2.3.bn1.weight', (128,),
    'block2.3.conv_bn1.bn.bias', (128,), 'layer2.3.bn1.bias', (128,),
    'block2.3.conv_bn1.bn.running_mean', (128,), 'layer2.3.bn1.running_mean', (128,),
    'block2.3.conv_bn1.bn.running_var', (128,), 'layer2.3.bn1.running_var', (128,),
    'block2.3.conv_bn2.conv.weight', (128, 128, 3, 3), 'layer2.3.conv2.weight', (128, 128, 3, 3),
    'block2.3.conv_bn2.bn.weight', (128,), 'layer2.3.bn2.weight', (128,),
    'block2.3.conv_bn2.bn.bias', (128,), 'layer2.3.bn2.bias', (128,),
    'block2.3.conv_bn2.bn.running_mean', (128,), 'layer2.3.bn2.running_mean', (128,),
    'block2.3.conv_bn2.bn.running_var', (128,), 'layer2.3.bn2.running_var', (128,),
    'block3.0.conv_bn1.conv.weight', (256, 128, 3, 3), 'layer3.0.conv1.weight', (256, 128, 3, 3),
    'block3.0.conv_bn1.bn.weight', (256,), 'layer3.0.bn1.weight', (256,),
    'block3.0.conv_bn1.bn.bias', (256,), 'layer3.0.bn1.bias', (256,),
    'block3.0.conv_bn1.bn.running_mean', (256,), 'layer3.0.bn1.running_mean', (256,),
    'block3.0.conv_bn1.bn.running_var', (256,), 'layer3.0.bn1.running_var', (256,),
    'block3.0.conv_bn2.conv.weight', (256, 256, 3, 3), 'layer3.0.conv2.weight', (256, 256, 3, 3),
    'block3.0.conv_bn2.bn.weight', (256,), 'layer3.0.bn2.weight', (256,),
    'block3.0.conv_bn2.bn.bias', (256,), 'layer3.0.bn2.bias', (256,),
    'block3.0.conv_bn2.bn.running_mean', (256,), 'layer3.0.bn2.running_mean', (256,),
    'block3.0.conv_bn2.bn.running_var', (256,), 'layer3.0.bn2.running_var', (256,),
    'block3.0.shortcut.conv.weight', (256, 128, 1, 1), 'layer3.0.downsample.0.weight', (256, 128, 1, 1),
    'block3.0.shortcut.bn.weight', (256,), 'layer3.0.downsample.1.weight', (256,),
    'block3.0.shortcut.bn.bias', (256,), 'layer3.0.downsample.1.bias', (256,),
    'block3.0.shortcut.bn.running_mean', (256,), 'layer3.0.downsample.1.running_mean', (256,),
    'block3.0.shortcut.bn.running_var', (256,), 'layer3.0.downsample.1.running_var', (256,),
    'block3.1.conv_bn1.conv.weight', (256, 256, 3, 3), 'layer3.1.conv1.weight', (256, 256, 3, 3),
    'block3.1.conv_bn1.bn.weight', (256,), 'layer3.1.bn1.weight', (256,),
    'block3.1.conv_bn1.bn.bias', (256,), 'layer3.1.bn1.bias', (256,),
    'block3.1.conv_bn1.bn.running_mean', (256,), 'layer3.1.bn1.running_mean', (256,),
    'block3.1.conv_bn1.bn.running_var', (256,), 'layer3.1.bn1.running_var', (256,),
    'block3.1.conv_bn2.conv.weight', (256, 256, 3, 3), 'layer3.1.conv2.weight', (256, 256, 3, 3),
    'block3.1.conv_bn2.bn.weight', (256,), 'layer3.1.bn2.weight', (256,),
    'block3.1.conv_bn2.bn.bias', (256,), 'layer3.1.bn2.bias', (256,),
    'block3.1.conv_bn2.bn.running_mean', (256,), 'layer3.1.bn2.running_mean', (256,),
    'block3.1.conv_bn2.bn.running_var', (256,), 'layer3.1.bn2.running_var', (256,),
    'block3.2.conv_bn1.conv.weight', (256, 256, 3, 3), 'layer3.2.conv1.weight', (256, 256, 3, 3),
    'block3.2.conv_bn1.bn.weight', (256,), 'layer3.2.bn1.weight', (256,),
    'block3.2.conv_bn1.bn.bias', (256,), 'layer3.2.bn1.bias', (256,),
    'block3.2.conv_bn1.bn.running_mean', (256,), 'layer3.2.bn1.running_mean', (256,),
    'block3.2.conv_bn1.bn.running_var', (256,), 'layer3.2.bn1.running_var', (256,),
    'block3.2.conv_bn2.conv.weight', (256, 256, 3, 3), 'layer3.2.conv2.weight', (256, 256, 3, 3),
    'block3.2.conv_bn2.bn.weight', (256,), 'layer3.2.bn2.weight', (256,),
    'block3.2.conv_bn2.bn.bias', (256,), 'layer3.2.bn2.bias', (256,),
    'block3.2.conv_bn2.bn.running_mean', (256,), 'layer3.2.bn2.running_mean', (256,),
    'block3.2.conv_bn2.bn.running_var', (256,), 'layer3.2.bn2.running_var', (256,),
    'block3.3.conv_bn1.conv.weight', (256, 256, 3, 3), 'layer3.3.conv1.weight', (256, 256, 3, 3),
    'block3.3.conv_bn1.bn.weight', (256,), 'layer3.3.bn1.weight', (256,),
    'block3.3.conv_bn1.bn.bias', (256,), 'layer3.3.bn1.bias', (256,),
    'block3.3.conv_bn1.bn.running_mean', (256,), 'layer3.3.bn1.running_mean', (256,),
    'block3.3.conv_bn1.bn.running_var', (256,), 'layer3.3.bn1.running_var', (256,),
    'block3.3.conv_bn2.conv.weight', (256, 256, 3, 3), 'layer3.3.conv2.weight', (256, 256, 3, 3),
    'block3.3.conv_bn2.bn.weight', (256,), 'layer3.3.bn2.weight', (256,),
    'block3.3.conv_bn2.bn.bias', (256,), 'layer3.3.bn2.bias', (256,),
    'block3.3.conv_bn2.bn.running_mean', (256,), 'layer3.3.bn2.running_mean', (256,),
    'block3.3.conv_bn2.bn.running_var', (256,), 'layer3.3.bn2.running_var', (256,),
    'block3.4.conv_bn1.conv.weight', (256, 256, 3, 3), 'layer3.4.conv1.weight', (256, 256, 3, 3),
    'block3.4.conv_bn1.bn.weight', (256,), 'layer3.4.bn1.weight', (256,),
    'block3.4.conv_bn1.bn.bias', (256,), 'layer3.4.bn1.bias', (256,),
    'block3.4.conv_bn1.bn.running_mean', (256,), 'layer3.4.bn1.running_mean', (256,),
    'block3.4.conv_bn1.bn.running_var', (256,), 'layer3.4.bn1.running_var', (256,),
    'block3.4.conv_bn2.conv.weight', (256, 256, 3, 3), 'layer3.4.conv2.weight', (256, 256, 3, 3),
    'block3.4.conv_bn2.bn.weight', (256,), 'layer3.4.bn2.weight', (256,),
    'block3.4.conv_bn2.bn.bias', (256,), 'layer3.4.bn2.bias', (256,),
    'block3.4.conv_bn2.bn.running_mean', (256,), 'layer3.4.bn2.running_mean', (256,),
    'block3.4.conv_bn2.bn.running_var', (256,), 'layer3.4.bn2.running_var', (256,),
    'block3.5.conv_bn1.conv.weight', (256, 256, 3, 3), 'layer3.5.conv1.weight', (256, 256, 3, 3),
    'block3.5.conv_bn1.bn.weight', (256,), 'layer3.5.bn1.weight', (256,),
    'block3.5.conv_bn1.bn.bias', (256,), 'layer3.5.bn1.bias', (256,),
    'block3.5.conv_bn1.bn.running_mean', (256,), 'layer3.5.bn1.running_mean', (256,),
    'block3.5.conv_bn1.bn.running_var', (256,), 'layer3.5.bn1.running_var', (256,),
    'block3.5.conv_bn2.conv.weight', (256, 256, 3, 3), 'layer3.5.conv2.weight', (256, 256, 3, 3),
    'block3.5.conv_bn2.bn.weight', (256,), 'layer3.5.bn2.weight', (256,),
    'block3.5.conv_bn2.bn.bias', (256,), 'layer3.5.bn2.bias', (256,),
    'block3.5.conv_bn2.bn.running_mean', (256,), 'layer3.5.bn2.running_mean', (256,),
    'block3.5.conv_bn2.bn.running_var', (256,), 'layer3.5.bn2.running_var', (256,),
    'block4.0.conv_bn1.conv.weight', (512, 256, 3, 3), 'layer4.0.conv1.weight', (512, 256, 3, 3),
    'block4.0.conv_bn1.bn.weight', (512,), 'layer4.0.bn1.weight', (512,),
    'block4.0.conv_bn1.bn.bias', (512,), 'layer4.0.bn1.bias', (512,),
    'block4.0.conv_bn1.bn.running_mean', (512,), 'layer4.0.bn1.running_mean', (512,),
    'block4.0.conv_bn1.bn.running_var', (512,), 'layer4.0.bn1.running_var', (512,),
    'block4.0.conv_bn2.conv.weight', (512, 512, 3, 3), 'layer4.0.conv2.weight', (512, 512, 3, 3),
    'block4.0.conv_bn2.bn.weight', (512,), 'layer4.0.bn2.weight', (512,),
    'block4.0.conv_bn2.bn.bias', (512,), 'layer4.0.bn2.bias', (512,),
    'block4.0.conv_bn2.bn.running_mean', (512,), 'layer4.0.bn2.running_mean', (512,),
    'block4.0.conv_bn2.bn.running_var', (512,), 'layer4.0.bn2.running_var', (512,),
    'block4.0.shortcut.conv.weight', (512, 256, 1, 1), 'layer4.0.downsample.0.weight', (512, 256, 1, 1),
    'block4.0.shortcut.bn.weight', (512,), 'layer4.0.downsample.1.weight', (512,),
    'block4.0.shortcut.bn.bias', (512,), 'layer4.0.downsample.1.bias', (512,),
    'block4.0.shortcut.bn.running_mean', (512,), 'layer4.0.downsample.1.running_mean', (512,),
    'block4.0.shortcut.bn.running_var', (512,), 'layer4.0.downsample.1.running_var', (512,),
    'block4.1.conv_bn1.conv.weight', (512, 512, 3, 3), 'layer4.1.conv1.weight', (512, 512, 3, 3),
    'block4.1.conv_bn1.bn.weight', (512,), 'layer4.1.bn1.weight', (512,),
    'block4.1.conv_bn1.bn.bias', (512,), 'layer4.1.bn1.bias', (512,),
    'block4.1.conv_bn1.bn.running_mean', (512,), 'layer4.1.bn1.running_mean', (512,),
    'block4.1.conv_bn1.bn.running_var', (512,), 'layer4.1.bn1.running_var', (512,),
    'block4.1.conv_bn2.conv.weight', (512, 512, 3, 3), 'layer4.1.conv2.weight', (512, 512, 3, 3),
    'block4.1.conv_bn2.bn.weight', (512,), 'layer4.1.bn2.weight', (512,),
    'block4.1.conv_bn2.bn.bias', (512,), 'layer4.1.bn2.bias', (512,),
    'block4.1.conv_bn2.bn.running_mean', (512,), 'layer4.1.bn2.running_mean', (512,),
    'block4.1.conv_bn2.bn.running_var', (512,), 'layer4.1.bn2.running_var', (512,),
    'block4.2.conv_bn1.conv.weight', (512, 512, 3, 3), 'layer4.2.conv1.weight', (512, 512, 3, 3),
    'block4.2.conv_bn1.bn.weight', (512,), 'layer4.2.bn1.weight', (512,),
    'block4.2.conv_bn1.bn.bias', (512,), 'layer4.2.bn1.bias', (512,),
    'block4.2.conv_bn1.bn.running_mean', (512,), 'layer4.2.bn1.running_mean', (512,),
    'block4.2.conv_bn1.bn.running_var', (512,), 'layer4.2.bn1.running_var', (512,),
    'block4.2.conv_bn2.conv.weight', (512, 512, 3, 3), 'layer4.2.conv2.weight', (512, 512, 3, 3),
    'block4.2.conv_bn2.bn.weight', (512,), 'layer4.2.bn2.weight', (512,),
    'block4.2.conv_bn2.bn.bias', (512,), 'layer4.2.bn2.bias', (512,),
    'block4.2.conv_bn2.bn.running_mean', (512,), 'layer4.2.bn2.running_mean', (512,),
    'block4.2.conv_bn2.bn.running_var', (512,), 'layer4.2.bn2.running_var', (512,),
    'logit.weight', (1000, 512), 'fc.weight', (1000, 512),
    'logit.bias', (1000,), 'fc.bias', (1000,),

]


###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


#############  resnext50 pyramid feature net #######################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel, channel, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(channel, out_channel, kernel_size=3, padding=1, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):
        z = F.relu(self.conv_bn1(x), inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z, inplace=True)
        return z


class ResNet34(nn.Module):

    def __init__(self, num_class=1000):
        super(ResNet34, self).__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            BasicBlock(64, 64, 64, stride=1, is_shortcut=False, ),
            *[BasicBlock(64, 64, 64, stride=1, is_shortcut=False, ) for i in range(1, 3)],
        )
        self.block2 = nn.Sequential(
            BasicBlock(64, 128, 128, stride=2, is_shortcut=True, ),
            *[BasicBlock(128, 128, 128, stride=1, is_shortcut=False, ) for i in range(1, 4)],
        )
        self.block3 = nn.Sequential(
            BasicBlock(128, 256, 256, stride=2, is_shortcut=True, ),
            *[BasicBlock(256, 256, 256, stride=1, is_shortcut=False, ) for i in range(1, 6)],
        )
        self.block4 = nn.Sequential(
            BasicBlock(256, 512, 512, stride=2, is_shortcut=True, ),
            *[BasicBlock(512, 512, 512, stride=1, is_shortcut=False, ) for i in range(1, 3)],
        )
        self.logit = nn.Linear(512, num_class)

    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        logit = self.logit(x)
        return logit


class Resnet34_classification(nn.Module):
    def __init__(self, num_class=4):
        super(Resnet34_classification, self).__init__()
        e = ResNet34()
        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4,
        ])
        e = None  # dropped
        self.feature = nn.Conv2d(512, 32, kernel_size=1)  # dummy conv for dim reduction
        self.logit = nn.Conv2d(32, num_class, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        for i in range(len(self.block)):
            x = self.block[i](x)
            # print(i, x.shape)

        x = F.dropout(x, 0.5, training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
        return logit


model_classification = Resnet34_classification()
model_classification.load_state_dict(
    torch.load('../input/clsification/00007500_model.pth', map_location=lambda storage, loc: storage), strict=True)


class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def sharpen(p, t=0.5):
    if t != 0:
        return p ** t
    else:
        return p

augment = ['null']


def get_classification_preds(net, test_loader):
    test_probability_label = []
    test_id = []

    net = net.cuda()
    for t, (fnames, images) in enumerate(tqdm(test_loader)):
        batch_size, C, H, W = images.shape
        images = images.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1:  # null
                logit = net(images)
                probability = torch.sigmoid(logit)

                probability_label = sharpen(probability, 0)
                num_augment += 1

            if 'flip_lr' in augment:
                logit = net(torch.flip(images, dims=[3]))
                probability = torch.sigmoid(logit)

                probability_label += sharpen(probability)
                num_augment += 1

            if 'flip_ud' in augment:
                logit = net(torch.flip(images, dims=[2]))
                probability = torch.sigmoid(logit)

                probability_label += sharpen(probability)
                num_augment += 1

            probability_label = probability_label / num_augment

        probability_label = probability_label.data.cpu().numpy()

        test_probability_label.append(probability_label)
        test_id.extend([i for i in fnames])

    test_probability_label = np.concatenate(test_probability_label)
    return test_probability_label, test_id


sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"
batch_size = 1

# mean and std
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
threshold_label = [0.50, 0.50, 0.50, 0.50, ]
probability_label, image_id = get_classification_preds(model_classification, testset)
predict_label = probability_label > np.array(threshold_label).reshape(1, 4, 1, 1)

image_id_class_id = []
encoded_pixel = []
for b in range(len(image_id)):
    for c in range(4):
        image_id_class_id.append(image_id[b] + '_%d' % (c + 1))
        if predict_label[b, c] == 0:
            rle = ''
        else:
            rle = '1 1'
        encoded_pixel.append(rle)

df_classification = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
