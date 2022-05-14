'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-01-13 16:28:04
@LastEditTime : 2022-05-10 23:22:59
'''
from torch import nn
import torch
from torchvision import models
import timm


class DeepCNN():
    def __init__(self, is_pretrain=True, n_class=2):
        super().__init__()

        self.is_pretrain = is_pretrain
        self.n_class = n_class
        # self.input_size = input_size
        self.resnet = timm.create_model('resnet101',
                                        pretrained=self.is_pretrain,
                                        num_classes=self.n_class)
        self.xception = timm.create_model('xception',
                                          pretrained=self.is_pretrain,
                                          num_classes=self.n_class)
        self.efficient = timm.create_model('efficientnet_b4',
                                           pretrained=self.is_pretrain,
                                           num_classes=self.n_class)
