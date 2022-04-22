'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-03-28 11:40:22
@LastEditTime : 2022-04-22 16:32:28
'''
from .dnn import DeepCNN


def get_detector(name):
    DNN = DeepCNN()
    if name == 'xception':
        return DNN.xception
    elif name == 'resnet':
        return DNN.resnet
    elif name == 'efficientnet':
        return DNN.efficient
    