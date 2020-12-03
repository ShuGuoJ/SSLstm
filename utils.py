import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat

def loadLabel(path):
    '''
    :param path:
    :return: 训练样本标签， 测试样本标签
    '''
    assert os.path.exists(path), '{},路径不存在'.format(path)
    # keys:{train_gt, test_gt}
    gt = loadmat(path)
    return gt['train_gt'], gt['test_gt']


def weight_init(m):
    if isinstance(m, nn.LSTM):
        init.normal_(m.weight_hh_l0, 0, 5e-3)
        init.normal_(m.weight_ih_l0, 0, 5e-3)
        init.constant_(m.bias_hh_l0, 0)
        init.constant_(m.bias_ih_l0, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 5e-3)
        init.constant_(m.bias, 0)
