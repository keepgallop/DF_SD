'''
@Description  : losses
@Author       : Chi Liu
@Date         : 2022-02-21 23:15:44
@LastEditTime : 2022-03-25 22:32:39
'''
import torch
import torch.nn as nn
from pytorch_msssim import SSIM

import lpips
from utils.focal_frequency_loss import FocalFrequencyLoss as FFL
from utils.psd_loss import PSDLoss as PSD

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1.0 - super(SSIM_Loss, self).forward(img1, img2)


def spatial_loss(pred, target, loss_type):
    assert loss_type in ["l2", "ssim", "perceptual"], "unknown loss type"
    if loss_type == "l2":
        criterion = nn.MSELoss()
    elif loss_type == 'ssim':
        criterion = SSIM_Loss(data_range=1.0,
                              size_average=True,
                              channel=3,
                              nonnegative_ssim=True)
    elif loss_type == "perceptual":
        criterion = lpips.LPIPS(net='vgg').to(device)

    if loss_type != "perceptual":
        err = criterion(pred, target)
    else:
        err = criterion(pred, target).mean()

    return err


def spectral_loss(pred, target, loss_type, is_reg, alpha):
    assert loss_type in ["fft", "focal_fft", "dct", "psd"], "unknown loss type"
    if loss_type == "psd":
        is_reg = False

    if loss_type == "fft":
        # alpha = 0 equals to normal FFT loss
        criterion = FFL(loss_weight=1.0, alpha=0.0, fre_mode='fft')
    elif loss_type == "focal_fft":
        criterion = FFL(loss_weight=1.0, alpha=1.0, fre_mode='fft')
    elif loss_type == "dct":
        criterion = FFL(loss_weight=1.0, alpha=0.0, fre_mode='dct')
    elif loss_type == "psd":
        criterion = PSD()

    if is_reg:
        reg = PSD()
        err = criterion(pred, target) + alpha * reg(pred, target)
    else:
        err = criterion(pred, target)
    return err


def detection_loss(pred, target):
    criterion = nn.BCELoss()
    return criterion(pred, target)
