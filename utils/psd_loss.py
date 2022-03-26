'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-03-25 17:53:20
@LastEditTime : 2022-03-25 22:21:02
'''
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class PSDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def RGB2gray(self, rgb):
        if rgb.size(1) == 3:
            r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        elif rgb.size(1) == 1:
            return rgb[:, 0, :, :]

    def shift(self, x):
        out = torch.zeros(x.size())
        H, W = x.size(-2), x.size(-1)
        out[:, :int(H / 2), :int(W / 2)] = x[:, int(H / 2):, int(W / 2):]
        out[:, :int(H / 2), int(W / 2):] = x[:, int(H / 2):, :int(W / 2)]
        out[:, int(H / 2):, :int(W / 2)] = x[:, :int(H / 2), int(W / 2):]
        out[:, int(H / 2):, int(W / 2):] = x[:, :int(H / 2), :int(W / 2)]
        return out

    def azimuthalAverage(self, spe_ts):
        """
        Calculate the azimuthally averaged radial profile from a W*W spectrum spe_ts
        """
        x_index = torch.zeros_like(spe_ts)
        W = spe_ts.size(0)
        for i in range(W):
            x_index[i, :] = torch.arange(0, W)
        x_index = x_index.to(dtype=torch.int)
        y_index = x_index.transpose(0, 1)
        radius = torch.sqrt((x_index - 10 / 2)**2 + (y_index - 10 / 2)**2)
        radius = radius.to(dtype=torch.int)
        radius = torch.flatten(radius)
        radius_bin = torch.bincount(radius)
        ten_bin = torch.bincount(radius, spe_ts.flatten())
        radial_prof = ten_bin / (radius_bin + 1e-10)
        return radial_prof

    def get_fft_feature(self, x_rgb):
        """get 1d psd profile

        Args:
            x_rgb (torch.Tensor): RGB image batch tensor with size N*W*W

        Returns:
            torch.Tensor: 1d psd profile
        """

        epsilon = 1e-8

        x_gray = self.RGB2gray(x_rgb)
        fft = torch.rfft(x_gray, 2, onesided=False)
        fft += epsilon
        magnitude_spectrum = torch.log((torch.sqrt(fft[:, :, :, 0]**2 +
                                                   fft[:, :, :, 1]**2)) +
                                       epsilon)
        magnitude_spectrum = self.shift(magnitude_spectrum)

        out = []
        for i in range(magnitude_spectrum.size(0)):
            out.append(
                self.azimuthalAverage(
                    magnitude_spectrum[i]).float().unsqueeze(0))
        out = torch.cat(out, dim=0)
        out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (
            torch.max(out, dim=1, keepdim=True)[0] -
            torch.min(out, dim=1, keepdim=True)[0])
        return out

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
        """
        pred_psd = self.get_fft_feature(pred)
        target_psd = self.get_fft_feature(target)
        mse = nn.MSELoss()
        return mse(pred_psd, target_psd).to(device)
