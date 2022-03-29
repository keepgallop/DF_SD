'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-03-25 17:53:20
@LastEditTime : 2022-03-29 19:54:36
'''
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.transforms import GaussianBlur
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# class PSDLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def RGB2gray(self, rgb):
#         if rgb.size(1) == 3:
#             r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
#             gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#             return gray
#         elif rgb.size(1) == 1:
#             return rgb[:, 0, :, :]

#     def shift(self, x):
#         out = torch.zeros(x.size())
#         H, W = x.size(-2), x.size(-1)
#         out[:, :int(H / 2), :int(W / 2)] = x[:, int(H / 2):, int(W / 2):]
#         out[:, :int(H / 2), int(W / 2):] = x[:, int(H / 2):, :int(W / 2)]
#         out[:, int(H / 2):, :int(W / 2)] = x[:, :int(H / 2), int(W / 2):]
#         out[:, int(H / 2):, int(W / 2):] = x[:, :int(H / 2), :int(W / 2)]
#         return out

#     def azimuthalAverage(self, spe_ts):
#         """
#         Calculate the azimuthally averaged radial profile from a W*W spectrum spe_ts
#         """
#         x_index = torch.zeros_like(spe_ts)
#         W = spe_ts.size(0)
#         for i in range(W):
#             x_index[i, :] = torch.arange(0, W)
#         x_index = x_index.to(dtype=torch.int)
#         y_index = x_index.transpose(0, 1)
#         radius = torch.sqrt((x_index - 10 / 2)**2 + (y_index - 10 / 2)**2)
#         radius = radius.to(dtype=torch.int)
#         radius = torch.flatten(radius)
#         radius_bin = torch.bincount(radius)
#         ten_bin = torch.bincount(radius, spe_ts.flatten())
#         radial_prof = ten_bin / (radius_bin + 1e-10)
#         return radial_prof

#     def get_fft_feature(self, x_rgb):
#         """get 1d psd profile

#         Args:
#             x_rgb (torch.Tensor): RGB image batch tensor with size N*W*W

#         Returns:
#             torch.Tensor: 1d psd profile
#         """

#         epsilon = 1e-8

#         x_gray = self.RGB2gray(x_rgb)
#         fft = torch.rfft(x_gray, 2, onesided=False)
#         fft += epsilon
#         magnitude_spectrum = torch.log((torch.sqrt(fft[:, :, :, 0]**2 +
#                                                    fft[:, :, :, 1]**2)) +
#                                        epsilon)
#         magnitude_spectrum = self.shift(magnitude_spectrum)

#         out = []
#         for i in range(magnitude_spectrum.size(0)):
#             out.append(
#                 self.azimuthalAverage(
#                     magnitude_spectrum[i]).float().unsqueeze(0))
#         out = torch.cat(out, dim=0)
#         out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (
#             torch.max(out, dim=1, keepdim=True)[0] -
#             torch.min(out, dim=1, keepdim=True)[0])
#         return out

#     def forward(self, pred, target):
#         """
#         Args:
#             pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (torch.Tensor): of shape (N, C, H, W). Target tensor.
#         """
#         pred_psd = self.get_fft_feature(pred)
#         target_psd = self.get_fft_feature(target)
#         mse = nn.MSELoss()
#         return mse(pred_psd, target_psd).to(device)


class PSDLoss(nn.Module):

    f_cache = "spectralloss.{}.cache"

    ############################################################
    def __init__(self,
                 rows,
                 cols,
                 eps=1E-8,
                 cache=False,
                 is_avg=False,
                 is_thre=False,
                 is_filter=False):
        super(PSDLoss, self).__init__()
        self.img_size = rows
        self.is_avg = is_avg
        self.is_thre = is_thre
        self.is_filter = is_filter
        self.eps = eps
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(rows / 2)
        # number of cols after onesided fft
        cols_onesided = int(cols / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices(
            (rows, cols_onesided)) - np.array([[[shift_rows]], [[0]]])
        r = np.sqrt(r[0, :, :]**2 + r[1, :, :]**2)
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r, axes=0)
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        # repeat slice for each radius
        r = torch.from_numpy(r).expand(r_max + 1, -1, -1).to(torch.float)
        radius_to_slice = torch.arange(r_max + 1).view(-1, 1, 1)
        # generate mask for each radius
        mask = torch.where(
            r == radius_to_slice,
            torch.tensor(1, dtype=torch.float),
            torch.tensor(0, dtype=torch.float),
        )
        # how man entries for each radius?
        mask_n = torch.sum(mask, axis=(1, 2))
        mask = mask.unsqueeze(0)  # add batch dimension
        # normalization vector incl. batch dimension
        mask_n = (1 / mask_n.to(torch.float)).unsqueeze(0)
        self.criterion_l1 = torch.nn.L1Loss(reduction="sum")
        self.r_max = r_max
        self.vector_length = r_max + 1

        self.register_buffer("mask", mask)
        self.register_buffer("mask_n", mask_n)

        if cache and os.path.isfile(self.f_cache.format(self.img_size)):
            self._load_cache()
        else:
            self.is_fitted = False
            self.register_buffer("mean", None)

        if device is not None:
            self.to(device)
        self.device = device

    ############################################################
    def _save_cache(self):
        torch.save(self.mean, self.f_cache.format(self.img_size))
        self.is_fitted = True

    ############################################################
    def _load_cache(self):
        mean = torch.load(self.f_cache.format(self.img_size),
                          map_location=self.mask.device)
        self.register_buffer("mean", mean)
        self.is_fitted = True

    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################

    ############################################################
    def fft(self, data):
        if len(data.shape) == 4 and data.shape[1] == 3:
            # convert to grayscale
            data = (0.299 * data[:, 0, :, :] + 0.587 * data[:, 1, :, :] +
                    0.114 * data[:, 2, :, :])

        fft = torch.rfft(data, signal_ndim=2, onesided=True)
        # fft = torch.fft.rfft(data)
        # abs of complex
        fft_abs = torch.sum(fft**2, dim=3)
        fft_abs = fft_abs + self.eps
        fft_abs = 20 * torch.log(fft_abs)
        return fft_abs

    ############################################################
    def spectral_vector(self, data):
        """Assumes first dimension to be batch size."""
        fft = (self.fft(data).unsqueeze(1).expand(-1, self.vector_length, -1,
                                                  -1)
               )  # repeat img for each radius

        # apply mask and compute profile vector
        profile = (fft * self.mask).sum((2, 3))
        # normalize profile into [0,1]
        profile = profile * self.mask_n
        profile = profile - profile.min(1)[0].view(-1, 1)
        profile = profile / profile.max(1)[0].view(-1, 1)

        return profile

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_filter:
            filter = GaussianBlur(3, 0.2)
            pred = pred - filter(pred)
            target = target - filter(target)

        if self.is_thre:
            frequency_thre = int(0.1 * self.vector_length)
        else:
            frequency_thre = 0

        pred_profiles = self.spectral_vector(pred)[:, frequency_thre:]
        target_profiles = self.spectral_vector(target)[:, frequency_thre:]

        if self.is_avg:
            target_profiles_avg = target_profiles.mean(dim=0)
            target_profiles = torch.zeros_like(target_profiles)
            for i in range(target_profiles.shape[0]):
                target_profiles[i, :] = target_profiles_avg
        pred_profiles = Variable(pred_profiles, requires_grad=False).to(device)
        target_profiles = Variable(target_profiles,
                                   requires_grad=True).to(device)

        # criterion = nn.BCELoss()
        criterion = nn.MSELoss()
        return criterion(pred_profiles, target_profiles)