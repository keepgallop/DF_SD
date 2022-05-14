'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-01-14 17:20:38
@LastEditTime : 2022-05-09 21:44:56
'''

from skimage.restoration import (denoise_wavelet, estimate_sigma)
import glob
import os

import random

from PIL import Image as I

import torchvision.transforms as transforms
from io import BytesIO
import numpy as np
from functools import reduce
from scipy.fft import dct
import pywt


class DataAugmentation(object):
    """Data augmentation strategies (aka. Defense strategies). All functions get PIL image input and return to PIL image.
    """

    def __init__(
        self,
        prob=0.2,
        is_blur=True,
        is_jpeg=True,
        is_noise=True,
        is_jitter=True,
        is_geo=True,
        is_crop=True,
        is_rot=True,
        is_high=False,
    ):

        self.prob = prob
        self.strategy_set = []
        if is_blur:
            self.strategy_set.append(self.gaussian_blur)
        if is_jpeg:
            self.strategy_set.append(self.jpeg_compression)
        if is_noise:
            self.strategy_set.append(self.gaussian_noise)
        if is_jitter:
            self.strategy_set.append(self.color_jitter)
        if is_geo:
            self.strategy_set.append(self.geometry)
        if is_crop:
            self.strategy_set.append(self.cropping)
        if is_rot:
            self.strategy_set.append(self.rotation)
        if is_high:
            self.strategy_set.append(self.high_level_color)

    def __call__(self, img):
        return self.random_combine()(img)

    def random_combine(self):
        def compose(f, g):
            return lambda x: g(f(x))

        func_list = [self.no_defense]
        for func in self.strategy_set:
            prob_ = random.uniform(0.0, 1.0)
            if prob_ <= self.prob:
                func_list.append(func)
        return reduce(compose, func_list, lambda x: x)

    def no_defense(self, img):
        return img

    def gaussian_blur(self, img):
        kernel_size = random.choice([1, 3, 5, 7, 9])
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2)),
            transforms.ToPILImage(),
        ])
        img = tf(img)
        return img

    def jpeg_compression(self, img):
        quality = random.randint(10, 75)
        buffer = BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        img_jpeg = I.open(buffer)
        return img_jpeg

    def gaussian_noise(self, img):
        # variance from U[5.0,20.0]
        img = np.array(img)
        variance = np.random.uniform(low=5.0, high=20.0)
        img = np.copy(img).astype(np.float64)
        noise = variance * np.random.randn(*img.shape)
        img += noise
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        img = I.fromarray(img)
        return img

    def color_jitter(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.2,
                                   hue=0.2),
            transforms.ToPILImage(),
        ])
        img = tf(img)
        return img

    def geometry(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=self.prob),
            transforms.RandomVerticalFlip(p=self.prob),
            transforms.ToPILImage(),
        ])
        img = tf(img)
        return img

    def high_level_color(self, img):
        """Danger to use: May significantly change the spectrum
        """
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomPosterize(2, p=self.prob),
            # transforms.RandomSolarize(threshold, p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=self.prob),
            transforms.RandomAutocontrast(p=self.prob),
            transforms.RandomEqualize(p=self.prob),
            transforms.ToPILImage(),
        ])
        img = tf(img)
        return img

    def cropping(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(img.size,
                                         scale=(0.8, 1.0),
                                         ratio=(0.75, 1.3333333333333333)),
            transforms.ToPILImage(),
        ])
        img = tf(img)
        return img

    def rotation(self, img):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=10),
            transforms.ToPILImage(),
        ])
        img = tf(img)
        return img


class DCT(object):
    """2D-DCT transformation. All functions get PIL image input and return to np.array.
    """

    def __init__(self, is_norm=True, is_log_scale=True):
        self.func_list = [self.dct2]
        if is_log_scale:
            self.func_list.append(self.log_scale)
        if is_norm:
            self.func_list.append(self.normalize)

    def __call__(self, img):
        return self.combine()(img).astype(np.float32)

    def combine(self):
        def compose(f, g):
            return lambda x: g(f(x))

        return reduce(compose, self.func_list, lambda x: x)

    def dct2(self, arr):
        arr = np.array(arr)
        arr = dct(arr, type=2, norm="ortho", axis=0)
        arr = dct(arr, type=2, norm="ortho", axis=1)
        return arr.astype(np.float32)

    def normalize(self, arr):
        arr = (np.max(arr, axis=(0, 1)) - arr) / (np.max(arr, axis=(0, 1)) -
                                                  (np.min(arr, axis=(0, 1))))
        return arr.astype(np.float32)

    def log_scale(self, arr, epsilon=1e-12):
        """Log scale the input array.
        """
        arr = np.abs(arr)
        arr += epsilon  # no zero in log
        arr = np.log(arr)
        return arr.astype(np.float32)


class Wavelet(object):
    """wavelet transformation. All functions get PIL image input and return to np.array.
    """

    def __init__(self, is_norm=False):
        self.is_norm = is_norm

    def __call__(self, img):
        if self.is_norm:
            return self.normalize(self.wavelet(img)).astype(np.float32)
        else:
            return self.wavelet(img).astype(np.float32)

    def wavelet(self, arr):
        arr = np.array(arr) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(arr, 'haar', axes=(0, 1))
        return cA

    def normalize(self, arr):
        arr = (np.max(arr, axis=(0, 1)) - arr) / (np.max(arr, axis=(0, 1)) -
                                                  (np.min(arr, axis=(0, 1))))
        return arr


class WaveletFilterResize(object):
    """wavelet filter and then resize to downsize
    """

    def __init__(self, is_norm=False, sigma=0.5):
        self.is_norm = is_norm
        self.sigma = sigma

    def __call__(self, img):
        if self.is_norm:
            return self.normalize(self.wfr(img)).astype(np.float32)
        else:
            return self.wfr(img).astype(np.float32)

    def wfr(self, arr):
        arr = np.array(arr)
        fil_arr = denoise_wavelet(arr, multichannel=True, convert2ycbcr=True,
                                  method='BayesShrink', mode='soft', sigma=self.sigma)*255
        fil_arr = fil_arr.astype(np.uint8)
        fil_arr = I.fromarray(fil_arr)
        w, h = fil_arr.size[:2]
        fil_arr = fil_arr.resize((int(w/2), int(h/2)))
        return np.array(fil_arr)

    def normalize(self, arr):
        arr = (arr - np.min(arr, axis=(0, 1))) / (np.max(arr, axis=(0, 1)) -
                                                  (np.min(arr, axis=(0, 1))))
        return arr


class FrequencyExchange(object):
    def __init__(self, sources, target, thre):
        """Exchange the high-frequency component of the target image with that of the source image(s). 

        Args:
            source: a batch of images
            target: target image
            thre: frequency threshold
        """
        assert isinstance(sources, list)
        self.sources = sources
        self.target = target
        self.thre = thre
        self.shape = target.shape
        self.mask = self.circular_mask()

    def __call__(self):
        return self.hfe()

    def circular_mask(self):
        h, w, _ = self.shape
        Y, X = np.ogrid[:h, :w]
        center = (int(w / 2), int(h / 2))
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= self.thre
        return np.dstack([mask, mask, mask])

    def hfe(self):
        avg_s_high_fre = []
        for source in self.sources:
            s_fre = np.fft.fft2(source, axes=(0, 1))
            s_fre_shift = np.fft.fftshift(s_fre)
            s_low_fre = s_fre_shift * self.mask
            s_high_fre = s_fre_shift * (1 - self.mask)
            avg_s_high_fre.append(s_high_fre)
        s_high_fre = np.mean(np.stack(avg_s_high_fre, axis=0), axis=0)

        t_fre = np.fft.fft2(self.target, axes=(0, 1))
        t_fre_shift = np.fft.fftshift(t_fre)
        t_low_fre = t_fre_shift * self.mask
        t_high_fre = t_fre_shift * (1 - self.mask)

        t_exchange_fre = t_low_fre + s_high_fre
        # t_exchange_fre = t_low_fre
        t_exchange_im = np.abs(np.fft.ifft2(np.fft.ifftshift(t_exchange_fre), axes=(0, 1)))
        view = np.clip(t_exchange_im, 0, 255)
        view = view.astype('uint8')
        return view
