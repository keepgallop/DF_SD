'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-03-28 11:40:22
@LastEditTime : 2022-05-09 20:01:10
'''
import os
import random

import numpy as np
import torch
import torch.nn.init as init
from torchvision.utils import make_grid, save_image
import numpy as np
from torchvision import transforms
from utils import psd_loss
import glob
import matplotlib.pyplot as plt
from PIL import Image as I


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def print_and_write_log(log_file, message):
    print(message)
    with open(log_file, 'a+') as f:
        f.write('%s\n' % message)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grid_save(ims, p):
    grid = make_grid(ims)
    save_image(grid, p)


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Args:
        input_image (torch.tensor): the input tensor array.
        imtype (type): the desired type of the converted numpy image array.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.detach()
        else:
            return input_image
        image_tensor = denormalize(image_tensor)
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def print_and_write_log(message, log_file=None):
    """Print message and write to a log file.

    Args:
        message (str): The message to print out and log.
        log_file (str, optional): Path to the log file. Default: None.
    """
    print(message)
    if log_file is not None:
        with open(log_file, 'a+') as f:
            f.write('%s\n' % message)


def save_batch_image(ims, from_name_list, to_path):
    """save batched torch.tensor images from graph
    """
    for im, name in zip(ims, from_name_list):
        name = name.split('/')[-1]
        im_to_path = os.path.join(to_path, name)
        save_image(im, im_to_path)


def visualize_spectral_distribution(bp='', ap='', num=1000, types=['real', 'progan', 'stgan', 'mmdgan'], shape=(128, 128)):
    """visualize spectral distribution before and after attack

    Args:
        bp (str): before path
        ap (str): after path
        num (int): numbers 
        types (list): selected labels
    """
    PSD = psd_loss.PSDLoss(shape[0], shape[1])

    b_cans = glob.glob(os.path.join(bp + "*.png"))
    a_cans = glob.glob(os.path.join(ap + "*.png"))
    random.shuffle(b_cans)
    random.shuffle(a_cans)
    plt.figure(figsize=(15, 10))
    for t in types:
        bt_list = []
        at_list = []
        for i, j in zip(b_cans, a_cans):
            if t in i:
                bt_list.append(i)
            if t in j:
                at_list.append(j)
        bt_list = bt_list[:num]
        at_list = at_list[:num]
        # print(t, len(bt_list), len(at_list), bt_list[:10], at_list[:10])
        b_ims = torch.from_numpy(np.array([np.array(I.open(i)) for i in bt_list])).to('cuda:0')
        a_ims = torch.from_numpy(np.array([np.array(I.open(i)) for i in at_list])).to('cuda:0')

        b_ims = b_ims.permute(0, 3, 1, 2).to(torch.float)
        a_ims = a_ims.permute(0, 3, 1, 2).to(torch.float)

        b_spectrum = PSD.spectral_vector(b_ims).cpu()
        a_spectrum = PSD.spectral_vector(a_ims).cpu()
        plt.plot(b_spectrum.mean(0), '-.', label=f'{t}-before')
        plt.plot(a_spectrum.mean(0), '-', label=f'{t}-after')
    plt.legend()


def average_spectra(im_list, number):
    ims = np.array([np.array(Image.open(i).convert('L')) for i in im_list[0:number]])
    ffts = np.fft.fftshift(np.fft.fft2(ims))
    fft_mean = np.mean(abs(ffts), axis=0)
    fft_mean = np.log(fft_mean + 1e-10)
    return fft_mean
