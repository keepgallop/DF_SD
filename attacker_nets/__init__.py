'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-03-28 11:40:22
@LastEditTime : 2022-05-03 16:19:39
'''
from .ae import AE
from .rdn import RDN
from .unet import UNet
from .vae import VAE


def get_attacker(name):
    if name == 'unet':
        return UNet()
    elif name == 'rdn':
        return RDN(nDenselayer=3, nFeat=64, scale=2)
    elif name == 'ae':
        return AE()
    elif name == 'vae':
        return VAE(input_dim=3, dim=128)
    else:
        raise TypeError('attacker network not supported!')
