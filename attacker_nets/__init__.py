'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-03-28 11:40:22
@LastEditTime : 2022-05-10 22:00:35
'''
from .ae import AE
from .rdn import RDN
from .unet import UNet
from .vae import VAE
from .stage2_G import UNet_S2
from .vdsr import VDSR
from .edsr import EDSR
from .stage2_G1 import RDN_S2


def get_attacker(name):
    if name == 'unet':
        return UNet()
    elif name == 'rdn':
        return RDN(nDenselayer=3, nFeat=64, scale=2)
    elif name == 'ae':
        return AE()
    elif name == 'vae':
        return VAE(input_dim=3, dim=128)
    elif name == 'stage2_unet':
        return UNet_S2()
    elif name == 'stage2_rdn':
        return RDN_S2(nDenselayer=5, nFeat=64, scale=1)
    elif name == 'vdsr':
        return VDSR()
    elif name == 'edsr':
        return EDSR()
    else:
        raise TypeError('attacker network not supported!')
