'''
@Description  : module test code
@Author       : Chi Liu
@Date         : 2022-03-25 18:40:38
@LastEditTime : 2022-03-27 20:55:42
'''
import sys

sys.path.append('../')
import torch
from loss import spatial_loss, spectral_loss

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

##############################################################################
#
#                             test dataset loader
#
##############################################################################
# from transformations import Wavelet

# wav = Wavelet()
# input_im = torch.rand((512, 512, 3))
# out = wav(input_im)
# print(out.shape)
##############################################################################
#
#                             test loss functions
#
##############################################################################

# test_input = torch.rand((10, 3, 128, 128)).to(device)
# test_output = torch.rand((10, 3, 128, 128)).to(device)
# for loss_type in ["l2", "ssim", "perceptual"]:
#     print('spatial', loss_type,
#           spatial_loss(test_output, test_input, loss_type=loss_type))
# for loss_type in ["fft", "focal_fft", "dct", "psd"]:
#     print(
#         'frequency w/o reg',
#         loss_type,
#         spectral_loss(
#             test_output,
#             test_input,
#             loss_type=loss_type,
#             is_reg=False,
#             alpha=1,
#         ),
#     )
#     print(
#         'frequency w/ reg',
#         loss_type,
#         spectral_loss(
#             test_output,
#             test_input,
#             loss_type=loss_type,
#             is_reg=True,
#             alpha=1,
#         ),
#     )

##############################################################################
#
#                              test attacker networks
#
##############################################################################
from attacker_nets import AE, VAE, RDN, UNet
from torchsummary import summary

input_im = torch.rand((10, 3, 128, 128)).to(device)

# ae_net = AE().to(device)
vae_net = VAE(input_dim=3, dim=100).to(device)
# rdn_net = RDN(scale_factor=2,
#               num_channels=3,
#               num_features=64,
#               growth_rate=64,
#               num_blocks=16,
#               num_layers=8).to(device)
# unet = UNet().to(device)

# ae_out = ae_net(input_im)
vae_out = vae_net(input_im)
# rdn_out = rdn_net(input_im)
# unet_out = unet(input_im)

# print(ae_out.shape, vae_out.shape, rdn_out.shape, unet_out.shape)
# summary(ae_net, (3, 128, 128))
