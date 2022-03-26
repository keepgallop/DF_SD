'''
@Description  : module test code
@Author       : Chi Liu
@Date         : 2022-03-25 18:40:38
@LastEditTime : 2022-03-25 22:40:42
'''
import sys

sys.path.append('../')
import torch as T
from loss import spatial_loss, spectral_loss

if T.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

############### data loader test start ##################

############### data loader test end ##################

############### loss function test start ##################

test_input = T.rand((10, 3, 128, 128)).to(device)
test_output = T.rand((10, 3, 128, 128)).to(device)
for loss_type in ["l2", "ssim", "perceptual"]:
    print('spatial', loss_type,
          spatial_loss(test_output, test_input, loss_type=loss_type))
for loss_type in ["fft", "focal_fft", "dct", "psd"]:
    print(
        'frequency w/o reg',
        loss_type,
        spectral_loss(
            test_output,
            test_input,
            loss_type=loss_type,
            is_reg=False,
            alpha=1,
        ),
    )
    print(
        'frequency w/ reg',
        loss_type,
        spectral_loss(
            test_output,
            test_input,
            loss_type=loss_type,
            is_reg=True,
            alpha=1,
        ),
    )

############### loss function test end ##################
