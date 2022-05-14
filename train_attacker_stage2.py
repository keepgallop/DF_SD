import argparse
import os
import copy
from xmlrpc.client import boolean

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm

from utils.utils import set_random_seed, AverageMeter, grid_save, print_and_write_log, tensor2im

from torch.utils.data.dataloader import DataLoader

from dataset import AttackDataset2, Subset
from attacker_nets import get_attacker
from transformations import Wavelet, DataAugmentation
from torchsummary import summary
from loss import spatial_loss, spectral_loss
import functools
from metrics import psnr, ssim, lfd
import distutils.util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=0.5)
    parser.add_argument('--lr-decay-epoch', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-save', type=int, default=10)
    parser.add_argument('--sample-interval', type=int, default=50)

    parser.add_argument(
        '--aug',
        type=distutils.util.strtobool,
        default='true',
        help='whether applying data augmentation in training an attacker model'
    )

    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument(
        '--spa_loss',
        type=str,
        default='l1',
        choices=["l2", "l1", "ssim", "perceptual", "mix", "none"])
    parser.add_argument('--fre_loss',
                        type=str,
                        default='focal_fft',
                        choices=["fft", "focal_fft", "dct", "psd", "none"])
    parser.add_argument(
        '--reg',
        type=distutils.util.strtobool,
        default='true',
    )
    parser.add_argument('--lambda1', type=float,
                        default=1)  # weight for frequency loss
    parser.add_argument('--lambda2', type=float,
                        default=1)  # weight for frequency regularization loss
    parser.add_argument('--att_net', type=str, default='stage2_G')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--length', type=int, default=0)

    args = parser.parse_args()
    args.outputs_dir = os.path.join(
        args.outputs_dir,
        f'x{args.im_size}-{args.att_net}-{args.spa_loss}-{args.fre_loss}-lambda1-{args.lambda1}-lambda2-{args.lambda2}-reg-{int(args.reg)}-aug-{int(args.aug)}'
    )

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    print(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    set_random_seed(args.seed)

    model = get_attacker(args.att_net).to(device)
    # print(summary(model))

    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(
                args.weights_file,
                map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    spa_crit = functools.partial(spatial_loss, loss_type=args.spa_loss)
    fre_crit = functools.partial(spectral_loss,
                                 loss_type=args.fre_loss,
                                 is_reg=args.reg,
                                 alpha=args.lambda2,
                                 im_size=args.im_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.aug:
        train_trans = [
            DataAugmentation(prob=0.2,
                             is_blur=True,
                             is_jpeg=True,
                             is_noise=True,
                             is_jitter=False,
                             is_geo=False,
                             is_crop=False,
                             is_rot=False,
                             is_high=False),
        ]

    else:
        train_trans = []

    valid_trans = []

    dataset = AttackDataset2(args.data_path, length=args.length)
    train_indices = random.sample(range(len(dataset)), int(0.8 * len(dataset)))
    valid_indices = [i for i in range(len(dataset)) if i not in train_indices]

    train_dataset = Subset(dataset, train_indices,
                           transform=train_trans,)

    valid_dataset = Subset(dataset, valid_indices,
                           transform=valid_trans,)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    sample_step = 0
    best_epoch_lfd = 20

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (args.lr_decay
                                           ** (epoch // args.lr_decay_epoch))

        model.train()
        epoch_losses = AverageMeter()
        epoch_pix_losses = AverageMeter()
        epoch_fre_losses = AverageMeter()

        with tqdm(desc='Attacker training phase =>',
                  total=(len(train_dataset) -
                         len(train_dataset) % args.batch_size),
                  ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch,
                                                    args.num_epochs - 1))

            for data in train_dataloader:
                before_ims, after_ims, _ = data  # downsampled images (x1/2) and raw images

                before_ims = before_ims.to(device)
                after_ims = after_ims.to(device)

                out_ims = model(after_ims)
                # import ipdb; ipdb.set_trace()

                s_loss = spa_crit(out_ims, before_ims)
                f_loss = fre_crit(out_ims, before_ims)

                loss = s_loss + f_loss * args.lambda1

                epoch_losses.update(loss.item(), len(after_ims))
                epoch_pix_losses.update(s_loss.item(), len(after_ims))
                epoch_fre_losses.update(f_loss.item(), len(after_ims))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(after_ims))

                sample_step += 1

                if sample_step % args.sample_interval == 0:
                    grid_save(
                        torch.cat([before_ims, out_ims], 2),
                        os.path.join(args.outputs_dir,
                                     f'step_{sample_step}-epoch_{epoch}.png'))
        train_mss = f'epoch {epoch}: loss: {epoch_losses.avg:.3f}; pixel loss: {epoch_pix_losses.avg:.3f}; frequancy loss: {epoch_fre_losses.avg:.3f};'
        print_and_write_log(train_mss,
                            os.path.join(args.outputs_dir, 'logs.txt'))

        if (epoch + 1) % args.num_save == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_lfd = AverageMeter()
        before_ims = []
        after_ims = []
        out_ims = []
        for data in valid_dataloader:
            before_im, after_im, _ = data  # ! note: batch size = 1

            before_im = before_im.to(device)
            after_im = after_im.to(device)

            with torch.no_grad():
                out_im = model(after_im)
            before_ims.append(before_im)
            after_ims.append(after_im)
            out_ims.append(out_im)

            out_im = tensor2im(out_im[0])
            before_im = tensor2im(before_im[0])
            after_im = tensor2im(after_im[0])
            epoch_psnr.update(psnr(out_im, before_im), n=1)
            epoch_ssim.update(ssim(out_im, before_im), n=1)
            epoch_lfd.update(lfd(out_im, before_im), n=1)

        ran_inx = random.sample([i for i in range(len(before_ims))], 30)
        before_ims = [before_ims[i]
                      for i in ran_inx]  # sample 100 images for visualization
        after_ims = [after_ims[i]
                     for i in ran_inx]
        out_ims = [out_ims[i] for i in ran_inx]
        before_ims = torch.cat(before_ims, 3)  # cat in cols
        after_ims = torch.cat(after_ims, 3)
        out_ims = torch.cat(out_ims, 3)  # cat in cols
        grid_save(torch.cat([before_ims, after_ims, out_ims], 2),
                  os.path.join(args.outputs_dir, f'epoch_{epoch}-valid.png'))
        valid_mss = f'epoch {epoch}: eval psnr: {epoch_psnr.avg:.3f}; eval ssim: {epoch_ssim.avg:.3f}; eval lfd: {epoch_lfd.avg:.3f};'
        print_and_write_log(valid_mss,
                            os.path.join(args.outputs_dir, 'logs.txt'))

        if epoch_lfd.avg <= best_epoch_lfd:
            best_epoch_lfd = epoch_lfd.avg
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                best_weights,
                os.path.join(args.outputs_dir, 'best.pth'))
    print_and_write_log(f'best model = epoch {best_epoch}',
                        os.path.join(args.outputs_dir, 'logs.txt'))

torch.cuda.empty_cache()
