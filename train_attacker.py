import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from utils.utils import set_random_seed, AverageMeter, grid_save, calc_psnr, convert_rgb_to_y, denormalize

from torch.utils.data.dataloader import DataLoader
from dataset import AttackDataset
from attacker_nets import AE, VAE, RDN, UNet
from transformations import Wavelet, DataAugmentation
from torchsummary import summary
from loss import spatial_loss, spectral_loss
import functools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)

    # parser.add_argument('--eval-file', type=str, required=True)

    parser.add_argument('--weights-file', type=str)
    # parser.add_argument('--num-features', type=int, default=64)
    # parser.add_argument('--growth-rate', type=int, default=64)
    # parser.add_argument('--num-blocks', type=int, default=16)
    # parser.add_argument('--num-layers', type=int, default=8)
    # parser.add_argument('--scale', type=int, default=4)
    # parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=0.5)
    parser.add_argument('--lr-decay-epoch', type=int, default=10)

    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-save', type=int, default=10)
    parser.add_argument('--sample-interval', type=int, default=200)

    parser.add_argument(
        '--augment',
        action='store_true',
        help='whether applying data augmentation in training an attacker model'
    )
    # parser.add_argument('--completion', action='store_true', help='completion')
    # parser.add_argument('--colorization',
    #                     action='store_true',
    #                     help='colorization')

    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument('--spa_loss',
                        type=str,
                        default='l2',
                        choices=["l2", "ssim", "perceptual", "none"])
    parser.add_argument('--fre_loss',
                        type=str,
                        default='fft',
                        choices=["fft", "focal_fft", "dct", "psd", "none"])
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--lambda1', type=float,
                        default=1)  # weight for frequency loss
    parser.add_argument('--lambda2', type=float,
                        default=1)  # weight for frequency regularization loss
    parser.add_argument('--att_net', type=str, default='unet')
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()

    args.outputs_dir = os.path.join(
        args.outputs_dir,
        f'x{args.im_size}-{args.spa_loss}-{args.fre_loss}-{int(args.regularization)}'
    )

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    set_random_seed(args.seed)

    if args.att_net == 'unet':
        model = UNet().to(device)
    elif args.att_net == 'rdn':
        num_features = 128 if args.im_size == 128 else 256
        model = RDN(num_features=num_features).to(device)
    elif args.att_net == 'ae':
        model = AE().to(device)
    elif args.att_net == 'vae':
        dim = 128 if args.im_size == 128 else 256
        model = VAE(input_dim=3, dim=dim).to(device)
    else:
        raise TypeError('attacker network not supported!')

    print(summary(model))

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
                                 alpha=args.lambda2)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.augment:
        train_trans = [DataAugmentation(), Wavelet(is_norm=True)]
    else:
        train_trans = Wavelet(is_norm=True)
    valid_trans = Wavelet(is_norm=True)
    train_dataset = AttackDataset(
        args.data_csv_file,
        transform=train_trans,
        allocation='train',
    )

    valid_dataset = AttackDataset(
        args.data_csv_file,
        transform=valid_trans,
        allocation='valid',
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    sample_step = 0
    #todo metrics define  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (args.lr_decay
                                           **(epoch // args.lr_decay_epoch))

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(desc='Attacker training phase =>',
                  total=(len(train_dataset) -
                         len(train_dataset) % args.batch_size),
                  ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch,
                                                    args.num_epochs - 1))

            for data in train_dataloader:
                ds_ims, raw_ims, _ = data  # downsampled images (x1/2) and raw images

                ds_ims = ds_ims.to(device)
                raw_ims = raw_ims.to(device)

                out_ims = model(ds_ims)
                #import ipdb; ipdb.set_trace()

                s_loss = spa_crit(out_ims, raw_ims) if not spa_crit else 0
                f_loss = fre_crit(out_ims, raw_ims) if not fre_crit else 0

                loss = s_loss + f_loss * args.lambda1

                epoch_losses.update(loss.item(), len(ds_ims))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(ds_ims))

                sample_step += 1

                if sample_step % args.sample_interval == 0:
                    grid_save(
                        raw_ims,
                        os.path.join(
                            args.outputs_dir,
                            f'step_{sample_step}-epoch_{epoch}-raw.png'))
                    grid_save(
                        out_ims,
                        os.path.join(
                            args.outputs_dir,
                            f'step_{sample_step}-epoch_{epoch}-out.png'))

        if (epoch + 1) % args.num_save == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        #todo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            preds = convert_rgb_to_y(denormalize(preds.squeeze(0)),
                                     dim_order='chw')
            labels = convert_rgb_to_y(denormalize(labels.squeeze(0)),
                                      dim_order='chw')

            preds = preds[args.scale:-args.scale, args.scale:-args.scale]
            labels = labels[args.scale:-args.scale, args.scale:-args.scale]

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    #print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    #torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
