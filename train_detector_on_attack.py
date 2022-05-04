import argparse
import os
import copy


import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from utils.utils import set_random_seed, AverageMeter, print_and_write_log
from torch.utils.data.dataloader import DataLoader
from dataset import DetectDataset, DetectDatasetFromFolder
from detector_nets import get_detector
from transformations import DataAugmentation, DCT
from loss import detection_loss
import distutils.util
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

if __name__ == '__main__':
    print("Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Warning: This is a test copied from original detector!!!!!!!!!!!!!!!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr-decay', type=float, default=0.5)
    parser.add_argument('--lr-decay-patient', type=int, default=5)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-save', type=int, default=10)
    # parser.add_argument('--sample-interval', type=int, default=50)

    parser.add_argument(
        '--aug',
        type=distutils.util.strtobool,
        default='true',
        help='whether applying data augmentation as defense during model training'
    )

    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument('--feature-space', type=str, default='rgb')
    parser.add_argument('--det_net', type=str, default='xception')
    parser.add_argument('--fake-class', nargs='+')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train-length', type=int, default=0)
    parser.add_argument('--valid-length', type=int, default=0)

    args = parser.parse_args()
    args.outputs_dir = os.path.join(
        args.outputs_dir,
        f'x{args.im_size}-{args.det_net}-aug-{int(args.aug)}-{str().join(args.fake_class)}-{args.feature_space}'
    )

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    print("experimental folder = ", args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    set_random_seed(args.seed)

    model = get_detector(args.det_net).to(device)
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

    crit = detection_loss

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    sch = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.lr_decay_patient, verbose=True)

    train_trans = []
    valid_trans = []
    if args.aug:
        train_trans.append(DataAugmentation(prob=0.1,
                                            is_blur=True,
                                            is_jpeg=True,
                                            is_noise=False,
                                            is_jitter=False,
                                            is_geo=False,
                                            is_crop=False,
                                            is_rot=False,
                                            is_high=False),
                           )
    if args.feature_space == 'dct':
        train_trans.append(DCT())
        valid_trans.append(DCT())

    dataset = DetectDatasetFromFolder('./test_results/unet/after/',
                                      transform=train_trans,)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print('dataset slice: \n', dataset.__details__())

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

    best_epoch = 0
    best_epoch_acc = 0.5
    for epoch in range(args.num_epochs):

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(desc='Detector training phase =>',
                  total=(len(train_dataset) -
                         len(train_dataset) % args.batch_size),
                  ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch,
                                                    args.num_epochs-1))

            for data in train_dataloader:
                ims, gt_labels, _ = data

                ims = ims.to(device)
                gt_labels = gt_labels.to(device)

                pred_labels = model(ims)

                loss = crit(pred_labels, gt_labels)

                epoch_losses.update(loss.item(), len(ims))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(ims))

        train_mss = f'epoch {epoch}: loss: {epoch_losses.avg:.3f};'
        print_and_write_log(train_mss,
                            os.path.join(args.outputs_dir, 'logs.txt'))

        if epoch % args.num_save == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_acces = AverageMeter()
        epoch_v_losses = AverageMeter()

        correct = 0
        total = 0
        for data in valid_dataloader:
            ims, gt_labels, _ = data
            ims = ims.to(device)
            gt_labels = gt_labels.to(device)

            with torch.no_grad():
                pred_labels = model(ims)
            v_loss = crit(pred_labels, gt_labels)
            _, predicted = torch.max(pred_labels.data, 1)
            total += gt_labels.size(0)
            correct += (predicted == gt_labels).sum().item()
            epoch_v_losses.update(v_loss.item(), len(ims))

        sch.step(epoch_v_losses.avg)
        valid_mss = f'epoch {epoch}: eval acc: {(correct / total):.5f}; eval loss: {epoch_v_losses.avg:.3f};'
        print_and_write_log(valid_mss,
                            os.path.join(args.outputs_dir, 'logs.txt'))

        if (correct / total) >= best_epoch_acc:
            best_epoch_acc = correct / total
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                best_weights,
                os.path.join(args.outputs_dir, 'best.pth'))
    print_and_write_log(f'best model = epoch {best_epoch}',
                        os.path.join(args.outputs_dir, 'logs.txt'))

torch.cuda.empty_cache()
