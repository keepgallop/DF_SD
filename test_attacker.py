import argparse
import os
import copy


import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from utils.utils import set_random_seed, AverageMeter, print_and_write_log, save_batch_image, tensor2im, grid_save
from torch.utils.data.dataloader import DataLoader
from dataset import AttackDataset, DetectDatasetFromFolder
from detector_nets import get_detector
from attacker_nets import get_attacker
from transformations import DCT, Wavelet
import distutils.util
import numpy as np
from metrics import psnr, ssim, lfd
import pandas as pd


def ana_results(results):
    mss = ''
    for fake_class in ['real', 'progan', 'mmdgan', 'stgan', 'stargan', 'crgan', 'sngan']:
        correct = 0
        total = 0
        for i in range(len(results)):
            if fake_class in att_results['name'][i]:
                # print(att_results['gt'][i])
                total += 1
                if results['gt'][i] == results['pred'][i]:
                    correct += 1
        mss += f'{fake_class} images: eval acc: {(correct / total):.5f};\n'
    return mss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--attacker-ckpt', type=str)
    parser.add_argument('--detector-ckpt', type=str)
    parser.add_argument('--det_net', type=str, default='xception')
    parser.add_argument('--att_net', type=str, default='rdn')
    parser.add_argument('--allocation', type=str, default='test')
    parser.add_argument('--feature-space', type=str, default='rgb')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument(
        '--new-attack',
        type=distutils.util.strtobool,
        default='false',
    )

    args = parser.parse_args()

    det_nat_name = args.detector_ckpt.split('/')[-2].split('-')[-2]
    result_outputs_dir = os.path.join(
        args.outputs_dir,
        f'x{args.im_size}-{args.det_net}-{det_nat_name}-{args.att_net}-{args.feature_space}'
    )

    if not os.path.exists(result_outputs_dir):
        os.makedirs(result_outputs_dir)
    print(result_outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    #### attack process ####
    attsample_path = os.path.join(args.outputs_dir, args.att_net)
    before_save_path = os.path.join(attsample_path, 'before')  # save raw images
    after_save_path = os.path.join(attsample_path, 'after')  # save attack samples

    if args.new_attack:
        ####  load test data ####

        test_att_trans = Wavelet(is_norm=True)

        test_att_dataset = AttackDataset(args.data_csv_file,
                                         transform=test_att_trans,
                                         allocation=args.allocation,
                                         length=0)

        test_att_dataloader = DataLoader(dataset=test_att_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,)

        #### load pretained ckpts #####
        att_model = get_attacker(args.att_net).to(device)
        att_model.load_state_dict(torch.load(args.attacker_ckpt))

        print("========> Creating new attack samples.")
        if not os.path.exists(before_save_path):
            os.makedirs(before_save_path)
        if not os.path.exists(after_save_path):
            os.makedirs(after_save_path)
        with torch.no_grad():
            att_model.eval()
            epoch_psnr = AverageMeter()
            epoch_ssim = AverageMeter()
            epoch_lfd = AverageMeter()

            raw_ims = []
            out_ims = []
            im_names = []
            with tqdm(desc='Attack sample generation phase =>',
                      total=(len(test_att_dataset) -
                             len(test_att_dataset) % args.batch_size), ncols=80) as t:
                t.set_description('Attack sample generation Progress:')
                for data in test_att_dataloader:
                    ds_im, raw_im, im_name = data  # note: batch size = 1

                    im_name = [i.lstrip('./dataset/').replace('/', '-') for i in im_name]

                    ds_im = ds_im.to(device)
                    raw_im = raw_im.to(device)
                    out_im = att_model(ds_im)

                    save_batch_image(raw_im, im_name, before_save_path)
                    save_batch_image(out_im, im_name, after_save_path)

                    for i, j in zip(out_im, raw_im):
                        i = tensor2im(i)
                        j = tensor2im(j)
                        epoch_psnr.update(psnr(i, j), n=1)
                        epoch_ssim.update(ssim(i, j), n=1)
                        epoch_lfd.update(lfd(i, j), n=1)
                    t.update(len(ds_im))
            test_mss = f'eval psnr: {epoch_psnr.avg:.3f}; eval ssim: {epoch_ssim.avg:.3f}; eval lfd: {epoch_lfd.avg:.3f};'
            print_and_write_log(test_mss,
                                os.path.join(result_outputs_dir, 'att_logs.txt'))
        torch.cuda.empty_cache()
        print("========> Creation over.")
    else:
        print(f"========> Detect attack samples in {after_save_path}.")
        print(f"========> Detect raw samples in {before_save_path}.")

    #### detect process ####
    print("========> Detecting attack samples.")
    det_model = get_detector(args.det_net).to(device)
    det_model.load_state_dict(torch.load(args.detector_ckpt))

    test_det_trans = None if args.feature_space == 'rgb' else DCT()
    test_det_dataset_before = DetectDatasetFromFolder(before_save_path,
                                                      transform=test_det_trans)
    test_det_dataset_after = DetectDatasetFromFolder(after_save_path,
                                                     transform=test_det_trans)

    test_det_dataloader_before = DataLoader(dataset=test_det_dataset_before,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,)
    test_det_dataloader_after = DataLoader(dataset=test_det_dataset_after,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,)

    with torch.no_grad():
        det_model.eval()

        epoch_acces = AverageMeter()
        correct = 0
        total = 0
        raw_results = {'name': [],
                       'ori': [],
                       'gt': [],
                       'pred': []}
        with tqdm(desc='Raw sample detection phase =>',
                  total=(len(test_det_dataset_before) -
                         len(test_det_dataset_before) % args.batch_size), ncols=80) as t:
            t.set_description('Raw sample detection Progress:')
            for data in test_det_dataloader_before:
                ims, gt_labels, im_names = data
                ims = ims.to(device)
                gt_labels = gt_labels.to(device)
                pred_labels = det_model(ims)
                _, predicted = torch.max(pred_labels.data, 1)
                total += gt_labels.size(0)
                correct += (predicted == gt_labels).sum().item()

                for i, j, k, l in zip(im_names, gt_labels.cpu().numpy(), predicted.cpu().numpy(),  pred_labels.cpu().numpy()):
                    raw_results['name'].append(i)
                    raw_results['gt'].append(int(j))
                    raw_results['pred'].append(int(k))
                    raw_results['ori'].append(l)
                t.update(len(ims))
        valid_mss = f'Raw images: eval acc: {(correct / total):.5f};'
        print_and_write_log(valid_mss,
                            os.path.join(result_outputs_dir, 'det_logs.txt'))
        raw_results = pd.DataFrame(raw_results)
        raw_results.to_csv(os.path.join(result_outputs_dir, 'before_result.csv'))

        epoch_acces = AverageMeter()
        correct = 0
        total = 0
        attack_results = {'name': [],
                          'ori': [],
                          'gt': [],
                          'pred': []}
        with tqdm(desc='Attack sample generation phase =>',
                  total=(len(test_det_dataset_after) -
                         len(test_det_dataset_after) % args.batch_size), ncols=80) as t:
            t.set_description('Progress:')
            for data in test_det_dataloader_after:
                ims, gt_labels, im_names = data
                ims = ims.to(device)
                gt_labels = gt_labels.to(device)
                pred_labels = det_model(ims)
                _, predicted = torch.max(pred_labels.data, 1)
                total += gt_labels.size(0)
                correct += (predicted == gt_labels).sum().item()

                for i, j, k, l in zip(im_names, gt_labels.cpu().numpy(), predicted.cpu().numpy(),  pred_labels.cpu().numpy()):
                    attack_results['name'].append(i)
                    attack_results['gt'].append(j)
                    attack_results['pred'].append(k)
                    attack_results['ori'].append(l)
                t.update(len(ims))

        valid_mss = f'Attack images: eval acc: {(correct / total):.5f};'
        print_and_write_log(valid_mss,
                            os.path.join(result_outputs_dir, 'det_logs.txt'))
        att_results = pd.DataFrame(attack_results)
        att_results.to_csv(os.path.join(result_outputs_dir, 'after_result.csv'))

        print_and_write_log('raw details:\n', os.path.join(result_outputs_dir, 'det_logs.txt'))
        print_and_write_log(ana_results(raw_results),
                            os.path.join(result_outputs_dir, 'det_logs.txt'))
        print_and_write_log('attack details:\n', os.path.join(result_outputs_dir, 'det_logs.txt'))
        print_and_write_log(ana_results(att_results),
                            os.path.join(result_outputs_dir, 'det_logs.txt'))

    torch.cuda.empty_cache()
    print("========> Detecting attack samples over.")
