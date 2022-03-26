'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-01-13 16:29:20
@LastEditTime : 2022-01-22 20:07:59
'''
import sys

sys.path.append("..")
import torch as T
from DeepCNN import DeepCNN
import torchvision
import marapapmann.pylib as py
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import marapapmann as M
import tqdm
import os
from defense import Defense
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ===============================================
# =               Parse arguments               =
# ===============================================

# Set up arguments
py.arg('--n_ep', type=int, default=100, help='Epoch numbers.')
py.arg('--bs', type=int, default=256, help='Batch size.')
py.arg('--dir_exp', type=str, default='./exp/', help='Experiment directory.')

py.arg('--n_max_keep',
       type=int,
       default=5,
       help='Maximum number of checkpoints to keep.')

py.arg('--dir_data',
       type=str,
       default='./detection_dataset/',
       help='Data directory.')

py.arg('--is_defense', type=bool, default=False, help='defense or not.')

py.arg('--opt', type=str, default='sgd', help='Optimizer')
py.arg('--lr', type=float, default=1e-2, help='Learning rate.')

py.arg('--cln_space', type=bool, default=True, help='Clean test space.')
py.arg('--detector', type=str, default='xception', help='detector name.')
# Parse arguments
args = py.args()

# =======================================================
# =                Set global parameters                =
# =======================================================

if T.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

py.mkdir(args.dir_exp)

dir_ckpt = py.join(args.dir_exp, 'ckpt')
pth_log = py.join(args.dir_exp, 'training_log.txt')
trainset_dir = py.join(args.dir_data, 'train/')
validset_dir = py.join(args.dir_data, 'valid/')
testset_dir = py.join(args.dir_data, 'test/')

py.mkdir(dir_ckpt)

if args.cln_space:
    rm_count = 0
    for root, _, files in os.walk(args.dir_exp):
        for file in files:
            os.remove(py.join(root, file))
            rm_count += 1
    T.cuda.empty_cache()

    print(f'Done. {rm_count} File Removed')

##############################################################################
#
#                             Load dataset
#
##############################################################################
trans_funs = []
if args.is_defense:
    defense_strategy = Defense().set_strategy
    trans_funs.append(defense_strategy)
trans_funs.append(transforms.ToTensor())
trans_funs.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

train_transform = transforms.Compose(trans_funs)
other_transform = transforms.Compose(trans_funs[-2:])

trainset = datasets.ImageFolder(trainset_dir, transform=train_transform)
validset = datasets.ImageFolder(validset_dir, transform=other_transform)
testset = datasets.ImageFolder(testset_dir, transform=other_transform)

trainloader = T.utils.data.DataLoader(trainset,
                                      batch_size=args.bs,
                                      shuffle=True,
                                      num_workers=2)
validloader = T.utils.data.DataLoader(validset,
                                      batch_size=args.bs,
                                      shuffle=False,
                                      num_workers=2)
testloader = T.utils.data.DataLoader(testset,
                                     batch_size=args.bs,
                                     shuffle=False,
                                     num_workers=2)
#* Test snippet -----------------
# import matplotlib.pyplot as plt
# import numpy as np

# classes = ['real', 'fake']
# # functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()

#     return np.transpose(npimg, (1, 2, 0))
# # get some random training images
# dataiter = iter(testloader)
# images, labels = dataiter.next()
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(args.bs)))
# im = imshow(torchvision.utils.make_grid(images))
# plt.imsave('./temp.jpg', im)
#* Test snippet -----------------

##############################################################################
#
#                       Detector Network
#
##############################################################################
print("Create detector.")
if args.detector == 'resnet':
    model = DeepCNN().resnet.to(device)
elif args.detector == 'xception':
    model = DeepCNN().xception.to(device)

##############################################################################
#
#                       Losses, Optimiziers
#
##############################################################################
criterion = nn.CrossEntropyLoss()
if args.opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
elif args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
sch = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

##############################################################################
#
#                      Training loop
#
##############################################################################

min_valid_loss = np.inf

n_iter = 0
for ep_ in tqdm.trange(args.n_ep, desc='Epoch Loop'):
    train_loss = 0.0
    model.train()  # Optional when not using Model Specific layer
    for data, labels in tqdm.tqdm(trainloader, desc='Iter Loop'):

        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        target = model(data)
        loss = criterion(target, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_iter += 1

        if n_iter % 400 == 0:
            sch.step(loss)

    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    for data, labels in validloader:
        data, labels = data.to(device), labels.to(device)

        target = model(data)
        loss = criterion(target, labels)
        valid_loss = loss.item() * data.size(0)

    f_log = open(pth_log, 'a')
    log = f'Epoch {ep_+1} \t Training Loss: {train_loss / len(trainloader)} \t Validation Loss: {valid_loss / len(validloader)} \n'
    f_log.write(log)

    if min_valid_loss > valid_loss:
        # log = f'Validation Loss Decreased({min_valid_loss:.3f}--->{valid_loss:.3f}) \t Saving The Model \n'
        min_valid_loss = valid_loss
        f_log.write(log)
        # Saving State Dict
        ckpt = {
            'ep': ep_,
            'n_iter': n_iter,
            'model': model.state_dict(),
            'optim': optimizer.state_dict()
        }
        M.torchlib.save_checkpoint(ckpt,
                                   py.join(dir_ckpt,
                                           'epoch_%d.dict' % (ep_ + 1)),
                                   max_keep=args.n_max_keep)
    f_log.close()