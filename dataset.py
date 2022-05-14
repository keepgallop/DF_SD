'''
@Description  :
@Author       : Chi Liu
@Date         : 2022-02-19 15:28:59
@LastEditTime : 2022-05-14 19:17:20
'''
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from PIL import Image
from sklearn.utils import shuffle
import glob
import os
import random


class AttackDataset(Dataset):
    def __init__(self,
                 csv_file,
                 transform=None,
                 allocation='train',
                 length=0,
                 color_mode='RGB'):
        """ Attack dataset
        Args:
            csv_file (str): path of the csv file. col1: 'filename': abs filepath; col2: 'two_cls': real/fake label;
                            col3: 'mul_cls': multi-class label; col4: 'allocation': train/valid/test
            transform: data transform function. Defaults to None.
            allocation (str, optional): data allocation to train/valid/test subsets. Defaults to 'train'.
            length (int, optional): dataset size to use. Defaults to 0 = full dataset.
            color_mode (str, optional): Defaults to 'RGB'.
        """

        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.allocation = allocation
        self.color_mode = color_mode
        if self.allocation in ['train', 'valid']:
            self.set_allocation = self.df.loc[(self.df.two_cls == 'r') & (
                self.df.allocation == self.allocation)]  # only real samples
        elif self.allocation == 'test':
            self.set_allocation = self.df.loc[(self.df.allocation == self.allocation)]
        elif self.allocation == 'attack-on-train':
            self.set_allocation = self.df.loc[(self.df.allocation == 'train')]

        self.set = (self.set_allocation if length == 0 else
                    self.set_allocation[:length])  # sampling length

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.set.iloc[idx, 0]
        raw_image = Image.open(img_name)
        tt = ToTensor()
        if self.color_mode == 'YCbCr':
            raw_image = raw_image.convert('YCbCr')

        if self.transform:
            image = raw_image.copy()
            if isinstance(self.transform, list):
                for t in self.transform:
                    image = t(image)
            else:
                image = self.transform(image)
            return tt(image), tt(raw_image), img_name
        else:
            return tt(raw_image), tt(raw_image), img_name


class AttackDataset2(Dataset):
    def __init__(self,
                 folder_path,
                 length=0,
                 mode='train'):
        """ Attack dataset from Folder used in the 2nd attack phase.
        Args:
            folder_path (str): path of the folder.
            ./dataset/ # folder_path
                -before/
                -after/
            length: dataset size
        """
        if mode == 'train':
            self.before_im_paths = glob.glob(os.path.join(os.path.join(folder_path, 'before'), '*-real-*.png'))
        elif mode == 'attack':
            self.before_im_paths = [i for i in glob.glob(os.path.join(os.path.join(
                folder_path, 'before'), '*.png')) if 'real' not in i]
        self.folder_path = folder_path
        self.sampled_img_names = ([i.split('/')[-1] for i in self.before_im_paths] if length == 0 else random.sample(
            [i.split('/')[-1] for i in self.before_im_paths], length))

    def __len__(self):
        return len(self.sampled_img_names)

    def __details__(self):
        return self.sampled_img_names[0:20]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.sampled_img_names[idx]  # file path
        img_pair = []

        for stage in ['before', 'after']:
            im_path = os.path.join(self.folder_path, stage, img_name)
            image = Image.open(im_path)
            img_pair.append(image)
        return img_pair[0], img_pair[1], img_name


class Subset(Dataset):
    r"""
    Subset of AttackDataset2 at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        tt = ToTensor()
        before_im, after_im, im_name = self.dataset[self.indices[idx]]
        if isinstance(self.transform, list):
            for t in self.transform:
                before_im = t(before_im)
                after_im = t(after_im)
        else:
            before_im = self.transform(before_im)
            after_im = self.transform(after_im)

        return tt(before_im), tt(after_im), im_name

    def __len__(self):
        return len(self.indices)


class DetectDataset(Dataset):
    def __init__(self,
                 csv_file,
                 transform=None,
                 allocation='train',
                 fake_class=[],
                 length=0,
                 color_mode='RGB'):
        """ Detector dataset
        Args:
            csv_file (str): path of the csv file. col1: 'filename': abs filepath; col2: 'two_cls': real/fake label;
                            col3: 'mul_cls': multi-class label; col4: 'allocation': train/valid/test
            transform: data transform function. Defaults to None.
            allocation (str, optional): data allocation to train/valid/test subsets. Defaults to 'train'.
            fake_class (list): select fake classes for training. Default to [] (i.e., all classes)
            length (int, optional): dataset size to use. Defaults to 0 = full dataset.
            color_mode (str, optional): Defaults to 'RGB'.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.allocation = allocation
        self.color_mode = color_mode
        self.fake_class = fake_class
        assert isinstance(self.fake_class, list), "fake_class must be a name list"
        if self.fake_class == ['all']:  # all fake classes are included
            self.set_allocation = self.df.loc[(
                self.df.allocation == self.allocation)]
        else:
            self.set_allocation1 = self.df.loc[(self.df.mul_cls.isin(self.fake_class)) & (
                self.df.allocation == self.allocation)]
            self.set_allocation2 = self.df.loc[(self.df.mul_cls == 'real') & (
                self.df.allocation == self.allocation)]
            self.set_allocation2 = self.set_allocation2[:len(self.set_allocation1)]

            self.set_allocation = self.set_allocation1.append(self.set_allocation2)
            self.set_allocation = shuffle(self.set_allocation)
            self.set_allocation.reset_index(inplace=True, drop=True)

        self.set = (self.set_allocation if length == 0 else
                    self.set_allocation[:length])  # sampling length

    def __details__(self):
        return self.set.head(20)

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.set.iloc[idx, 0]  # file path
        img_label = 0 if self.set.iloc[idx, 1] == 'r' else 1  # two-class label. Real('r) = 0, Fake('f) = 1
        img_mlabel = self.set.iloc[idx, 2]  # multi-class label.
        image = Image.open(img_name)
        tt = ToTensor()
        if self.color_mode == 'YCbCr':
            image = image.convert('YCbCr')

        if self.transform:
            image = image.copy()
            if isinstance(self.transform, list):
                for t in self.transform:
                    image = t(image)
            else:
                image = self.transform(image)
        return tt(image), img_label, img_mlabel, img_name


class DetectDatasetFromFolder(Dataset):
    def __init__(self,
                 folder_path,
                 transform=None,
                 fake_class=['all'],
                 length=0,
                 color_mode='RGB'):
        """ Detector dataset from Folder used in the attack evaluation phase.
        Args:
            folder_path (str): path of the folder.
            transform: data transform function. Defaults to None.
            color_mode (str, optional): Defaults to 'RGB'.
        """
        self.im_paths = glob.glob(os.path.join(folder_path, '*.png'))
        self.fake_class = fake_class
        self.length = length
        assert isinstance(self.fake_class, list), "fake_class must be a name list"

        if self.fake_class == ['all']:  # all fake classes are included
            self.im_paths = self.im_paths
            real_list = [i for i in self.im_paths if 'real' in i]
            real_list = random.sample(real_list, int(len(real_list)*0.25))  # keep class balance
            fake_list = [i for i in self.im_paths if 'real' not in i]
            self.im_paths = real_list + fake_list
        else:
            fake_list = []
            real_list = [i for i in self.im_paths if 'real' in i]
            for i in self.im_paths:
                for j in self.fake_class:
                    if j in i:
                        fake_list.append(i)
            real_list = random.sample(real_list, len(fake_list))  # keep class balance
            self.im_paths = real_list + fake_list

        if self.length == 0:
            self.im_paths = random.sample(self.im_paths, len(self.im_paths))
        else:
            self.im_paths = random.sample(self.im_paths, self.length)

        self.transform = transform
        self.color_mode = color_mode

    def __len__(self):
        return len(self.im_paths)

    def __details__(self):
        return self.im_paths[0:20]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.im_paths[idx]  # file path
        img_label = 0 if 'real' in img_name else 1  # two-class label. Real('r) = 0, Fake('f) = 1
        image = Image.open(img_name)
        tt = ToTensor()
        if self.color_mode == 'YCbCr':
            image = image.convert('YCbCr')

        if self.transform:
            image = image.copy()
            if isinstance(self.transform, list):
                for t in self.transform:
                    image = t(image)
            else:
                image = self.transform(image)
        return tt(image), img_label, img_name
