'''
@Description  :
@Author       : Chi Liu
@Date         : 2022-02-19 15:28:59
@LastEditTime : 2022-05-03 23:52:47
'''
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from PIL import Image
from sklearn.utils import shuffle
import glob
import os


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
                 color_mode='RGB'):
        """ Detector dataset from Folder used in the attack evaluation phase.
        Args:
            folder_path (str): path of the folder.
            transform: data transform function. Defaults to None.
            color_mode (str, optional): Defaults to 'RGB'.
        """
        self.im_paths = glob.glob(os.path.join(folder_path, '*.png'))
        # self.im_paths = [i for i in glob.glob(os.path.join(folder_path, '*.png'))
        #  if ('stargan' not in i) and ('mmdgan' not in i)]
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
