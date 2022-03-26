'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-02-19 15:28:59
@LastEditTime : 2022-03-25 22:32:55
'''
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from PIL import Image


class AttackDataset(Dataset):
    """Attacker dataset."""
    def __init__(self,
                 csv_file,
                 transform=None,
                 set_mode='train',
                 length=0,
                 color_mode='RGB'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. Col1 saves the absolute path. Col2 saves the label. 
            transform (callable, optional): Optional transform to be applied on a sample.
            set_mode: "train":training phase; "infer":inference phase.
            length: sample length.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.set_mode = set_mode
        self.color_mode = color_mode
        if self.set_mode == "train":
            self.all_set = self.df[self.df['label'] == 0]  # all real samples
        elif self.set_mode == "infer":
            self.all_set = self.df[self.df['label'] == 1]  # all fake samples
        else:
            raise TypeError("set_mode not supported")
        self.all_set = self.all_set.sample(frac=1).reset_index(
            drop=True)  # shuffle
        self.set = (self.all_set if length == 0 else self.all_set[:length]
                    )  # sampling length

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
            image = self.transform(raw_image)
            return tt(image), tt(raw_image), img_name
        else:
            return tt(raw_image), img_name


class DetectDataset(Dataset):
    """Detector dataset."""
    def __init__(self, csv_file, transform=None, set_mode='train', length=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. Col1 saves the absolute path. Col2 saves the label. 
            transform (callable, optional): Optional transform to be applied on a sample.
            set_mode: "train":training phase; "infer":inference phase.
            length: sample length.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.set_mode = set_mode
        if self.set_mode == "train":
            self.set = self.df[self.df['set_mode'] ==
                               'train']  # predefined training samples
        elif self.set_mode == "valid":
            self.set = self.df[self.df['set_mode'] ==
                               'valid']  # predefined validation samples
        elif self.set_mode == "test":
            self.set = self.df[self.df['set_mode'] ==
                               'test']  # predefined testing samples
        else:
            raise TypeError("set_mode not supported")
        self.set = self.set.sample(frac=1).reset_index(drop=True)  # shuffle

        self.set = self.set if length == 0 else self.set[:
                                                         length]  # sampling length

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tt = ToTensor()
        img_name = self.set.iloc[idx, 0]
        image = Image.open(img_name)
        label = self.set.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        return tt(image), label, img_name