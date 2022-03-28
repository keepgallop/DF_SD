'''
@Description  : script to formulate dataset directory to a csv file with shuffle operation
@Author       : Chi Liu
@Date         : 2022-03-28 12:16:16
@LastEditTime : 2022-03-28 22:51:52
'''
#%%
from fileinput import filename
import os
import glob
import shutil
import random

random.seed(256)

celeba_path = '../dataset/0-Real/'
progan_path = '../../TR-Net/20211220_liuchi/data/progan/sewed/'
stgan_path = '../../TR-Net/20211220_liuchi/data/stgan/sewed/'
stargan_path = '../../TR-Net/20211220_liuchi/data/stargan/sewed/'
mmdgan_path = '../dataset/4-MMDGAN/'

to_path = '../dataset/celeba-128/'

#%%
celeba_list = glob.glob(celeba_path + '*.png')
print(len(celeba_list))
select_celeba_list = random.sample(celeba_list, 88000)
for i in select_celeba_list:
    shutil.copy(i, os.path.join(to_path, 'real'))
print('celeba done')

for i in glob.glob(os.path.join(progan_path, '*.png')):
    shutil.copy(i, os.path.join(to_path, 'progan'))
for i in glob.glob(os.path.join(stgan_path, '*.png')):
    shutil.copy(i, os.path.join(to_path, 'stgan'))
for i in glob.glob(os.path.join(stargan_path, '*.png')):
    shutil.copy(i, os.path.join(to_path, 'stargan'))
mmdgan_list = glob.glob(os.path.join(mmdgan_path, '*.png'))
select_mmdgan_list = random.sample(celeba_list, 22000)
for i in select_mmdgan_list:
    shutil.copy(i, os.path.join(to_path, 'mmdgan'))
print('fake done')

#%%
from PIL import Image
import torchvision.transforms as transforms


def split_img(p):
    im = Image.open(p)
    im = transforms.ToTensor()(im)  # C x H x W
    im_real = im[:, :, :128]
    im_fake = im[:, :, 128:]
    im_real = transforms.ToPILImage()(im_real)
    im_fake = transforms.ToPILImage()(im_fake)
    im_fake.save(p)


for gan in ['progan', 'stgan', 'stargan']:
    path = os.path.join(to_path, gan)
    for i in glob.glob(os.path.join(path, '*.png')):
        split_img(i)
        print(gan, '--> done')

#%%
import pandas as pd


def create_128_list(data_path):
    """create dataset csv file 

    Args:
        data_path (str): dataset father path
    return:
        csv file. 
        col1: 'filename': abs filename; 
        col2: 'two_cls': real/fake label;
        col3: 'mul_cls': multi-class label;
        col4: 'allocation': train/valid/test
    """

    file_names = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('png'):
                file_names.append(os.path.join(root, file))
    assert len(file_names) == 176000
    random.shuffle(file_names)

    two_cls = []
    mul_cls = []
    allocation = []
    for i, file in enumerate(file_names):
        if 'real' in file:
            two_cls.append('r')
            mul_cls.append('real')
        elif 'progan' in file:
            two_cls.append('f')
            mul_cls.append('progan')
        elif 'stgan' in file:
            two_cls.append('f')
            mul_cls.append('stgan')
        elif 'stargan' in file:
            two_cls.append('f')
            mul_cls.append('stargan')
        elif 'mmdgan' in file:
            two_cls.append('f')
            mul_cls.append('mmdgan')
        if i + 1 <= 144000:
            allocation.append('train')
        elif i + 1 <= 160000:
            allocation.append('valid')
        else:
            allocation.append('test')
    df = pd.DataFrame(
        data={
            'filename': file_names,
            'two_cls': two_cls,
            'mul_cls': mul_cls,
            'allocation': allocation,
        })
    return df


df = create_128_list('../dataset/celeba-128/')

#%%
print(df.head(10))
print(len(df))
print(
    sum(df['two_cls'] == 'r'),
    sum(df['mul_cls'] == 'real'),
    sum(df['mul_cls'] == 'progan'),
    sum(df['mul_cls'] == 'stgan'),
    sum(df['mul_cls'] == 'stargan'),
    sum(df['mul_cls'] == 'mmdgan'),
)

print(sum(df['allocation'] == 'train'), sum(df['allocation'] == 'valid'),
      sum(df['allocation'] == 'test'))

print(
    sum(df[df['allocation'] == 'train']['two_cls'] == 'r'),
    sum(df[df['allocation'] == 'valid']['two_cls'] == 'r'),
    sum(df[df['allocation'] == 'test']['two_cls'] == 'r'),
)

# %%
df.loc[(df.two_cls == 'r') & (df.allocation == 'train')]

# %%
tmp = [i.replace('../', './') for i in df['filename']]
# %%
df['filename'] = tmp

# %%
df.to_csv('../dataset/celeba-128/celeba_128.csv', index=False)