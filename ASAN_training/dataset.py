import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.image import imread
import random

def make_dataset(root):
    imgs = []
    imgpath = os.path.join(root,"train_image_mul_LVmask")
    maskpath = os.path.join(root,"train_normal_edemascar_2class_mask")
    ImageSet = os.listdir(maskpath)   
    ImageSet.sort()
    MaskSet = os.listdir(maskpath)
    MaskSet.sort()
    n = len(os.listdir(maskpath))
    modal=('C0','DE','T2')
    for i in range(n):
        path_temp = MaskSet[i]
        path_split = path_temp.split('_')[:6]
        img_temp0 = path_split[0]+'_'+path_split[1]+'_'+path_split[2]+'_'+modal[0]+'_'+path_split[4]
        img_temp1 = path_split[0]+'_'+path_split[1]+'_'+path_split[2]+'_'+modal[1]+'_'+path_split[4]
        img_temp2=path_split[0]+'_'+path_split[1]+'_'+path_split[2]+'_'+modal[2]+'_'+path_split[4]
        img0 = os.path.join(imgpath,img_temp0)
        img1 = os.path.join(imgpath,img_temp1)
        img2 = os.path.join(imgpath,img_temp2)
        mask = os.path.join(maskpath,path_temp)
        imgs.append((img0,img1,img2,mask))
    return imgs

class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,normalization=None):

        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.normalization = normalization

    def __getitem__(self, index):
        img0_path,img1_path,img2_path,y_path = self.imgs[index]
        img_0 = Image.open(img0_path).convert('L')
        img_1 = Image.open(img1_path).convert('L')
        img_2 = Image.open(img2_path).convert('L')
        img_y = Image.open(y_path).convert('L')
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            random.seed(seed)
            img_0 = self.transform(img_0)
            random.seed(seed)
            img_1 = self.transform(img_1)
            random.seed(seed)
            img_2 = self.transform(img_2)
        if self.target_transform is not None:
            random.seed(seed)
            img_y = self.target_transform(img_y)
        if self.normalization is not None:
            img_0 = self.normalization(img_0)
            img_1 = self.normalization(img_1)
            img_2 = self.normalization(img_2)
        img_y = torch.FloatTensor(np.array(img_y))
        img_y = img_y.unsqueeze(dim=0)
        return img_0,img_1,img_2,img_y

    def __len__(self):
        return len(self.imgs)

class LivertestDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,normalization=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.normalization = normalization

    def __getitem__(self, index):
        img0_path,img1_path,img2_path,y_path = self.imgs[index]
        img_0 = Image.open(img0_path).convert('L')
        img_1 = Image.open(img1_path).convert('L')
        img_2 = Image.open(img2_path).convert('L')
        img_y = Image.open(y_path).convert('L')
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            random.seed(seed)
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        if self.target_transform is not None:
            random.seed(seed)
            img_y = self.target_transform(img_y)
        if self.normalization is not None:
            img_0 = self.normalization(img_0)
            img_1 = self.normalization(img_1)
            img_2 = self.normalization(img_2)
        img_y = torch.FloatTensor(np.array(img_y))
        img_y = img_y.unsqueeze(dim=0)
        return img_0,img_1,img_2,img_y,y_path

    def __len__(self):
        return len(self.imgs)