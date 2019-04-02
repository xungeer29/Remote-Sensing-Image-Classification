# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/remote_sensing_image_classification')
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config

def read_txt(path):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(' ')
            ims.append(im)
            labels.append(int(label))
    return ims, labels

class RSDataset(Dataset):
    def __init__(self, txt_path, width=256, height=256, transform=None, test=False):
        self.ims, self.labels = read_txt(txt_path)
        self.width = width
        self.height = height
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        im_path = self.ims[index]
        label = self.labels[index]
        im_path = os.path.join(config.data_root, im_path)
        im = Image.open(im_path)
        if self.transform is not None:
            im = self.transform(im)

        return im, label

    def __len__(self):
        return len(self.ims)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = RSDataset('./data/train.txt', width=256, height=256, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=1, num_workers=0)
    #for im, loc, cls in dataloader_train:
    for data in dataloader_train:
        print data
        #print loc, cls
    
