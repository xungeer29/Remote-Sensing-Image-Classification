# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/remote_sensing_image_classification')
import os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.dataset import *
from networks.network import *
from config import config

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    #parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
    #                    default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=500, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.001, type=float)
    parser.add_argument('--model', default='ResNet34', type=str, metavar='Model',
                        help='model type: ResNet18, ResNet34, ResNet50, ResNet101')
    parser.add_argument('--width', dest='width', help='The width of image.',
                        default=224, type=int)
    parser.add_argument('--height', dest='height', help='The height of image.',
                        default=224, type=int)

    args = parser.parse_args()

    return args

def train():
    # model
    if config.model == 'ResNet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet152':
        backbone = models.resnet152(pretrained=True)
        model = ResNet152(backbone, num_classes=config.num_classes)
    else:
        print('ERROR: No model {}!!!'.format(config.model))
    print model
    model.cuda()
    
    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99), weight_decay=0.0002)

    # data
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = RSDataset('./data/train.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
    
    # train
    model.train()
    for epoch in range(config.num_epochs):
        ep_start = time.time()
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(label).cuda()

            output = model(input)
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: loc_x %.4f, loc_y %.4f, cls %.4f'
                       %(epoch+1, args.num_epochs, i+1, len(dst_train)//args.batch_size, 
                       loss_x.data, loss_y.data, loss_cls.data))
        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < args.num_epochs:
            print('epoch time: {}'.format(epoch_time))
            print 'Taking snapshot...'
            torch.save(model, '{}_{}.pth'.format(args.model, epoch))

if __name__ == '__main__':
    train()
