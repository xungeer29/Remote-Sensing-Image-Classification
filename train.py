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

    # train data
    transform = transforms.Compose([transforms.Scale(256),
                                    transforms.RandomSizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(10),
                                    transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_train = RSDataset('./data/train.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)

    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./data/valid.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
               config.model, config.num_classes, config.num_epoch, config.learning_rate, 
               config.width, config.height, config.iter_smooth))

    # load checkpoint
    if os.path.exists(os.path.join('./checkponts', config.checkpoint)):
        model = torch.load(os.path.join('./checkponts', config.checkpoint))

    # train
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    for epoch in range(config.num_epochs):
        ep_start = time.time()
        model.train()
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(label).cuda().long()

            output = model(input)
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1 = accuracy(output.data, target.data, topk=(1,))
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += top1
            sum += 1

            if (i+1) % config.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, top1: %.4f'
                       %(epoch+1, configs.num_epochs, i+1, len(dst_train)//configs.batch_size, 
                       train_loss_sum/sum, train_top1_sum/sum))
                log.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, top1: %.4f\n'
                           %(epoch+1, configs.num_epochs, i+1, len(dst_train)//configs.batch_size, 
                           train_loss_sum/sum, train_top1_sum/sum))
                sum = 0
                train_loss_sum = 0
                train_top1_sum = 0

        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < args.num_epochs:
            val_loss, val_top1, val_time = eval(model, dataloader_valid)
            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f'
                   %(epoch+1, configs.num_epochs, val_loss, val_top1, val_time))
            print('epoch time: {}'.format(epoch_time))
            print('Taking snapshot...')
            if not os.path.exists('./checkpoint'):
                os.makedirs('./checkpoints')
            torch.save(model, '{}/{}_{}_{%.4f}.pth'.format('checkpoints', args.model, epoch, val_top1))

            log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f\n'
                       %(epoch+1, configs.num_epochs, val_loss, val_top1, val_time))
    log.write('-'*30+'\n')
    log.close()

# validation
def eval(model, dataloader_valid):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    val_time_start = time.time()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    time = (time.time() - val_time_start) / 60.
    return avg_loss, avg_top1, time

if __name__ == '__main__':
    train()
