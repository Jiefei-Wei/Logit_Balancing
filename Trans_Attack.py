#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:54:35 2021
"""
from __future__ import print_function
import sys
import os
import os.path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models import *
import torchattacks

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--source-model-path',
                    # default='./TRADES-AWP_cifar10_linf_wrn34-10.pt',
                    default='./model_cifar_wrn.pt',
                    # default='/media/jiefei/Guji Mind1/checkponts/lb/checkpoints/WideResNet_128_1_90_Adam.pth',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='/media/jiefei/Guji Mind1/Guji_Projects/Checkpoints/GN_002/WideResNet_128_0_100_Adam_sdlog002d_193_GN.pth',
                    help='target model for black-box attack evaluation')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def trans_attack(model_target, model_source, device, test_loader):
    model_target.eval()
    model_source.eval()
    
    robust_err_total = 0
    # filehandle = open('./results/logitmin_APGD_L.txt', 'w')

    filename = "/media/jiefei/Guji Mind1/Guji_Projects/rs/TRADES_Trans_pgd.txt"
    file_exists = os.path.isfile(filename) 
 
    if file_exists:
        print("file is exist")
        sys.exit(0)
    else:
        f = open(filename, "w")

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        X, y = Variable(data), Variable(target)
        # attack = torchattacks.APGD(model_source, eps=8./255.)
        # attack = torchattacks.GN(model)
        # attack = torchattacks.Square(model_source, eps=8./255., verbose=False)
        # attack = torchattacks.APGDT(model_source, eps=8./255., verbose=False)
        # attack = torchattacks.OnePixel(model_source)
        # attack = torchattacks.PGDL2(model_source, eps=1.0, alpha=0.2, steps=40, random_start=True)
        # attack = torchattacks.CW(model_source, c=1.0, kappa=0, steps=1000, lr=0.01)
        attack = torchattacks.PGD(model_source, eps=8./255., alpha=0.8/255., steps=20, random_start=True)
        # attack = torchattacks.FAB(model_source, eps=8./255.)
        
        adv_images = attack(X, y)
        
        # GN = torchattacks.GN(model)
        # mask_images = GN(adv_images, y)
        
        
        err_cw = (model_target(adv_images).data.max(1)[1] != y.data).float().sum()
        robust_err_total += err_cw
        
        out = -F.log_softmax(model_target(adv_images), dim=1)
        advlogit_min = 0.0
        advlogit_max = 0.0
        for i in range(len(out)):
            advlogit_min += out[i].data.min()
            advlogit_max += out[i].data.max()
        print("Min: {}".format(advlogit_min/200))
        print("Max: {}".format(advlogit_max/200))
        f.write("{}, {} \n".format(advlogit_min/200, advlogit_max/200))
        
        print('acc (white-box): ', err_cw)
        
    print('acc (CW): ', (10000-robust_err_total)/10000)
    f.write('acc (CW): {}'.format((10000-robust_err_total)/10000))

def main():

    print('Transfer attack')
    net_target = WideResNet_1()
    checkpoint = torch.load(args.target_model_path)
    net_target.load_state_dict(checkpoint['net'])
    net_target = nn.DataParallel(net_target)
    model_target = net_target.to(device)
    
    
    net_source = WideResNet_1()
    
    net_source.load_state_dict(torch.load(args.source_model_path))
    
    # checkpoint = torch.load(args.source_model_path)
    # net_source.load_state_dict(checkpoint['net'])
    
    net_source = nn.DataParallel(net_source)
    model_source = net_source.to(device)
    
    
    # net_source = WideResNet34()
    # net_source = nn.DataParallel(net_source)
    # model_source = net_source.to(device)
    # model_source.load_state_dict(torch.load(args.source_model_path))

    trans_attack(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
