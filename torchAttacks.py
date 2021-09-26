#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
parser.add_argument('--model-path',
                    default='./Pre-trained_Logit-Balancing_Model.pth',
                    # default='./model_cifar_wrn.pt',
                    # default='./TRADES-AWP_cifar10_linf_wrn34-10.pt',
                    help='model for white-box attack evaluation')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def testAttacks(model, device, test_loader):
    model.eval()
    robust_err_total = 0
    # filehandle = open('./results/logitmin_APGD_L.txt', 'w')

    filename = "/media/jiefei/Guji Mind1/Guji_Projects/rs/LB002_PGD64.txt"
    file_exists = os.path.isfile(filename) 
 
    if file_exists:
        print("file is exist")
        sys.exit(0)
    else:
        f = open(filename, "w")

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        X, y = Variable(data), Variable(target)
        # attack = torchattacks.APGD(model, eps=8./255.)
        # attack = torchattacks.GN(model, sigma=16./255.)
        # attack = torchattacks.Square(model, eps=8./255., verbose=False)
        # attack = torchattacks.APGDT(model, eps=8./255., verbose=False)
        # attack = torchattacks.OnePixel(model)
        # attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=40, random_start=True)
        # attack = torchattacks.CW(model, c=1.0, kappa=0, steps=1000, lr=0.01)
        attack = torchattacks.PGD(model, eps=64./255.,alpha=16/255, steps=20, random_start=True)
        # attack = torchattacks.FAB(model, eps=8./255.)
        
        # attack = torchattacks.PGD_Adaptive(model, eps=8./255., alpha=0.8/255., steps=20, random_start=True)
        # attack = torchattacks.PGDL2_Adaptive(model, eps=1.0, alpha=0.1, steps=160, random_start=True)
        
        # attack = torchattacks.AutoAttack(model, norm='Linf', eps=8./255., version='standard', n_classes=10, seed=None, verbose=False)
        
        adv_images = attack(X, y)
        
        err_cw = (model(adv_images).data.max(1)[1] != y.data).float().sum()
        robust_err_total += err_cw
        
        out = -F.log_softmax(model(adv_images), dim=1)
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

    # white-box attack
    print('pgd white-box attack')
    net = WideResNet_1()
    # net = WideResNet34()
    # net = nn.DataParallel(net)
    
    # Load the pretrained model
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['net'])
    net = nn.DataParallel(net)
    model = net.to(device)

    
    # net.load_state_dict(torch.load(args.model_path))
    
    # net = nn.DataParallel(net)
    # model = net.to(device)

    # model.load_state_dict(torch.load(args.model_path))

    testAttacks(model, device, test_loader)


if __name__ == '__main__':
    main()
