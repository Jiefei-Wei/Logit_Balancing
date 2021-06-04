#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:56:31 2020

@author: jiefei
"""

'''Train CIFAR10 Logit_Balancing with PyTorch.'''


import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models import *
from models.wideresnet import *
from trades import trades_loss

acc = 0
bestAccuracy = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def train(args, model, device, train_loader, optimizer, epoch, epochs, LBWeight):
    global start_epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        confidences = F.softmax(model(data), dim=1)
        output = F.log_softmax(model(data), dim=1)
        pred = output.argmax(dim=1)
        losses = F.nll_loss(output, target, reduction = 'none')
        # set reduction of loss ot none, get each loss
        # for loop if pred == target set loss to 0
        
        for i in range(len(losses)):
            if pred[i] == target[i]:
                Error_log=torch.cat([output[i,:pred[i]],output[i,pred[i]+1:]])
                #print(Error_log)
                add_loss = torch.std(Error_log)
                #print(add_loss)
                edecay = confidences[i,target[i]].item()
                losses[i]=0.0*losses[i] + LBWeight*add_loss*edecay
        loss = torch.mean(losses)
        loss.backward()
        optimizer.step()

        
        if batch_idx == 0:
            print('Train Epoch: {}/{} [{:05}/{} ({:>3}%)]\tLoss: {:.4f}'.format(
                epoch+start_epoch, start_epoch+epochs, batch_idx+1 * len(data), len(train_loader.dataset),
                int(100. * batch_idx+1 / len(train_loader)), loss.item()))
        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {}/{} [{:05}/{} ({:>3}%)]\tLoss: {:.4f}'.format(
                epoch+start_epoch, start_epoch+epochs, (batch_idx+1) * len(data), len(train_loader.dataset),
                int(100. * (batch_idx+1) / len(train_loader)), loss.item()))

            if args.dry_run:
                break

def test(model, device, test_loader):
    global acc
    global bestAccuracy
    model.eval()
    test_loss = 0
    correct = 0
    Error_log=0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #confidences = F.softmax(model(data), dim=1)
            output = F.log_softmax(model(data), dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(len(pred)):
                Error_log += torch.std(torch.cat([output[i,:pred[i]],output[i,pred[i]+1:]]))
            
    Error_log /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n average err cfd: {}\n'.format(
        test_loss, correct, len(test_loader.dataset), acc, Error_log))
    
    


def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model_name', type=str, default=WideResNet(), metavar='N',
                        help='input the model-name for traning more info check the models folder. e.g net = SimpleDLA(), ResNet18(), GoogLeNet(), DenseNet121(), MobileNet()')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--LBWeight', type=float, default=0.04,
                        help='Logit Balancing Weight (default: 0.02)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizers')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--target', type=int, default=100, metavar='N',
                        help='tarining target. Stop traning when acc > target')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume from checkpoint')
    args = parser.parse_args()
    
    global bestAccuracy, acc, start_epoch
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, **train_kwargs)
    
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, **test_kwargs)
    
    
    print('==> Building model...model = {0}'.format(args.model_name))
    net = args.model_name
    
    if args.resume:
    # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if os.path.isfile('/media/jiefei/Guji Mind/Guji_Projects/Checkpoints/stdlog001/WideResNet_128_0_100_Adam_sdlog001d_92.pth'.format(args.target ,args.optim)):
            checkpoint = torch.load('/media/jiefei/Guji Mind/Guji_Projects/Checkpoints/stdlog001/WideResNet_128_0_100_Adam_sdlog001d_92.pth'.format(args.target ,args.optim))
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            bestAccuracy = checkpoint['acc']
            print('==> Resuming from epoch: {}\n'.format(start_epoch))
        else:
            assert os.path.isdir('/media/jiefei/Guji Mind/Guji_Projects/Checkpoints/stdlog002/WideResNet_128_0_100_Adam_sdlog002d_92.pth'.format(args.target ,args.optim)), 'Error: no checkpoint directory found!'


    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    model = net.to(device)
    
    
    if args.optim == 'Adam': 
        optimizer = optim.Adam(model.parameters())
    if args.optim == 'SGD': 
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9)
    if args.optim == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters())
    
    print(optimizer)
    
    scheduler = StepLR(optimizer, step_size=40, gamma=args.gamma)
    
    LBWeight=args.LBWeight
    test(model, device, test_loader)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.epochs, LBWeight)
        test(model, device, test_loader)
        scheduler.step()
        if args.save_model:
            if acc > bestAccuracy:
                print('Saving... Best Accuracy = {}%\n'.format(acc))
                bestAccuracy=acc
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch+start_epoch,
            }

            torch.save(state, '/media/jiefei/Guji Mind/Guji_Projects/Checkpoints/stdlog004/WideResNet_128_0_{}_{}_sdlog004d_{}.pth'.format(args.target, args.optim, epoch))
                
        if bestAccuracy > args.target :
            print('Accuarcy over {}%! Stop Training...'.format(args.target))
            break

    
if __name__ == '__main__':
    main()
