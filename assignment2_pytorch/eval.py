'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import get_accuracy_per_class, get_top3_per_class 
from utils_pt import get_predict_labels, draw_confusion_matrix, get_list_predicted_data, get_output_softmax_list
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='batch size')
parser.add_argument('--checkpoint', '-c', default=None, help='checkpoint file path', required=True)
parser.add_argument('--draw_confusion', '-d', action='store_true', help='draw confusion matrix')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
# Load checkpoint.
print('==> Load checkpoint..')
checkpoint_dir = 'checkpoint'
assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print('==> After training %.1f get Best Acc: %.3f' % (start_epoch, best_acc))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def eval(testloader, model):
    print('Start evaluating')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    probabilities = []
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        y_true += targets.tolist()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        y_pred += (get_predict_labels(outputs))
        probabilities += (get_output_softmax_list(outputs))
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Loss: %.3f | Acc: %.3f%% | (Correct/Total): (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return y_true, y_pred, probabilities

y_true, y_pred, probabilities = eval(testloader, net)
print('Accuracy per class')
print(get_accuracy_per_class(get_list_predicted_data(y_pred, probabilities), y_true))
print('Top3 per class')
print(get_top3_per_class(get_list_predicted_data(y_pred, probabilities), y_true))
if args.draw_confusion:
   draw_confusion_matrix(y_true, y_pred, classes)
