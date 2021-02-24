from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('..')
from models import *
from utils import Logger
import numpy as np
import progressbar
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR model Train')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                help='random seed (default: 1)')
parser.add_argument('--alpha', type=float, default=2.0, help='Orthogonality of the weight') # 2.0-7.5
parser.add_argument('--delta', type=float, default=0.1, help='change of args of weight') # 0.08 0.1 0.12
# delta 0.08-4 0.1-6 0.12-7
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    print("No cuda participate.")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


model = WResNet34()
if args.cuda:
    model.cuda()
filename = '../checkpoint/wres34_ft_nloss/wres34_a0.05_d0.1.pth'
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint)

with torch.no_grad():
    for name, parameters in model.named_parameters():
        print(name,':',parameters.size())
        if name == 'linear.weight': weight = parameters.cpu().numpy()


def inference():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # bar = progressbar.ProgressBar(max_value=10000//args.test_batch_size + 1)
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # bar.update(bi)
        # bar.finish()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * float(correct) / len(test_loader.dataset)))
    print(f'weight_size: {weight.shape}')
    np.save('../weight_vector/resnet34/weight_ours', weight)

if __name__=='__main__':
    inference()