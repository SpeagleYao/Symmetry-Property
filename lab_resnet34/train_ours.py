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
from loss import L2Loss, WeightLoss
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
parser.add_argument('--lr', type=float, default=1, metavar='LR',
                help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=499, metavar='N',
                help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default='log_rebuttal')
parser.add_argument('--alpha', type=float, default=0.2, help='Orthogonality of the weight') # 2.0-7.5
parser.add_argument('--delta', type=float, default=0.1, help='change of args of weight') # 0.08 0.1 0.12
# delta 0.08-4 0.1-6 0.12-7
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args.cuda = not args.no_cuda and torch.cuda.is_available()

# log_filename = 'wres50_vallina.txt'
log_filename = 'resnet34_ours_a{}_d{}.txt'.format(args.alpha, args.delta)
sys.stdout = Logger(os.path.join(args.save_dir, log_filename))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    print("No cuda participate.")

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                     ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


model = WResNet34()
if args.cuda:
    model.cuda()

criterion_weight = WeightLoss(args.delta)
criterion_l2 = L2Loss(args.delta)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    bar = progressbar.ProgressBar(max_value=50000//args.batch_size + 1)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        w, output = model(data)
        # output = output * 10
        loss_l2 = criterion_l2(output, target) #intra
        loss_w = criterion_weight(w) #inter
        loss = loss_l2 + args.alpha * loss_w
        # loss = F.cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tLoss_l2: {:.6f}\tLoss_w: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_l2.item(), loss_w.item() *args.alpha))
        bar.update(batch_idx)
    bar.finish()
    filename = '../cp_res34/resnet34_ours_a{}_d{}.pth'.format(args.alpha, args.delta)
    torch.save(model.state_dict(), filename)


def inference():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * float(correct) / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75, args.epochs * 0.9]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print("Learning rate reduced.")
    train(epoch)
    inference()