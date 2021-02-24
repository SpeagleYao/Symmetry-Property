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
import numpy as np
import matplotlib.pyplot as plt
from models import *
from utils import Logger
from time import time
from sklearn.manifold import TSNE
from adv_attack import fgsm, pgd, mim, cw
from data_loader import clean_loader_cifar, adv_loader_data
from robust_inference import robust_inference

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR model Train')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                help='random seed (default: 1)')
parser.add_argument('--eps', type=float, default=0.03)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
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

num_classes = 10
use_gpu = args.cuda
model = WVGG16_feat()
if args.cuda:
    model.cuda()
filename = '../checkpoint/vgg16_feat/vgg16_a2.0_d0.1.pth'
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint)

def pgd(img, label, model, criterion=F.cross_entropy, eps=0.03, iters=10,step=0.007, target_setting=False):
    adv = img.clone().detach()
    adv.requires_grad = True
    if adv.grad is not None:
        adv.grad.data.zero_()

    for j in range(iters):
        out_adv = model(adv)
        try:
            loss = criterion(out_adv, label)
        except:
            loss = criterion(out_adv[-1], label)
        loss.backward()

        noise = adv.grad
        if target_setting:
            adv.data = adv.data - step * noise.sign()
        else:
            adv.data = adv.data + step * noise.sign()

        adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
        adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    l2 = torch.norm((adv - img).reshape(img.shape[0], -1), dim=1).mean()

    return adv.detach(),l2

def craft_adv_samples(data_loader, model, args, attack_method):
    adv_samples = []
    target_tensor = []
    L2_list = []
    model.eval()
    for bi, batch in enumerate(data_loader):
        inputs, targets = batch
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        if attack_method == 'pgd1':
            crafted, l2 = pgd(inputs, targets, model, iters=1, eps=args.eps)
        elif attack_method == 'pgd10':
            crafted, l2 = pgd(inputs, targets, model, iters=10, eps=args.eps)
        elif attack_method == 'pgd20':
            crafted, l2 = pgd(inputs, targets, model, iters=20, eps=args.eps)
        elif attack_method == 'pgd50':
            crafted, l2 = pgd(inputs, targets, model, iters=50, eps=args.eps)
        elif attack_method == 'pgd100':
            crafted, l2 = pgd(inputs, targets, model, iters=100, eps=args.eps)    
        else:
            raise NotImplementedError
        adv_samples.append(crafted)
        target_tensor.append(targets)
        L2_list.append(l2)
        break

    return torch.cat(adv_samples, 0), torch.cat(target_tensor, 0), sum(L2_list)/len(L2_list)

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        feature, _, _ = model(data)
        break

print(feature.size())
print(target.size())

clean_loader = clean_loader_cifar(args)
adv_samples, targets, l2_mean = craft_adv_samples(clean_loader, model, args, 'pgd10')
with torch.no_grad():   
    adv_feature, _, _ = model(adv_samples)
feature = torch.cat((feature, adv_feature[0, :].unsqueeze(0)), 0)
target = torch.cat((target, targets[0].unsqueeze(0)), 0)

adv_samples, targets, l2_mean = craft_adv_samples(clean_loader, model, args, 'pgd20')
with torch.no_grad():   
    adv_feature, _, _ = model(adv_samples)
feature = torch.cat((feature, adv_feature[0, :].unsqueeze(0)), 0)
target = torch.cat((target, targets[0].unsqueeze(0)), 0)

print(adv_feature.size())
print(feature.size())
print(target.size())

tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(feature)
# print(result.shape)
# fig = plot_embedding(result, target,
                        #  't-SNE embedding of the digits (time %.2fs)'
                        #  % (time() - t0))
result1 = result[:1000]
target1 = target[:1000]
result2 = result[1000:1001]
result3 = result[1001:]
plt.scatter(result1[:, 0], result1[:, 1], c=target1, alpha=0.2, label='ori')
plt.scatter(result1[0, 0], result2[0, 1], c='r', marker='o', alpha=1, label='att')
plt.scatter(result2[:, 0], result2[:, 1], c='r', marker='*', alpha=1, label='att')
plt.scatter(result3[:, 0], result3[:, 1], c='r', marker='v', alpha=1, label='att')

plt.show()
plt.savefig('vgg16_ours_att.png')