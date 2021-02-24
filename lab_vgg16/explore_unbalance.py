from __future__ import print_function
import argparse
import torch
import numpy as np
from adv_attack import fgsm, pgd #, bim, mim
from models import *
from torchsummary import summary
from data_loader import clean_loader_cifar, adv_loader_data, robust_inferece, robust_evaluate


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Attack')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                help='random seed (default: 1)')
parser.add_argument('--data-dir', type=str, default='data', metavar='N',
                help='')
parser.add_argument('--model-checkpoint', type=str, default='../checkpoint/vgg16/vgg16_vallina.pth', metavar='N',
                help='')
parser.add_argument('--attack-method', type=str, choices=['fgsm', 'pgd', 'bim', 'mim'], default='pgd')
parser.add_argument('--eps', type=float, default=0.03)
parser.add_argument('--eps-fgsm', type=float, default=0.03)
args = parser.parse_args()


def craft_adv_samples(data_loader, model, args):
    adv_samples = []
    target_tensor = []
    L2_list = []
    model.eval()
    for bi, batch in enumerate(data_loader):
        inputs, targets = batch
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        if args.attack_method == 'fgsm':
            crafted, l2 = fgsm(inputs, targets, model, eps=args.eps_fgsm)
        elif args.attack_method == 'pgd':
            crafted, l2 = pgd(inputs, targets, model, eps=args.eps, iters=10)
        adv_samples.append(crafted)
        target_tensor.append(targets)
        L2_list.append(l2)

    return torch.cat(adv_samples, 0), torch.cat(target_tensor, 0), sum(L2_list)/len(L2_list)


def main():
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    clean_loader = clean_loader_cifar(args)
    model = VGG16()
    #summary(model, input_size= (3, 32, 32), device='cpu')
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()
    model.load_state_dict(torch.load(args.model_checkpoint))

    robust_inferece(model, clean_loader, args, note='natural')
    robust_evaluate(model, clean_loader, args, note='natural')

    adv_samples, targets, l2_mean = craft_adv_samples(clean_loader, model, args)
    print(adv_samples.shape)
    print(targets.shape)
    if args.cuda:
        adv_samples = adv_samples.cpu()
        targets = targets.cpu()
    adv_loader = adv_loader_data(args, adv_samples, targets)
    robust_inferece(model, adv_loader, args, note=args.attack_method)
    confusion = robust_evaluate(model, adv_loader, args, note=args.attack_method)
    #np.save('confusion_matrix/vallina_fgsm.npy',confusion)

    unbalance_distribution = np.sum(confusion,axis=0)
    print('Unbalance Distribution:\n', unbalance_distribution)

if __name__ == '__main__':
    main()