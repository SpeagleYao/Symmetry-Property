from __future__ import print_function
import argparse
import os
import torch
import sys
sys.path.append('..')
#from models.vgg import VGG
from models import SphereProduct_ResNet18
from adv_attack import fgsm, pgd, mim, cw
from data_loader import clean_loader_cifar, adv_loader_data
from robust_inference import robust_inference


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Attack')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                help='random seed (default: 1)')
parser.add_argument('--data-dir', type=str, default='data', metavar='N',
                help='')
parser.add_argument('--model-checkpoint', type=str, default='../checkpoint/resnet18/resnet18_sphereproduct.pth', metavar='N')
parser.add_argument('--eps', type=float, default=0.03)
parser.add_argument('--eps-fgsm', type=float, default=0.03)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        if attack_method == 'fgsm':
            crafted, l2 = fgsm(inputs, targets, model, eps=args.eps_fgsm)
        elif attack_method == 'pgd10':
            crafted, l2 = pgd(inputs, targets, model, iters=10, eps=args.eps)
        elif attack_method == 'pgd20':
            crafted, l2 = pgd(inputs, targets, model, iters=20, eps=args.eps)
        elif attack_method == 'mim':
            crafted, l2 = mim(inputs, targets, model, eps=args.eps)
        elif attack_method == 'c&w':
            crafted, l2 = cw(inputs, targets, model)
        else:
            raise NotImplementedError
        adv_samples.append(crafted)
        target_tensor.append(targets)
        L2_list.append(l2)

    return torch.cat(adv_samples, 0), torch.cat(target_tensor, 0), sum(L2_list)/len(L2_list)


def main():

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    clean_loader = clean_loader_cifar(args)
    model = SphereProduct_ResNet18(m=4)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()
    model.load_state_dict(torch.load(args.model_checkpoint))

    robust_inference(model, clean_loader, args, note='natural')

    for attack_method in ['fgsm', 'pgd10', 'pgd20', 'c&w', 'mim']:
        adv_samples, targets, l2_mean = craft_adv_samples(clean_loader, model, args, attack_method)
        if args.cuda:
            adv_samples = adv_samples.cpu()
            targets = targets.cpu()
        adv_loader = adv_loader_data(args, adv_samples, targets)

        robust_inference(model, adv_loader, args, note=attack_method)

if __name__ == '__main__':
    main()