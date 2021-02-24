'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .face_model import LSoftmaxLinear


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_CNN(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_CNN, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        feature = out.view(out.size(0), -1)
        return feature

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class Lsoftmax_VGG16(nn.Module):
    def __init__(self,margin):
        super(Lsoftmax_VGG16, self).__init__()
        self.convlayers = VGG_CNN('VGG16')
        self.lsoftmax_linear = LSoftmaxLinear(
            input_features=512, output_features=10, margin=margin, device='cuda')
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, x, target=None):
        feature = self.convlayers(x)
        logit = self.lsoftmax_linear(feature, target)
        return feature, logit

if __name__=='__main__':
    net = Lsoftmax_VGG16()
    x = torch.randn(512, 3, 32, 32)
    f, y = net(x)
    print(f.size())
    print(y.size())