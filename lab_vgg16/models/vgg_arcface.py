'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .face_model import ArcMarginProduct

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

class ArcMargin_VGG16(nn.Module):
    def __init__(self,s,m):
        super(ArcMargin_VGG16, self).__init__()
        self.convlayers = VGG_CNN('VGG16')
        self.arcmargin_linear = ArcMarginProduct(
            in_features=512, out_features=10, s=s, m=m)

    def forward(self, x, target=None):
        feature = self.convlayers(x)
        logit = self.arcmargin_linear(feature, target)
        return feature, logit

if __name__=='__main__':
    net = VGG_CNN('VGG16')
    x = torch.randn(200, 3, 32, 32)
    feat = net(x)
    print(feat.size())