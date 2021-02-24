'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.nn import Parameter


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class FeatLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeatLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features    
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.bias)

    def forward(self, input):
        w = self.weight
        b = self.bias
        y = input.mm(w) + b
        return w, y

class PWVGG(nn.Module):
    def __init__(self, vgg_name):
        super(PWVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = FeatLinear(512, 10)

    def forward(self, x):
        out = self.features(x)
        feature = out.view(out.size(0), -1)
        w, out = self.classifier(feature)
        return w, out

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


def PWVGG16():
    return PWVGG('VGG16')

if __name__=='__main__':
    net = PWVGG('VGG16')
    x = torch.randn(512, 3, 32, 32)
    f, y = net(x)
    print(f.size())
    print(y.size())