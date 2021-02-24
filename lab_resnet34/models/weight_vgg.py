'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features    
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        xl = input.unsqueeze(-1).repeat(1,1,10)
        xl = torch.var(xl, dim=1)
        wl = self.weight.repeat(xl.size()[0], 1, 1)
        wl = torch.var(wl, dim=1)
        # print(xl.size(), wl.size())
        # print(xl[0], wl[0])

        # ll = torch.norm(xl-wl, dim=1)/10
        # print(xl.size(), wl.size(), ll.size())
        # print(f'x_norm:\t{torch.norm(xl, dim=1)}')
        # print(f'w_norm:\t{torch.norm(wl, dim=1)}')
        # print(f'gk:\t{torch.exp(-ll)}')
        # print(f'w_length:\t{torch.norm(self.weight, dim=0)}')
        x = F.normalize(input, dim=-1)
        w = F.normalize(self.weight, dim=0)
        cos_theta = x.mm(w)
        # print(cos_theta)
        # print(torch.norm(cos_theta, dim=-1))
        return w, cos_theta

class WVGG(nn.Module):
    def __init__(self, vgg_name):
        super(WVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.linear = CosineLinear(512, 10)

    def forward(self, x):
        out = self.features(x)
        feature = out.view(out.size(0), -1)
        w, y = self.linear(feature)
        return w, y

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


def WVGG16():
    return WVGG('VGG16')

if __name__=='__main__':
    net = WVGG16()
    x = torch.rand(7, 3, 32, 32)
    # f, y = net(x)
    # print(y[0])
    filename = '../checkpoint/vgg16_ft_nloss/vgg16_a2.0_d0.1.pth'
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint)
    f, y = net(x)
    # print(y[0])
    # print(f.size())
    # print(y.size())