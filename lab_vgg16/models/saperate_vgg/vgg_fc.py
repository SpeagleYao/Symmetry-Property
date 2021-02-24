'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

class VGG_FC(nn.Module):
    def __init__(self):
        super(VGG_FC, self).__init__()
        self.classifier = nn.Linear(512, 10)

    def forward(self, feat):
        out = self.classifier(feat)
        return out

if __name__=='__main__':
    net = VGG_FC()
    feat = torch.randn(2, 512)
    y = net(feat)
    print(y.size())