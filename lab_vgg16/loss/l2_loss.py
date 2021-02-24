import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import math

class L2Loss(nn.Module):
    def __init__(self, __delta__):
        super(L2Loss, self).__init__()
        self.delta = __delta__
        self.crit_map = torch.Tensor([
           # 0        1        2        3        4        5        6        7        8        9
            [self.cc('*'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('+'), self.cc('-')], # 0
            [self.cc('-'), self.cc('*'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('+')], # 1
            [self.cc('-'), self.cc('-'), self.cc('*'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('+'), self.cc('-'), self.cc('-'), self.cc('-')], # 2
            [self.cc('-'), self.cc('-'), self.cc('-'), self.cc('*'), self.cc('-'), self.cc('+'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-')], # 3
            [self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('*'), self.cc('-'), self.cc('-'), self.cc('+'), self.cc('-'), self.cc('-')], # 4
            [self.cc('-'), self.cc('-'), self.cc('-'), self.cc('+'), self.cc('-'), self.cc('*'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-')], # 5
            [self.cc('-'), self.cc('-'), self.cc('+'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('*'), self.cc('-'), self.cc('-'), self.cc('-')], # 6
            [self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('+'), self.cc('-'), self.cc('-'), self.cc('*'), self.cc('-'), self.cc('-')], # 7
            [self.cc('+'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('*'), self.cc('-')], # 8
            [self.cc('-'), self.cc('+'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('-'), self.cc('*')], # 9
        ]).cuda()

    def forward(self, input, target): 
        # input is costheta-Size([batch_size, 10])
        # target is the correct label([batch_size, 10] for one_hot)
        batch_size = input.shape[0]
        target = target.view(-1, 1)
        one_hot = (self.crit_map[target.squeeze(-1).long()]).cuda().scatter_(1, target.long(), 1)
        # print(one_hot.size())
        # crit_map = self.crit_map[target]
        loss = torch.norm((input-one_hot),p=2,dim=1)
        loss = torch.mean(loss)
        return loss

    def cc(self, sim, num=1): # num is the number of '+'

        if sim == '+':
            return math.cos(math.acos(-1/9) + self.delta)
        elif sim == '-':
            return math.cos(math.acos(-1/9) - self.delta * num / (9 - num))
        elif sim == '*':
            return 1

if __name__ == '__main__':
    target=torch.Tensor([1,2,2,1,3]).cuda()
    outputs=torch.randn((5,10)).cuda()
    net=L2Loss(0.1)
    loss=net(outputs,target)
    print(loss)