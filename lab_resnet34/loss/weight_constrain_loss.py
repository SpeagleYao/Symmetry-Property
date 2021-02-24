import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WeightConstrainLoss(nn.Module):

    def __init__(self, gamma=0.05):
        super(WeightConstrainLoss, self).__init__()
        self.gamma = gamma

    def forward(self, weight, target):
        batch_size = target.shape[0]     # 256
        feat_dim = weight.shape[0]      # 512

        target = target.reshape(-1, 1)  # 256,1
        weight = weight.reshape(-1, *weight.shape).repeat(batch_size, 1, 1)  # 256,512,10

        one_hot = torch.zeros(batch_size, 10).cuda().scatter_(1, target.long(), 1)  # 256,10
        one_hot = one_hot.unsqueeze(1).repeat(1, feat_dim, 1)  # 256,512,10
        zero_hot = 1 - one_hot  # 256,512,10

        true_class = torch.masked_select(weight, one_hot.byte()).reshape(batch_size, feat_dim, -1)  # 256,512,1
        false_class = torch.masked_select(weight, zero_hot.byte()).reshape(batch_size, feat_dim, -1)  # 256,512,9

        cos_theta = torch.matmul(true_class.permute(0, 2, 1), false_class).squeeze()  # 256,1,512 x 256,512,9 -> 256,9
        cos_theta = cos_theta * self.gamma

        loss_exp = torch.sum(torch.exp(cos_theta), dim=1)  # 256
        loss_log = torch.log(loss_exp)
        loss = loss_log / self.gamma

        loss = torch.mean(loss)

        return loss


class WeightLoss(nn.Module):

    def __init__(self, __delta__):
        super(WeightLoss, self).__init__()
        self.delta = __delta__
        '''
        self.importance_matrix = torch.Tensor([
           # 0  1  2  3  4  5  6  7  8  9
            [1, 1, 3, 1, 1, 1, 1, 1, 2, 1], # 0
            [1, 1, 1, 1, 1, 1, 2, 1, 2, 3], # 1
            [2, 1, 1, 3, 2, 1, 2, 1, 1, 1], # 2
            [1, 1, 3, 1, 2, 2, 2, 1, 1, 1], # 3
            [1, 1, 3, 2, 1, 1, 2, 1, 1, 1], # 4
            [1, 1, 3, 3, 1, 1, 1, 1, 1, 1], # 5
            [1, 1, 3, 2, 1, 1, 1, 1, 1, 1], # 6
            [1, 1, 3, 3, 2, 2, 1, 1, 1, 1], # 7
            [2, 1, 3, 2, 1, 1, 1, 1, 1, 1], # 8
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 1]  # 9
        ]).cuda()
        '''

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

    def forward(self, weight):

        cos_map = torch.matmul(weight.t(), weight)
        crit_map = self.crit_map.cuda()
        # print(crit_map)
        # power = torch.mul((cos_map - crit_map) ** 2, self.importance_matrix)
        power = (cos_map - crit_map) ** 2
        #power = torch.mul(cos_map ** 2, self.importance_matrix)
        loss = torch.sqrt(torch.sum(power))

        return loss
    
    def cc(self, sim, num=1): # num is the number of '+'

        if sim == '+':
            return math.cos(math.acos(-1/9) + self.delta)
        elif sim == '-':
            return math.cos(math.acos(-1/9) - self.delta * num / (9 - num))
        elif sim == '*':
            return 1

if __name__=='__main__':

    weight = torch.randn(512, 10).cuda()
    weight = F.normalize(weight,p=2,dim=0)
    criterion=WeightLoss()
    loss = criterion(weight)
    print(loss)