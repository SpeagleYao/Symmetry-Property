import math
import torch
import torch.nn as nn

class LGMLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, margin=0.1, lambda_=0.01):
        super(LGMLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.lambda_ = lambda_
        self.feat_dim = feat_dim
        self.ce = nn.CrossEntropyLoss().cuda()
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))

    def forward(self, feat, labels):
        batch_size = feat.size()[0]
        neg_sqr_dist = -0.5 * torch.sum((feat.unsqueeze(-1) - torch.transpose(self.means, 0, 1))**2, dim=1)
        labels_reshaped = labels.view(labels.size()[0], -1)
        if torch.cuda.is_available():
            ALPHA = torch.zeros(batch_size, self.num_classes).cuda().scatter_(1, labels_reshaped, self.margin)
            K = ALPHA + torch.ones([batch_size, self.num_classes]).cuda()
        else:
            ALPHA = torch.zeros(batch_size, self.num_classes).scatter_(1, labels_reshaped, self.margin)
            K = ALPHA + torch.ones([batch_size, self.num_classes])
        logits_with_margin = torch.mul(neg_sqr_dist, K)
        means_batch = torch.index_select(self.means, dim=0, index=labels)
        likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
        classification_loss = self.ce(logits_with_margin, labels)
        loss = classification_loss + likelihood_reg_loss
        return loss, neg_sqr_dist, logits_with_margin, classification_loss, likelihood_reg_loss