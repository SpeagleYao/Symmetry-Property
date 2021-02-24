import torch
import torch.nn as nn

class SimilarityPreservingLoss(nn.Module):
    def __init__(self,feat_dim=256):
        super(SimilarityPreservingLoss, self).__init__()
        self.feat_dim=feat_dim

    def forward(self, features, target):
        assert features.shape[-1]==self.feat_dim, 'feat_dim should be consistent'
        batch_size=features.shape[0]
        assert batch_size==target.shape[0], 'batch size should be consistent'

        tar_rows = target.repeat((batch_size,1))
        tar_cols = target.reshape(-1,1).repeat((1,batch_size))
        sp_tar = ((tar_rows-tar_cols)==0).float()

        feats_norm = torch.norm(features, p=2, dim=1)
        norm_rows = feats_norm.repeat((batch_size,1))
        norm_cols = feats_norm.reshape(-1,1).repeat((1,batch_size))
        norm_mat = norm_rows.mul(norm_cols)
        feats_pointmul = features.mm(features.t())
        sp_feats = feats_pointmul / norm_mat

        loss = torch.norm((sp_tar - sp_feats),p=2)

        return loss

if __name__=='__main__':
    target=torch.Tensor([1,2,2,1,3])
    feats=torch.randn((5,256))
    net=SimilarityPreservingLoss(feat_dim=256)
    loss=net(feats,target)
    print(loss)