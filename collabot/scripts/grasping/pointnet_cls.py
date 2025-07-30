import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

#large object grasp net using PointNet as backbone
class LoGNet_pn(nn.Module):
    def __init__(self,n_bin=12):
        super(LoGNet_pn, self).__init__()

        channel = 3

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, k)
        # self.dropout = nn.Dropout(p=0.4)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.to_approach = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)        )
        self.to_angle = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_bin)        )
        

        self.to_trans = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)        )

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        
        approach = self.to_approach(x)
        #normalize approach
        approach = approach / torch.norm(approach, dim=1, keepdim=True)

        angle = self.to_angle(x)


        trans_grasp = self.to_trans(x)


        return approach,angle,trans_grasp

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    import torch
    model = LoGNet_pn()
    xyz = torch.rand(2, 3, 1024)
    approach,angle,trans,score = model(xyz)
    print(approach)