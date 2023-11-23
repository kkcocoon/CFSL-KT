import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# In[proposed kernal triplet hard loss]:

def guassian_kernal(inputs, kernal_mul=2.0, kernal_num=5, fix_sigma=None):
    """ Gaussion kernal for KernalTripletHardLoss
    """
    n_samples = int(inputs.size()[0])

    total0 = inputs.unsqueeze(0).expand(int(inputs.size(0)), int(inputs.size(0)), int(inputs.size(1)))
    total1 = inputs.unsqueeze(1).expand(int(inputs.size(0)), int(inputs.size(0)), int(inputs.size(1)))

    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernal_mul ** (kernal_num // 2)

    bandwidth_list = [bandwidth * (kernal_mul ** i) for i in range(kernal_num)]

    kernel_val = [1-torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)

class KernalTripletHardLoss(nn.Module):
    """Kernal triplet hard loss.
    Args:
       margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(KernalTripletHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
           inputs: feature matrix with shape (batch_size, feat_dim)
           targets: ground truth labels with shape (batch_size,),  not one-hot martix
        """
        n = inputs.size(0)
        
        # adding normalization
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)

        dist = guassian_kernal(inputs, kernal_mul=2, kernal_num=5, fix_sigma=None)

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
     
# In[self-designed feature encoder network architecture]:

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class MyNetwork(nn.Module):
    def __init__(self, source_band_num, target_band_num, source_class_num, target_class_num, feature_dim=256):
        super(MyNetwork, self).__init__()

        self.source_mapping = Mapping(source_band_num, 128)
        self.target_mapping = Mapping(target_band_num, 128)

#        self.conv1 = nn.Conv2d(51, 128, 3, padding=0)
        self.conv1 = nn.Conv2d(128, 128, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(128, affine=True)

        self.conv2 = nn.Conv2d(128, 128, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(128, affine=True)

        self.conv3 = nn.Conv2d(128, 64, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(64, affine=True)

        self.fc1 = nn.Linear(1600, feature_dim)  # according to xfc.size()

        self.source_classifer = nn.Linear(feature_dim, source_class_num)

        self.target_classifer = nn.Linear(feature_dim, target_class_num)

    def forward(self, x, domain='target'):
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x)

        # 9x9 x 128
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 7x7 x 128
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 5x5 x 128
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 5x5 x 64
        xfc = x.view(x.size(0), -1)
        # print(xfc.size())  # to set the first dimension of self.fc1

        features = self.fc1(xfc)   # (batch_size, feature_dim)
        # print(features.size())
        if domain == 'target':
           output = self.target_classifer(features) #(batch_size, target_class)
        else:
           output = self.source_classifer(features) #(batch_size ,source_class)

        return features, output
        
     
      
# In[DomainClassifier]:       
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__() 
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(256, 1) 
        self.layer0 = nn.Linear(256,1024)

    def forward(self, x, iter_num=0):
        x = self.layer0(x)
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y
 