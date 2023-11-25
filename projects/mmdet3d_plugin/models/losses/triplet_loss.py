import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class TripletLoss(nn.Module):
    def __init__(self, num_views, alpha, loss_weight, valid_cost):
        super(TripletLoss, self).__init__()
        self.num_views = num_views
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.valid_cost = valid_cost

        self.triplet_loss = nn.TripletMarginLoss(margin=self.alpha, p=2)

    def forward(self, X, cost): #(num_views, 900, 256), #(900, num_gt)
        feat = X[:, cost.min(-1)[0] < self.valid_cost]

        _, num_sample, _ = feat.shape
        query_range = list(range(num_sample))
        indices = torch.randint(0, num_sample-1, (num_sample,))

        anchor_v, pos_v, neg_v = torch.stack([torch.randperm(self.num_views) for _ in range(num_sample)]).transpose(1, 0).to(feat.device)
        neg_idx = torch.tensor([(query_range[:i]+query_range[i+1:])[idx] for i, idx in enumerate(indices)]).to(feat.device) #(900, )

        query_range = torch.tensor(query_range).to(feat.device)
        anchor = feat[anchor_v, query_range]
        pos = feat[pos_v, query_range]
        neg = feat[neg_v, neg_idx]

        return self.loss_weight * self.triplet_loss(anchor, pos, neg) 


