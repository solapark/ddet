import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class PosLoss(nn.Module):
    def __init__(self, alpha, loss_weight, valid_cost, pos_thresh, neg_thresh):
        super(PosLoss, self).__init__()
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.valid_cost = valid_cost
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        self.triplet_loss = nn.TripletMarginLoss(margin=self.alpha, p=1)

    def forward(self, attn, query2gt, query, key): #(900, 3, 300), #(900, Ngt),  #(900, 3, 2), #(300, 3, 2)
        query2gt_min = query2gt.min(-1)[0]
        valid_idx = (query2gt_min < self.valid_cost)
        query = query[valid_idx]
        attn = attn[valid_idx]
        num_sample, num_view, num_key = attn.shape

        query = query.transpose(1,0) #(3,900,2)
        key = key.transpose(1,0) #(3,300,2)
        attn = attn.transpose(1,0) #(3,900,300)
        query2rpn = torch.stack([torch.cdist(q, k, p=2) for q, k in zip(query, key)], 0) #(3, 900, 300)
        
        sorted_dist, sorted_idx = torch.sort(query2rpn, -1)  #(3, 900, 300), (3, 900, 300)
        pos_idx = torch.randint(0, self.pos_thresh, (num_view*num_sample, ))
        neg_idx = torch.randint(self.pos_thresh, self.neg_thresh, (num_view*num_sample, ))

        attn = attn.reshape(num_view*num_sample, num_key)
        sorted_idx = sorted_idx.reshape(num_view*num_sample, num_key)
        query2rpn = query2rpn.reshape(num_view*num_sample, num_key)

        query_range = torch.arange(num_view*num_sample).to(attn.device)
        
        pos = attn[query_range, sorted_idx[query_range, pos_idx]].sigmoid().unsqueeze(-1) #(N, 1)
        neg = attn[query_range, sorted_idx[query_range, neg_idx]].sigmoid().unsqueeze(-1) #(N, 1)
        anchor = torch.ones_like(pos)
        return self.loss_weight * self.triplet_loss(anchor, pos, neg) 
