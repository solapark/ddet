import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class TripletLoss(nn.Module):
    def __init__(self, use_sigmoid, loss_weight):
        super(TripletLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid triplet loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, score_positive, score_negative, margin):
        # Calculate the loss
        if self.use_sigmoid :
            score_positive_sig = score_positive.sigmoid()
            score_negative_sig = score_negative.sigmoid()
        loss = self.loss_weight * (score_positive_sig - score_negative_sig - margin).abs().mean()
        return loss
