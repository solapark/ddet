# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
from copy import deepcopy

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost


@BBOX_ASSIGNERS.register_module()
class RpnAssigner:
    def __init__(self,
                 rpn_cost,
                 pos_thresh,
                 neg_thresh):
        self.rpn_cost = build_match_cost(rpn_cost)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.max_dist = 2**.5

    def assign(self, norm_query_cxcy, rpn_cxcywh, img_shape):
        norm_query_cxcy = norm_query_cxcy.transpose(1, 0) #(900, 3, 2)
        num_query, num_views = norm_query_cxcy.shape[:2] 

        normalized_rpn_cxcy = deepcopy(rpn_cxcywh[..., :2]) #(num_inst, num_cam, 2) projected cxcy by DLT
        H, W = img_shape
        normalized_rpn_cxcy[..., 0] /= W
        normalized_rpn_cxcy[..., 1] /= H
        dist = self.rpn_cost(norm_query_cxcy, normalized_rpn_cxcy) #(3, 900, 300) #(num_view, num_query, num_rpn)
        dist /= self.max_dist
        score = 1.-dist

        all_pos_idx, all_neg_idx, all_margin = [], [], []
        for i in range(num_views):
            for query_idx in range(num_query):
                pos_candidates = torch.where(score[i, query_idx] >= self.pos_thresh)[0]
                neg_candidates = torch.where((self.pos_thresh > score[i, query_idx]) & (score[i, query_idx] >= self.neg_thresh))[0]

                if len(pos_candidates) > 0 and len(neg_candidates) > 0:
                    pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,))]
                    neg_idx = neg_candidates[torch.randint(0, len(neg_candidates), (1,))]
                    margin = score[i, query_idx, pos_idx] - score[i, query_idx, neg_idx]

                    all_pos_idx.append([query_idx, i, pos_idx])
                    all_neg_idx.append([query_idx, i, neg_idx])
                    all_margin.append(margin)

        all_pos_idx = score.new_tensor(all_pos_idx, dtype=torch.long) #(N, 3) #(query_idx, view_idx, rpn_idx)
        all_neg_idx = score.new_tensor(all_neg_idx, dtype=torch.long) #(N, 3) #(query_idx, view_idx, rpn_idx)
        all_margin = score.new_tensor(all_margin) #(N, )

        return all_pos_idx.T.tolist(), all_neg_idx.T.tolist(), all_margin
