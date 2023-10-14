# ------------------------------------------------------------------------
# Copyright (c) 2023 CAPP, Sola Park. All Rights Reserved.
## ------------------------------------------------------------------------
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
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from projects.mmdet3d_plugin.core.bbox.util import cxcywh2x1y1x2y2

@BBOX_ASSIGNERS.register_module()
class EpipolarAssignerMtvReid2D(BaseAssigner):
    def __init__(self,
                 valid_cost_thresh):
        self.valid_cost_thresh = valid_cost_thresh

    def assign(self, det_rpn_idx, det_bbox_cxcywh, gt_rpn_idx, gt_bbox_cxcywh, gt_labels, cost):
        det_bbox = cxcywh2x1y1x2y2(det_bbox_cxcywh)
        gt_bbox = cxcywh2x1y1x2y2(gt_bbox_cxcywh)

        num_gt = len(gt_bbox)
        num_det, num_views = det_rpn_idx.shape #900, 3
        assigned_gt_inds = det_rpn_idx.new_full((num_det, ), 0, dtype=torch.long) #(900, )
        assigned_labels = det_rpn_idx.new_full((num_det, ), 0, dtype=torch.long) #(900, )
        assigned_rpn_idx = deepcopy(det_rpn_idx).to(torch.long) #(900, 3)

        is_small_cost = cost.to(det_bbox_cxcywh.device) < self.valid_cost_thresh #(900, num_gt)
        iou = det_bbox_cxcywh.new_zeros((num_det, num_gt, num_views)) #(900, num_gt, 3)
        for i in range(num_views):
            iou[..., i] = bbox_overlaps(det_bbox[:, i], gt_bbox[:, i]) #(900, 4) #(num_gt, 4) -> (900, num_gt)

        is_big_iou = iou > .5 #(900, num_gt, 3)
        is_many_big_iou = is_big_iou.sum(-1) > 1  #(900, num_gt)
        is_valid = (is_small_cost & is_many_big_iou) #(900, num_gt)

        is_min_cost = is_valid.new_zeros(is_valid.shape).to(torch.bool) #(900, num_gt)
        cost[~is_valid] = 100.
        is_min_cost[torch.arange(num_det), cost.min(-1)[1]] = True

        is_valid = (is_valid & is_min_cost)

        matched_row_inds, matched_col_inds = torch.where(is_valid) #23, 23
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        valid_gt_rpn_idx = gt_rpn_idx[matched_col_inds] #(num_gt) -> #(23, 3)
        valid_det_rpn_idx = det_rpn_idx[matched_row_inds] #(num_gt) -> #(23, 3)
        valid_is_big_iou = is_big_iou[matched_row_inds, matched_col_inds] #(23, 3)
        valid_is_small_iou = ~valid_is_big_iou #(23, 3)

        valid_assigned_rpn_idx = deepcopy(valid_det_rpn_idx)
        valid_assigned_rpn_idx[valid_is_small_iou] = valid_gt_rpn_idx[valid_is_small_iou]

        assigned_rpn_idx[matched_row_inds] = valid_assigned_rpn_idx
        self.assigned_rpn_idx = assigned_rpn_idx
        return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)

    def unique_row(self, matched_row_inds, matched_col_inds, cost):
        max_cost_indices = {}
        
        for row, col in zip(matched_row_inds, matched_col_inds):
            if row in max_cost_indices:
                if cost[row, col] > cost[row, max_cost_indices[row]]:
                    max_cost_indices[row] = col
            else:
                max_cost_indices[row] = col

        result_matched_row_inds = []
        result_matched_col_inds = []

        for row, col in zip(matched_row_inds, matched_col_inds):
            if col == max_cost_indices[row]:
                result_matched_row_inds.append(row)
                result_matched_col_inds.append(col)

        return matched_row_inds.new_tensor(result_matched_row_inds), matched_row_inds.new_tensor(result_matched_col_inds)

