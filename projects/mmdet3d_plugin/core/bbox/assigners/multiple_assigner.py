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
class MultipleAssignerMtvReid2D(BaseAssigner):
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

        iou = det_bbox_cxcywh.new_zeros((num_det, num_gt, num_views)) #(900, num_gt, 3)
        for i in range(num_views):
            iou[..., i] = bbox_overlaps(det_bbox[:, i], gt_bbox[:, i]) #(900, 4) #(num_gt, 4) -> (900, num_gt)
        is_big_iou = iou > .5 #(900, num_gt, 3)

        ###
        min_cost, min_cost_gt_inds = cost.to(det_bbox_cxcywh.device).min(-1) #(900,), #(900,)
        matched_row_inds = torch.where(min_cost < self.valid_cost_thresh)[0] #(num_valid, )
        matched_col_inds = min_cost_gt_inds[matched_row_inds] #(num_valid)

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
        ###
        return AssignResult(num_gt, assigned_gt_inds, None, labels=assigned_labels)
