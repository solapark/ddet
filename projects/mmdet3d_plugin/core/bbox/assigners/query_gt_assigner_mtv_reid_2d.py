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
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class QueryGtAssignerMtvReid2D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 visible_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 query_cost=dict(type='QueryCost', weight=1.0),
                 align_with_loss=False,
                 pc_range=None,
                 thresh=.1):
        self.cls_cost = build_match_cost(cls_cost)
        self.visible_cost = build_match_cost(visible_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.query_cost = build_match_cost(query_cost)
        self.align_with_loss = align_with_loss
        self.pc_range = pc_range
        self.thresh = thresh

    def assign(self, norm_pred_cxcy, cls_pred, visible_pred, reid_pred, idx_pred, gt_proj_cxcy, gt_labels, gt_visibles, gt_idx, gt_bboxes_ignore=None, eps=1e-7, img_shape=None):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_proj_cxcy.size(0), norm_pred_cxcy.size(0) #num_gt, num_pred

        # 1. assign -1 by default
        assigned_gt_inds = cls_pred.new_full((num_bboxes, ), -1, dtype=torch.long) #(900, )
        assigned_labels = cls_pred.new_full((num_bboxes, ), -1, dtype=torch.long) #(900, )
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        cost = 0

        # classification and bboxcost.
        if self.cls_cost.weight > 0 :
            cost = self.cls_cost(cls_pred, gt_labels)
            
        # visible_cost.
        #visible_cost = []
        #for i in range(gt_visibles.shape[-1]):
        #    v_cost = self.visible_cost(visible_pred[:, i:i+1], gt_visibles[:, i]) #(900, 50)
        #    visible_cost.append(v_cost)
        #visible_cost = torch.mean(torch.stack(visible_cost), 0) #(2, 900, 50) -> #(900, 50)
            
        #visible_cost = self.visible_cost(visible_pred, gt_visibles) #(900, 50, 2)
        #visible_cost = torch.mean(visible_cost, -1) #(900, 50)

        #query cost
        H, W = img_shape
        if self.query_cost.weight > 0 :
            normalized_gt_cxcy = deepcopy(gt_proj_cxcy) #(num_gt, num_cam, 2) projected cxcy by DLT
            normalized_gt_cxcy[..., 0] /= W
            normalized_gt_cxcy[..., 1] /= H
            normalized_gt_cxcy = torch.nan_to_num(normalized_gt_cxcy).flatten(1,2) #(num_gt, num_cam*2)
            query_cost = self.query_cost(norm_pred_cxcy, normalized_gt_cxcy, 1-gt_visibles) #(900, num_gt)
            cost = cost + query_cost

        # 3. do matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu() #(num_det = 900, num_gt)
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)

        min_cost, min_cost_gt_inds = cost.min(-1) #(900,), #(900,)
        matched_row_inds = torch.where(min_cost < self.thresh) #(num_valid, )
        matched_col_inds = min_cost_gt_inds[matched_row_inds] #(num_valid)
        
        matched_row_inds = matched_row_inds[0].to(cls_pred.device)
        matched_col_inds = matched_col_inds.to(cls_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
