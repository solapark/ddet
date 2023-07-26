# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.iou_calculators import bbox_overlaps


@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class BBoxMtv2DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., pred_size=10, num_views=2):
        self.weight = weight
        self.pred_size = pred_size
        self.num_views = num_views

    def __call__(self, bbox_pred, gt_bboxes, gt_visible):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), which are all in range [0, 1]. Shape
                [num_query, (num_views+1)*10].
            gt_bboxes (Tensor): Ground truth boxes with normalized coordinates
                (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), which are all in range [0, 1]. Shape
                coordinates (cx, cy, w, h). Shape 
                [num_gt, (num_view+1)*10].
            gt_visible (Tensor): visibility of Ground truth boxes
                1=non_visible, 0=visible
                Shape [num_gt, num_view].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_pred = len(bbox_pred)
        num_gt = len(gt_bboxes)
        bbox_pred = bbox_pred[:, self.pred_size:].reshape(num_pred, self.num_views, self.pred_size) #(900, 2, 10)
        gt_bboxes = gt_bboxes[:, self.pred_size:].reshape(num_gt, self.num_views, self.pred_size) #(num_inst, 2, 10)

        bbox_cost_list = [] #(num_gt, num_pred, 1)
        for i in range(num_gt):
            #cur_gt_visible = gt_visible[i] #(2, )
            visible_mask = gt_visible[i].to(torch.bool) #(2, )
            cur_gt_bboxes = gt_bboxes[i][visible_mask][:,  [0,1,2,5]].reshape(1, -1) #(1, num_valid_views*4)
            cur_bbox_pred = bbox_pred[:, visible_mask][:, :, [0,1,2,5]].reshape(num_pred, -1) #(900, num_valid_views*4)

            bbox_cost = torch.cdist(cur_bbox_pred, cur_gt_bboxes, p=1).squeeze(-1) #(900, )
            bbox_cost_list.append(bbox_cost)

        bbox_cost = torch.stack(bbox_cost_list, -1) #(num_pred, num_gt)
        return bbox_cost * self.weight
