# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from einops import rearrange
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
#from mmdet3d.core import bbox3d2result
from projects.mmdet3d_plugin.core.bbox.transformer import bboxmtv2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

from .vedet import VEDet

@DETECTORS.register_module()
class TMVReid(VEDet):
    def forward_train(self, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None, maps=None, img=None, rpn_cxcywh=None, rpn_emb=None, rpn_prob=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        inputs = (rpn_cxcywh, rpn_emb, rpn_prob)

        img_metas[0]['img'] = img
        losses = dict()
        losses_pts = self.forward_pts_train(inputs, gt_bboxes_3d, gt_labels_3d, maps, img_metas)
        losses.update(losses_pts)
        return losses

    def simple_test(self, img_metas, img=None, gt_map=None, rescale=False, rpn_cxcywh=None, rpn_emb=None, rpn_prob=None):
        """Test function without augmentaiton."""
        inputs = (rpn_cxcywh, rpn_emb, rpn_prob)

        results_list = [dict() for i in range(len(img_metas))]
        results = self.simple_test_pts(inputs, img_metas, gt_map, rescale=rescale)
        if 'bbox_results' in results:
            for result_dict, pts_bbox in zip(results_list, results['bbox_results']):
                result_dict['pts_bbox'] = pts_bbox

        return results_list

    def simple_test_pts(self, x, img_metas, gt_map=None, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas, is_test=True)
        results = dict()
        if outs.get('all_cls_scores', None) is not None:
            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
            #bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
            bbox_results = [bboxmtv2result(bboxes, visibles, scores, labels) for bboxes, visibles, scores, labels in bbox_list]
            results['bbox_results'] = bbox_results

        if gt_map is not None:
            seg_results = self.compute_seg_iou(outs)
            results['seg_results'] = seg_results

        return results


