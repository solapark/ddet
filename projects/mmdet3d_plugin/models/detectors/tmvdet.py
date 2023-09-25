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
class TMVDet(VEDet):
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

