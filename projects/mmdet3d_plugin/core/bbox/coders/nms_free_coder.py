# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
from copy import deepcopy

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, get_box_form_pred_idx
import torch.nn.functional as F

@BBOX_CODERS.register_module()
class TMVDetNMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        pass

    def encode(self):
        pass

    def decode_single(self, cls_scores, visible_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        visible_scores = visible_scores.sigmoid()[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_visibles = visible_scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            visibles = final_visibles[mask]
            labels = final_preds[mask]
            predictions_dict = {'bboxes': boxes3d, 'scores': scores, 'visibles': visibles, 'labels': labels}

        else:
#            raise NotImplementedError('Need to reorganize output as a batch, only '
#                                      'support post_center_range is not None for now!')
            boxes3d = final_box_preds
            scores = final_scores
            visibles = final_visibles
            labels = final_preds
            predictions_dict = {'bboxes': boxes3d, 'scores': scores, 'visibles': visibles, 'labels': labels}


        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_visible_scores = preds_dicts['all_visible_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_visible_scores[i], all_bbox_preds[i]))
        return predictions_list


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {'bboxes': boxes3d, 'scores': scores, 'labels': labels}

        else:
            raise NotImplementedError('Need to reorganize output as a batch, only '
                                      'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list


@BBOX_CODERS.register_module()
class NMSFreeClsCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        # cls_scores = cls_scores.sigmoid()
        # scores, indexs = cls_scores.view(-1).topk(max_num)
        # labels = indexs % self.num_classes
        # bbox_index = indexs // self.num_classes
        # bbox_preds = bbox_preds[bbox_index]

        cls_scores, labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = labels[indexs]
        bbox_preds = bbox_preds[indexs]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {'bboxes': boxes3d, 'scores': scores, 'labels': labels}

        else:
            raise NotImplementedError('Need to reorganize output as a batch, only '
                                      'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list

@BBOX_CODERS.register_module()
class TMVReidNMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 max_num=300,
                 num_classes=120,
                 num_views=3,
                 reid_score_threshold=None,
                 cls_score_threshold=None,
                 reid_sigmoid=True,
                 H=None,
                 W=None,
                ):
        self.max_num = max_num
        self.num_views = num_views
        self.reid_score_threshold = reid_score_threshold
        self.cls_score_threshold = cls_score_threshold
        self.num_classes = num_classes
        self.reid_sigmoid = reid_sigmoid
        if not self.reid_sigmoid : 
            self.pdist = torch.nn.PairwiseDistance(p=1)
        self.H = H
        self.W = W
        pass

    def encode(self):
        pass

    def decode_single(self, cls_scores, reid_scores, visible_scores, bbox_preds, query2ds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        if not self.reid_sigmoid :
            Q = len(query2ds)
            normalized_query2ds = deepcopy(query2ds) 
            normalized_query2ds[..., 0] /= self.W
            normalized_query2ds[..., 1] /= self.H
            normalized_query2ds = normalized_query2ds.reshape(Q, -1)

            normalized_bbox_preds = deepcopy(bbox_preds[..., :2]) 
            normalized_bbox_preds[..., 0] /= self.W
            normalized_bbox_preds[..., 1] /= self.H
            normalized_bbox_preds = normalized_bbox_preds.reshape(Q, -1)

            dist = self.pdist(normalized_query2ds, normalized_bbox_preds) #(900,)
            reid_scores, indexs = (1/dist).topk(max_num) #(300,), #(300,)
        else : 
            reid_scores, indexs = reid_scores.sigmoid().topk(max_num) #(300,), #(300,)
            #reid_scores = reid_scores.sigmoid() #(300,), #(300,)
            #indexs = torch.where(reid_scores>-1)[0]

        soft_cls_scores, labels = F.softmax(cls_scores, dim=-1).max(-1) #(900,), #(900,)
        #soft_cls_scores, indexs = soft_cls_scores.view(-1).topk(max_num) #(300,), #(300,)

        labels = labels[indexs]
        cls_scores_sig = cls_scores.sigmoid()[indexs, labels]
        bbox_preds = bbox_preds[indexs]
        visible_scores = visible_scores.sigmoid()[indexs]
        #reid_scores = reid_scores.sigmoid()[indexs]
        query2ds = query2ds[indexs]

        final_box_preds = bbox_preds
        final_reid_scores = reid_scores
        final_cls_scores = cls_scores_sig
        final_visibles = visible_scores
        final_preds = labels
        final_query2ds = query2ds

        # use score threshold
        mask = torch.ones_like(final_cls_scores, dtype=torch.bool) #(300, )
        if self.cls_score_threshold is not None:
            cls_thresh_mask = final_cls_scores > self.cls_score_threshold
            mask &= cls_thresh_mask

        if self.reid_score_threshold is not None:
            reid_thresh_mask = final_reid_scores > self.reid_score_threshold
            mask &= reid_thresh_mask

        boxes3d = final_box_preds[mask]
        reid_scores = final_reid_scores[mask]
        cls_scores = final_cls_scores[mask]
        visibles = final_visibles[mask]
        labels = final_preds[mask]
        query2ds = final_query2ds[mask]
        predictions_dict = {'bboxes': boxes3d, 'reid_scores': reid_scores, 'cls_scores': cls_scores, 'visibles': visibles, 'labels': labels, 'query2ds': query2ds}

        return predictions_dict

    def decode(self, preds_dicts, pred_box):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_reid_scores = preds_dicts['all_reid_scores'][-1]
        all_visible_scores = preds_dicts['all_visible_scores'][-1]
        all_idx_scores = preds_dicts['all_idx_scores'][-1]
        all_query2ds = preds_dicts['all_query2ds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            pred_idx_scores, pred_idx = F.softmax(all_idx_scores[i], dim=-1).max(-1) #(900, 3), (900, 3)
            bbox_preds = get_box_form_pred_idx(pred_box, pred_idx, self.num_views) #(900, 3, 4)
            predictions_list.append(self.decode_single(all_cls_scores[i], all_reid_scores[i], all_visible_scores[i], bbox_preds, all_query2ds[i]))
        return predictions_list


