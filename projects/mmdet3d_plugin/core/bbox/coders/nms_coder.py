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
import numpy as np

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, cxcywh2x1y1x2y2, x1y1x2y22cxcywh
import torch.nn.functional as F
from projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder import TMVReidNMSFreeCoder

@BBOX_CODERS.register_module()
class TMVDetNMSCoder(BaseBBoxCoder):
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
                 num_views,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 visible_threshold=None,
                 overlap_thresh=.9,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.visible_threshold = visible_threshold
        self.overlap_thresh=overlap_thresh
        self.num_views=num_views
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
            boxes3d = final_box_preds[thresh_mask]
            scores = final_scores[thresh_mask]
            visibles = final_visibles[thresh_mask]
            labels = final_preds[thresh_mask]

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

        else:
#            raise NotImplementedError('Need to reorganize output as a batch, only '
#                                      'support post_center_range is not None for now!')
            boxes3d = final_box_preds
            scores = final_scores
            visibles = final_visibles
            labels = final_preds

        visibles = visibles > self.visible_threshold
        num_bbox, _  = boxes3d.shape
        boxes3d = boxes3d.reshape(num_bbox, self.num_views+1, 9) #(300, num_views+1, 9)
        boxes3d = boxes3d[:, 1:, [0, 1, 3, 5]] #(300, num_views, 4) #cxcywh
        boxes3d = cxcywh2x1y1x2y2(boxes3d) 

        boxes3d, scores, visibles, labels = self.nms_classwise(boxes3d, scores, labels, visibles, overlap_thresh=self.overlap_thresh, max_boxes=self.max_num, overlap_view=overlap_view)
        
        boxes3d = x1y1x2y22cxcywh(boxes3d) #(N, num_views, 4)
        num_bbox = len(boxes3d)
        final_boxes3d = boxes3d.new_zeros(num_bbox, self.num_views+1, 9)
        final_boxes3d[:, 1:, [0, 1, 3, 5]] = boxes3d
        final_boxes3d = final_boxes3d.reshape(num_bbox, -1)
        
        predictions_dict = {'bboxes': final_boxes3d, 'scores': scores, 'visibles': visibles, 'labels': labels}

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
class TMVReidNMSCoder(TMVReidNMSFreeCoder):
    def __init__(self,
                 max_num=300,
                 num_classes=120,
                 num_views=3,
                 reid_score_threshold=None,
                 cls_score_threshold=None,
                 visible_score_threshold=None,
                 class_agnostic=False,
                 uncertainty=False,
                 overlap_view=1,
                 overlap_thresh=.9):
        super().__init__(max_num, num_classes, num_views, reid_score_threshold, cls_score_threshold, visible_score_threshold, uncertainty)
        self.overlap_thresh = overlap_thresh
        self.visible_threshold = visible_score_threshold
        self.class_agnostic = class_agnostic
        self.overlap_view = overlap_view

    def decode_single(self, cls_scores, reid_scores, idx_scores, visible_scores, bbox_preds, query2ds):
        predictions_dict = super().decode_single(cls_scores, reid_scores, idx_scores, visible_scores, bbox_preds, query2ds)
        boxes3d = predictions_dict['bboxes']
        reid_scores = predictions_dict['reid_scores']
        cls_scores = predictions_dict['cls_scores']
        view_cls_scores = predictions_dict['view_cls_scores']
        idx_scores = predictions_dict['idx_scores']
        visibles = predictions_dict['visibles']
        labels = predictions_dict['labels']
        view_labels = predictions_dict['view_labels']
        query2ds = predictions_dict['query2ds']

        boxes3d = cxcywh2x1y1x2y2(boxes3d) 
        is_vis = visibles>self.visible_score_threshold

        if self.class_agnostic : 
            boxes3d, cls_scores, is_vis, [view_cls_scores, reid_scores, visibles, idx_scores, query2ds, view_labels, labels] = nms(boxes3d, cls_scores, is_vis, [view_cls_scores, reid_scores, visibles, idx_scores, query2ds, view_labels, labels], overlap_thresh=self.overlap_thresh, max_boxes=self.max_num)
            # for align
            boxes3d, cls_scores, is_vis, [view_cls_scores, reid_scores, visibles, idx_scores, query2ds, view_labels], labels = nms_classwise(boxes3d, cls_scores, is_vis, [view_cls_scores, reid_scores, visibles, idx_scores, query2ds, view_labels], labels, overlap_thresh=1.5, max_boxes=self.max_num, overlap_view=self.overlap_view) 
        else : 
            boxes3d, cls_scores, is_vis, [view_cls_scores, reid_scores, visibles, idx_scores, query2ds, view_labels], labels = nms_classwise(boxes3d, cls_scores, is_vis, [view_cls_scores, reid_scores, visibles, idx_scores, query2ds, view_labels], labels, overlap_thresh=self.overlap_thresh, max_boxes=self.max_num, overlap_view=self.overlap_view)
        
        boxes3d = x1y1x2y22cxcywh(boxes3d) #(N, num_views, 4)
 
        predictions_dict = {'bboxes': boxes3d, 'reid_scores': reid_scores, 'idx_scores': idx_scores, 'cls_scores': cls_scores, 'view_cls_scores': view_cls_scores, 'visibles': visibles, 'labels': labels, 'view_labels': view_labels, 'query2ds': query2ds}
        return predictions_dict

def nms(boxes, probs, is_vis, side_infos, overlap_thresh=0.9, max_boxes=300, overlap_view=1):
    # boxes : (num_box, num_cam, 4)
    # probs : (num_box, )
    # is_valids : (num_box, num_cam)
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
    overlap_view_tensor = torch.tensor(overlap_view, device=is_vis.device)
    num_val_views = is_vis.sum(1)

    if len(boxes) == 0:
        return []

    #boxes[np.where(boxes<0)] = 0
    boxes[boxes < 0] = 0
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, :, 0] #(num_box, num_cam)
    y1 = boxes[:, :, 1]
    x2 = boxes[:, :, 2]
    y2 = boxes[:, :, 3]

    #np.testing.assert_array_less(x1, x2)
    #np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    #if boxes.dtype.kind == "i":
    #    boxes = boxes.astype("float")

    # initialize the list of picked indexes 
    pick = []

    # calculate the areas 
    area = (x2 - x1) * (y2 - y1) #(num_box, num_cam)

    # sort the bounding boxes 
    #idxs = np.argsort(probs) #(num_box,)
    _, idxs = torch.sort(probs)
    idxs = idxs.cpu().numpy()

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1_int = torch.maximum(x1[i], x1[idxs[:last]])
        yy1_int = torch.maximum(y1[i], y1[idxs[:last]])
        xx2_int = torch.minimum(x2[i], x2[idxs[:last]])
        yy2_int = torch.minimum(y2[i], y2[idxs[:last]])

        ww_int = torch.maximum(torch.zeros_like(xx1_int), xx2_int - xx1_int)
        hh_int = torch.maximum(torch.zeros_like(yy1_int), yy2_int - yy1_int)

        area_int = ww_int * hh_int

        area_union = area[i] + area[idxs[:last]] - area_int

        overlap = area_int / (area_union + 1e-6)
        overlap = overlap.cpu().numpy()

        # delete all indexes from the index list that have
        num_thresh_view = torch.minimum(overlap_view_tensor, torch.maximum(num_val_views[idxs[:last]], num_val_views[i])).cpu().numpy()
        valid = (overlap >= overlap_thresh) & is_vis[i].cpu().numpy() & is_vis[idxs[:last]].cpu().numpy()
        idxs = np.delete(idxs, np.concatenate(([last], np.where(np.sum(valid, 1) >= num_thresh_view)[0])))

        if len(pick) >= max_boxes:
            break

    boxes = boxes[pick]
    probs = probs[pick]
    is_vis = is_vis[pick]
    side_infos = [side_info[pick] for side_info in side_infos]

    return boxes, probs, is_vis, side_infos

def nms_classwise(boxes, scores, num_val_view, side_infos, labels, overlap_thresh=0.9, max_boxes=300, overlap_view=1):
    unique_labels = torch.unique(labels)  # Get unique class labels
    
    final_boxes = []
    final_scores = []
    final_num_val_view = []
    final_labels = []
    final_side_infos = [[] for _ in range(len(side_infos))]
    
    for label in unique_labels:
        # Select boxes, scores, visibles, and indices for the current class
        mask = labels == label
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        class_num_val_view = num_val_view[mask]
        class_side_infos = [side_info[mask] for side_info in side_infos]
        
        # Apply NMS for the current class
        class_boxes, class_scores, class_num_val_view, class_side_infos = nms(class_boxes, class_scores, class_num_val_view, class_side_infos,
                                                             overlap_thresh=overlap_thresh, max_boxes=max_boxes, overlap_view=overlap_view)
        
        final_boxes.append(class_boxes)
        final_scores.append(class_scores)
        final_num_val_view.append(class_num_val_view)
        for i in range(len(side_infos)) : final_side_infos[i].append(class_side_infos[i])
        final_labels.extend([label] * len(class_boxes))
    
    final_boxes = torch.cat(final_boxes)
    final_scores = torch.cat(final_scores)
    final_num_val_view = torch.cat(final_num_val_view)
    final_side_infos_list = [torch.cat(final_side_info) for final_side_info in final_side_infos]
    final_labels = torch.tensor(final_labels)
    
    return final_boxes, final_scores, final_num_val_view, final_side_infos_list, final_labels

