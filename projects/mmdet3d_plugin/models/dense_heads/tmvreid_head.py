# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, get_box_form_pred_idx
import numpy as np
from pytorch3d import transforms as tfms

from .vedet_head import VEDetHead
from .tmvdet_head import TMVDetHead

@HEADS.register_module()
class TMVReidHead(TMVDetHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 in_channels,
                 tp_train_only=False,
                 rpn_mask=False,
                 debug=False,
                 pred_size=10,
                 num_input=300,
                 input_emb_size=128,
                 idx_emb_size=127,
                 num_classes=10,
                 num_query=900,
                 position_range=None,
                 det_transformer=None,
                 det_feat_idx=0,
                 sync_cls_avg_factor=False,
                 grid_offset=0.0,
                 input_ray_encoding=None,
                 input_pts_encoding=None,
                 output_det_encoding=None,
                 output_seg_encoding=None,
                 emb_intrinsics=False,
                 code_weights=None,
                 num_decode_views=2,
                 reg_channels=None,
                 bbox_coder=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_visible=None,
                 loss_reid=None,
                 loss_idx=None,
                 loss_iou=None,
                 loss_seg=None,
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 valid_range=[0.0, 1.0],
                 init_cfg=None,
                 shared_head=True,
                 cls_hidden_dims=[],
                 reg_hidden_dims=[],
                 reid_hidden_dims=[],
                 idx_hidden_dims=[],
                 visible_hidden_dims=[],
                 with_time=False,
                 rpn_idx_learnable=True, 
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        self.debug = debug
        self.tp_train_only = tp_train_only
        self.rpn_mask = rpn_mask

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.num_decode_views = num_decode_views
        self.reg_channels = reg_channels if reg_channels else self.code_size
        if loss_cls is not None:
            self.bg_cls_weight = 0
            self.sync_cls_avg_factor = sync_cls_avg_factor
            class_weight = loss_cls.get('class_weight', None)
            if class_weight is not None and (self.__class__ is VEDetHead):
                assert isinstance(class_weight, float), 'Expected ' \
                    'class_weight to have type float. Found ' \
                    f'{type(class_weight)}.'
                # NOTE following the official DETR rep0, bg_cls_weight means
                # relative classification weight of the no-object class.
                bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
                assert isinstance(bg_cls_weight, float), 'Expected ' \
                    'bg_cls_weight to have type float. Found ' \
                    f'{type(bg_cls_weight)}.'
                class_weight = torch.ones(num_classes + 1) * class_weight
                # set background class as the last indice
                class_weight[num_classes] = bg_cls_weight
                loss_cls.update({'class_weight': class_weight})
                if 'bg_cls_weight' in loss_cls:
                    loss_cls.pop('bg_cls_weight')
                self.bg_cls_weight = bg_cls_weight

            if train_cfg:
                assert 'assigner' in train_cfg, 'assigner should be provided '\
                    'when train_cfg is set.'
                self.assigner = build_assigner(train_cfg['assigner'])
                # DETR sampling=False, so use PseudoSampler
                sampler_cfg = dict(type='PseudoSampler')
                self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_input = num_input
        self.last_timestamp = None
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.valid_range = valid_range
        super(VEDetHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        self.loss_cls = build_loss(loss_cls) if loss_cls else None
        self.loss_bbox = build_loss(loss_bbox) if loss_bbox else None
        self.loss_iou = build_loss(loss_iou) if loss_iou else None
        self.loss_seg = build_loss(loss_seg) if loss_seg else None
        self.loss_reid = build_loss(loss_reid) if loss_cls else None
        self.loss_idx = build_loss(loss_idx) if loss_cls else None
        self.loss_visible = build_loss(loss_visible) if loss_cls else None

        if self.loss_cls is not None and self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.loss_reid is not None and self.loss_reid.use_sigmoid:
            self.reid_out_channels = 1
        else:
            self.reid_out_channels = 2

        if self.loss_idx is not None and self.loss_idx.use_sigmoid:
            self.idx_out_channels = num_input
        else:
            self.idx_out_channels = num_input + 1

        if self.loss_visible is not None and self.loss_visible.use_sigmoid:
            self.visible_out_channels = 1
        else:
            self.visible_out_channels = 2

        self.grid_offset = grid_offset
        self.input_ray_encoding = build_positional_encoding(input_ray_encoding) if input_ray_encoding else None
        self.input_pts_encoding = build_positional_encoding(input_pts_encoding) if input_pts_encoding else None
        #self.output_det_encoding = build_positional_encoding(output_det_encoding) if output_det_encoding else None
        self.output_det_encoding = None
        self.output_det_2d_encoding = build_positional_encoding(output_det_encoding) if output_det_encoding else None
        self.output_seg_encoding = build_positional_encoding(output_seg_encoding) if output_seg_encoding else None

        self.det_feat_idx = det_feat_idx

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder) if bbox_coder else None
        self.pc_range = position_range

        if rpn_idx_learnable :
            self.idx_emb = nn.Parameter(torch.rand((num_input, 1, idx_emb_size))) #(300, 1, 127)
            self.register_parameter('idx_emb', self.idx_emb)
        else :
            idx_emb = torch.arange(0, 1.0, 1/num_input).repeat(idx_emb_size, 1, 1).transpose(2,0) #(300, 1, 127)
            self.idx_emb = nn.Parameter(torch.tensor(idx_emb, requires_grad=False), requires_grad=False)

        self.embed_dims = 256
        self.det_transformer = build_transformer(det_transformer) if det_transformer is not None else None
        self.with_time = with_time

        if self.det_transformer is not None:
            assert self.loss_bbox is not None
            query_points = nn.Parameter(torch.rand((num_query, 3)))
            self.register_parameter('query_points', query_points)

            num_layers = len(self.det_transformer.det_decoders.layers)

            # classification branch
            cls_branch = []
            for lyr, (in_channel, out_channel) in enumerate(
                    zip([self.output_det_2d_encoding.embed_dim] + cls_hidden_dims,
                        cls_hidden_dims + [self.cls_out_channels])):
                cls_branch.append(nn.Linear(in_channel, out_channel))
                if lyr < len(cls_hidden_dims):
                    cls_branch.append(nn.LayerNorm(out_channel))
                    cls_branch.append(nn.ReLU(inplace=True))
            cls_branch = nn.Sequential(*cls_branch)
            if shared_head:
                self.cls_branch = nn.ModuleList([cls_branch for _ in range(num_layers)])
            else:
                self.cls_branch = nn.ModuleList([deepcopy(cls_branch) for _ in range(num_layers)])

            '''
            # regression branch
            reg_branch = []
            for lyr, (in_channel, out_channel) in enumerate(
                    zip([self.output_det_encoding.embed_dim] + reg_hidden_dims, reg_hidden_dims + [self.reg_channels])):
                reg_branch.append(nn.Linear(in_channel, out_channel))
                if lyr < len(reg_hidden_dims):
                    reg_branch.append(nn.ReLU())
            reg_branch = nn.Sequential(*reg_branch)
            if shared_head:
                self.reg_branch = nn.ModuleList([reg_branch for _ in range(num_layers)])
            else:
                self.reg_branch = nn.ModuleList([deepcopy(reg_branch) for _ in range(num_layers)])
            '''
            self.reg_branch = None
        else:
            self.query_points = None
            self.cls_branch = None
            self.reg_branch = None

        if self.loss_seg is not None:
            # TODO: add semantic branch
            pass

        self.loss_visible = build_loss(loss_visible) if loss_visible else None
        self.pred_size = pred_size
        self.emb_intrinsics = emb_intrinsics

        reid_branch = []
        for lyr, (in_channel, out_channel) in enumerate(
                zip([self.output_det_2d_encoding.embed_dim] + reid_hidden_dims,
                    reid_hidden_dims + [self.reid_out_channels])):
            reid_branch.append(nn.Linear(in_channel, out_channel))
            if lyr < len(reid_hidden_dims):
                reid_branch.append(nn.LayerNorm(out_channel))
                reid_branch.append(nn.ReLU(inplace=True))
        reid_branch = nn.Sequential(*reid_branch)
        if shared_head:
            self.reid_branch = nn.ModuleList([reid_branch for _ in range(num_layers)])
        else:
            self.reid_branch = nn.ModuleList([deepcopy(reid_branch) for _ in range(num_layers)])

        idx_branch = []
        for lyr, (in_channel, out_channel) in enumerate(
                zip([self.output_det_2d_encoding.embed_dim] + idx_hidden_dims,
                    idx_hidden_dims + [self.idx_out_channels])):
            idx_branch.append(nn.Linear(in_channel, out_channel))
            if lyr < len(idx_hidden_dims):
                idx_branch.append(nn.LayerNorm(out_channel))
                idx_branch.append(nn.ReLU(inplace=True))
        idx_branch = nn.Sequential(*idx_branch)
        if shared_head:
            self.idx_branch = nn.ModuleList([idx_branch for _ in range(num_layers)])
        else:
            self.idx_branch = nn.ModuleList([deepcopy(idx_branch) for _ in range(num_layers)])

        visible_branch = []
        for lyr, (in_channel, out_channel) in enumerate(
                zip([self.output_det_2d_encoding.embed_dim] + visible_hidden_dims,
                    visible_hidden_dims + [self.visible_out_channels])):
            visible_branch.append(nn.Linear(in_channel, out_channel))
            if lyr < len(visible_hidden_dims):
                visible_branch.append(nn.LayerNorm(out_channel))
                visible_branch.append(nn.ReLU(inplace=True))
        visible_branch = nn.Sequential(*visible_branch)
        if shared_head:
            self.visible_branch = nn.ModuleList([visible_branch for _ in range(num_layers)])
        else:
            self.visible_branch = nn.ModuleList([deepcopy(visible_branch) for _ in range(num_layers)])

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        super(VEDetHead, self).init_weights()
        if self.loss_reid is not None and self.loss_reid.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for reid_branch in self.reid_branch:
                nn.init.constant_(reid_branch[-1].bias, bias_init)

        if self.loss_idx is not None and self.loss_idx.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for idx_branch in self.idx_branch:
                nn.init.constant_(idx_branch[-1].bias, bias_init)

        if self.loss_visible is not None and self.loss_visible.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for visible_branch in self.visible_branch:
                nn.init.constant_(visible_branch[-1].bias, bias_init)

    def forward(self, inputs, img_metas, is_test=False):
        """Forward function.
        Args:
            inputs (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        #print(img_metas[0]['filename'])
        self.img_metas = img_metas
        pred_box, pred_feats, pred_prob = inputs #(1, 300, 3, 4), (1, 300, 3, 128), (1, 300, 3), 
        self.pred_box, self.pred_feats, self.pred_prob = pred_box, pred_feats, pred_prob

        pred_box = pred_box.transpose(2, 1).unsqueeze(-2) #(1, 3, 300, 1, 4),
        pred_feats = pred_feats.transpose(2, 1).unsqueeze(-2) # (1, 3, 300, 1, 128),
        pred_prob = pred_prob.transpose(2, 1).unsqueeze(-1).unsqueeze(-1) # (1, 3, 300, 1, 1), 
        batch_size, num_cams, feat_h, feat_w, _ = pred_feats.shape #1, 3, 300, 1
        masks = pred_feats.new_zeros((batch_size, num_cams, feat_h, feat_w)).to(torch.bool) #(1, 3, 300, 1)
        if self.rpn_mask and not is_test:
            masks[:] = True
            target_idx = pred_box.new_tensor(self.img_metas[0]['pred_box_idx']).to(torch.long) #(num_target, 3)
            num_target = len(target_idx) #num_target
            cam_idx = torch.arange(self.num_decode_views).repeat(num_target, 1).flatten(0,1) #(3, )->(num_target, 3)->(num_target*3, )
            target_idx = target_idx.flatten(0,1) #(num_target*3, )
            masks[:, cam_idx[target_idx>-1], target_idx[target_idx>-1]] = False

        if self.input_ray_encoding is not None:
            pos_embeds = self.position_embedding(pred_box, img_metas) #(1, 3, 300, 1, 256)
        rpn_idx_emb = self.idx_emb.repeat(batch_size, num_cams, 1, 1, 1) #(1, 3, 300, 1, 3)
        #pred_prob[:] = 1.0
        feats = torch.cat([pred_feats, pred_prob, rpn_idx_emb], dim=-1)  #(1, 3, 300, 1, 128+1+127)
        feats = feats.flatten(2, 3) #(1, 3, 300*1, 256)

        # detection & segmentation
        cls_scores, bbox_preds, seg_scores = None, None, None
        if self.det_transformer is not None:
            init_det_points = self.query_points.repeat(batch_size, 1, 1, 1) if self.query_points is not None else None #(1, 1, 900, 3) #xyz in world_coord

            # transform query points to local viewpoints
            init_det_points_mtv, query3d_denorm = self.get_mtv_points_local(init_det_points, img_metas) #(1, 3, 900, 3) xyz in cam_coord #(1, 1, 900, 3)
            init_det_points_mtv, query2d_denorm = self.get_mtv_points_img(init_det_points_mtv, img_metas) #(1, 3, 900, 2) xy in img  #(1, 3, 900, 2)

            self.query3d_norm, self.query3d_denorm = init_det_points, query3d_denorm #(1, 1, 900, 3), #(1, 1, 900, 3)
            self.query2d_norm, self.query2d_denorm = init_det_points_mtv, query2d_denorm

            init_det_points, init_det_points_mtv = self.add_pose_info(init_det_points, init_det_points_mtv, img_metas) #(1, 2, 900, 13) 

            # TODO: seg points
            init_seg_points = None

            # transformer decode
            num_decode_views = init_det_points_mtv.shape[1] if init_det_points_mtv is not None else 0 #3
            det_outputs, regs, seg_outputs = self.det_transformer(feats, masks, pos_embeds, init_det_points,
                                                                  init_det_points_mtv, init_seg_points,
                                                                  #self.output_det_encoding, self.output_seg_encoding,
                                                                  [self.output_det_encoding, self.output_det_2d_encoding], self.output_seg_encoding,
                                                                  self.reg_branch, num_decode_views) #(6, 1, 3, 900, 256), [], []

            # detection from queries
            #if len(det_outputs) > 0 and len(regs) > 0:
            if len(det_outputs) > 0 :
                cls_scores = torch.stack(
                    [cls_branch(output) for cls_branch, output in zip(self.cls_branch, det_outputs)], dim=0) #(6, 1, 3, 900, 120)
                if cls_scores.dim() == 5:
                    #visible_scores = cls_scores[:, :, 1:, :, 0].transpose(2, 3) #(6, 1, 900, 3=num_views)
                    cls_scores = cls_scores[:, :, 0] #(6, 1, 900, 120)

                visible_scores = torch.stack(
                    [visible_branch(output) for visible_branch, output in zip(self.visible_branch, det_outputs)], dim=0) #(6, 1, 3, 900, 1)
                visible_scores = visible_scores[..., 0].transpose(2, 3) #(6, 1, 900, 3)

                reid_scores = torch.stack(
                    [reid_branch(output) for reid_branch, output in zip(self.reid_branch, det_outputs)], dim=0) #(6, 1, 3, 900, 1)
                reid_scores = reid_scores[:, :, 0, :, 0] #(6, 1, 900)

                idx_scores = torch.stack(
                    [idx_branch(output) for idx_branch, output in zip(self.idx_branch, det_outputs)], dim=0) #(6, 1, 3, 900, 300)
                idx_scores = idx_scores.transpose(2, 3) #(6, 1, 900, 3, 300)

                '''
                #visible_scores = torch.stack(
                #    [visible_branch(output) for visible_branch, output in zip(self.visible_branch, det_outputs)], dim=0) #(6, 1, 3, 900, 1)

                bbox_preds = torch.stack(regs, dim=0) if isinstance(regs, list) else regs
                #bbox_preds[..., 0:10] = 0
                if is_test :
                    bbox_preds[..., 0::10] = (
                        #bbox_preds[..., 0::10] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                        bbox_preds[..., 0::10] * input_img_w)
                    bbox_preds[..., 1::10] = (
                        #bbox_preds[..., 1::10] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                        bbox_preds[..., 1::10] * input_img_h)
                    #bbox_preds[..., 4::10] = (
                    #    bbox_preds[..., 4::10] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

                if self.with_time:
                    time_stamps = []
                    for img_meta in img_metas:
                        time_stamps.append(np.asarray(img_meta['timestamp']))
                    time_stamp = bbox_preds.new_tensor(time_stamps)
                    time_stamp = time_stamp.view(batch_size, -1, 6)
                    mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)
                    bbox_preds[..., 8::10] = bbox_preds[..., 8::10] / mean_time_stamp
                    bbox_preds[..., 9::10] = bbox_preds[..., 9::10] / mean_time_stamp
                '''

            # segmentation
            if len(seg_outputs) > 0:
                seg_scores = torch.stack([self.seg_branch(output) for output in seg_outputs], dim=0)

        outs = {
            'all_cls_scores': cls_scores, #(6, 1, 900, 120)
            'all_visible_scores': visible_scores, #(6, 1, 900, 3)
            'all_reid_scores': reid_scores, #(6, 1, 900)
            'all_idx_scores': idx_scores, #(6, 1, 900, 3, 300)
            'all_bbox_preds': bbox_preds, #None
            'all_seg_preds': seg_scores, #None
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(self, cls_score, visible_score, reid_score, idx_score, bbox_pred, gt_labels, gt_visibles, gt_idx, gt_proj_cxcy, gt_bboxes, gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = cls_score.size(0)
        # assigner and sampler
        #assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore,
        pred_proj_cxcy = self.query2d_norm[0].transpose(1,0).flatten(1,2) #(900, 3*2)
        assign_result = self.assigner.assign(pred_proj_cxcy, cls_score, visible_score, reid_score, idx_score, gt_proj_cxcy, gt_labels, gt_visibles, gt_idx, gt_bboxes_ignore, img_shape = self.img_metas[0]['pad_shape'][:2])
        sampling_result = self.sampler.sample(assign_result, pred_proj_cxcy, gt_proj_cxcy)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long) #(900, )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds] 
        label_weights = gt_bboxes.new_ones(num_bboxes) #(900, )
        if self.tp_train_only : 
            label_weights[:] = 0.
            label_weights[pos_inds] = 1.0

        # visible targets
        visible_targets = torch.ones_like(visible_score, dtype=torch.long) #(900, 3=num_views) #0=fg, 1=bg
        visible_targets[pos_inds] = gt_visibles[sampling_result.pos_assigned_gt_inds]
        visible_weights = torch.zeros_like(visible_score) #(900, 3=num_views)
        visible_weights[pos_inds] = 1.0

        # reid targets
        reid_targets = torch.ones_like(reid_score, dtype=torch.long) #(900, ) #0=fg, 1=bg
        reid_targets[pos_inds] = 0
        reid_weights = reid_score.new_ones(num_bboxes) #(900, )
        if self.tp_train_only : 
            reid_weights[:] = 0.
            reid_weights[pos_inds] = 1.0

        # idx targets
        idx_targets = gt_idx.new_full(visible_score.shape, self.num_input, dtype=torch.long) #(900, 3) = 300
        idx_targets[pos_inds] = gt_idx[sampling_result.pos_assigned_gt_inds]
        idx_weights = visible_weights #(900, 3)

        # bbox targets for debugging
        bbox_targets = gt_idx.new_zeros((num_bboxes, self.num_decode_views, 4), dtype=torch.float32) #(1, 3, 300, 1)
        bbox_targets[pos_inds] = get_box_form_pred_idx(self.pred_box[0], gt_idx[sampling_result.pos_assigned_gt_inds], self.num_decode_views)
        bbox_weights = torch.zeros_like(bbox_targets)

        #return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)
        return (labels, label_weights, visible_targets, visible_weights, reid_targets, reid_weights, idx_targets, idx_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    #def get_targets(self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list=None):
    def get_targets(self, cls_scores_list, visible_scores_list, reid_scores_list, idx_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_visibles_list, gt_idx_list, gt_proj_cxcy_list, gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, visibles_list, visible_weights_list, reid_list, reid_weights_list, idx_list, idx_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list, visible_scores_list, reid_scores_list, idx_scores_list, bbox_preds_list, gt_labels_list, gt_visibles_list, gt_idx_list, gt_proj_cxcy_list, 
                                      gt_bboxes_list, gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, visibles_list, visible_weights_list, reid_list, reid_weights_list, idx_list, idx_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    visible_scores,
                    reid_scores,
                    idx_scores,
                    bbox_preds,
                    seg_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_visibles_list,
                    gt_idx_list,
                    gt_proj_cxcy_list, 
                    gt_seg_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        loss_cls, loss_visible, loss_bbox, loss_seg = None, None, None, None
        if cls_scores is not None:
            num_imgs = cls_scores.size(0) #1
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            visible_scores_list = [visible_scores[i] for i in range(num_imgs)]
            reid_scores_list = [reid_scores[i] for i in range(num_imgs)]
            idx_scores_list = [idx_scores[i] for i in range(num_imgs)]
            #bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
            bbox_preds_list = [None for i in range(num_imgs)]
            targets = self.get_targets(cls_scores_list, visible_scores_list, reid_scores_list, idx_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_visibles_list, gt_idx_list, gt_proj_cxcy_list, gt_bboxes_ignore_list)
            (labels_list, label_weights_list, visibles_list, visible_weights_list, reid_list, reid_weights_list, idx_list, idx_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            visibles = torch.cat(visibles_list, 0)
            visible_weights = torch.cat(visible_weights_list, 0)
            reids = torch.cat(reid_list, 0)
            reid_weights = torch.cat(reid_weights_list, 0)
            idx = torch.cat(idx_list, 0)
            idx_weights = torch.cat(idx_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # classification loss
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels) #(900, 120)
            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

            cls_avg_factor = max(cls_avg_factor, 1)
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

            # visible loss
            visible_scores = visible_scores.reshape(-1, self.num_decode_views) #(900, 3)
            # construct weighted avg_factor to match with the official DETR repo
            num_total_pos_visible = (1-visibles).sum() # 39
            num_total_neg_visible = num_total_pos * self.num_decode_views - num_total_pos_visible # 21*3 - 39
            visible_cls_avg_factor = num_total_pos_visible * 1.0 + \
                num_total_neg_visible * self.bg_cls_weight
            visible_cls_avg_factor = max(visible_cls_avg_factor, 1)
            if self.sync_cls_avg_factor:
                visible_cls_avg_factor = reduce_mean(visible_scores.new_tensor([visible_cls_avg_factor]))

            visible_cls_avg_factor = max(visible_cls_avg_factor, 1)
            loss_visible = self.loss_visible(visible_scores.reshape(-1,1), visibles.reshape(-1,), visible_weights.reshape(-1,), avg_factor=visible_cls_avg_factor)

            # reid loss
            reid_scores = reid_scores.reshape(-1, self.reid_out_channels) #(900, 1)
            # construct weighted avg_factor to match with the official DETR repo
            reid_avg_factor = visible_cls_avg_factor
            loss_reid = self.loss_reid(reid_scores, reids, reid_weights, avg_factor=reid_avg_factor)

            # idx loss
            idx_scores = idx_scores.reshape(-1, self.idx_out_channels) #(900*3, 300)
            idx = idx.reshape(-1,) #(900*3, )
            idx_weights = idx_weights.reshape(-1, ) #(900*3, )
            # construct weighted avg_factor to match with the official DETR repo
            idx_avg_factor = visible_cls_avg_factor
            loss_idx = self.loss_idx(idx_scores, idx, idx_weights, avg_factor=idx_avg_factor)

            # Compute the average number of gt boxes accross all gpus, for
            # normalization purposes
            num_total_pos = loss_cls.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

            '''
            # regression L1 loss
            bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
            normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
            valid_code_idx = torch.where(self.code_weights>0)[0]
            #isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
            isnotnan = torch.isfinite(normalized_bbox_targets[:, valid_code_idx]).all(dim=-1)
            #bbox_weights = bbox_weights * self.code_weights 
            bbox_weights = bbox_weights * self.code_weights 
            normalized_bbox_targets = torch.nan_to_num(normalized_bbox_targets, nan=0, posinf=0, neginf=0)

            #input_img_h, input_img_w, _ = self.img_metas[0]['pad_shape'][0]
            #normalized_bbox_targets[..., 0::10] = normalized_bbox_targets[..., 0::10] / input_img_w
            #normalized_bbox_targets[..., 1::10] = normalized_bbox_targets[..., 1::10] / input_img_h
            '''

            if self.debug :
                #self.debug = 0
                import torch.nn.functional as F
                from ...core.bbox.util import denormalize_bbox
                from ...core.visualization.show_result_mtv2d import  imshow_gt_det_bboxes, show_result_mtv2d
                import os
                import mmcv

                save_dir = '/data3/sap/VEDet/result/tmvreid4'

                data_root = '/'+os.path.join(*self.img_metas[0]['filename'][0].split('/')[:-1])
                filename = self.img_metas[0]['filename'][0].split('/')[-1].split('.')[0].split('-')[:-1]
                scene_id = '-'.join(filename)

                idx_scores = idx_scores.reshape(-1, self.num_decode_views, self.idx_out_channels) #(900, 3, 300)
                pred_idx_scores, pred_idx = F.softmax(idx_scores, dim=-1).max(-1) #(900, 3), (900, 3)
                pred_cls_scores, pred_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)

                target_idx = torch.where(labels!=120)[0] #(7,)

                gt_labels = labels[target_idx]
                gt_bbox = bbox_targets[target_idx].transpose(1, 0).cpu().detach().numpy() #(num_views, num_gt, 4)
                gt_is_valid = (1-visibles)[target_idx].transpose(1, 0).cpu().detach().numpy() #(num_views, num_gt)
                cam_gt_cls = gt_labels.cpu().numpy().astype('str') #(num_gt, ) #str

                def result_from_idx(target_idx) :
                    cam_labels = pred_labels[target_idx]
                    cam_pred_idx = pred_idx[target_idx]
                    cam_is_valids = visible_scores[target_idx]

                    cam_det_bboxes = get_box_form_pred_idx(self.pred_box[0], cam_pred_idx, self.num_decode_views)

                    cam_det_cls = cam_labels.cpu().numpy().astype('str') #(num_pred,) #str
                    cam_det_scores = cls_scores.sigmoid()[target_idx][(torch.arange(len(cam_labels)), cam_labels)] #(num_pred,) #float
                    cam_reid_scores = reid_scores.sigmoid()[target_idx].squeeze()

                    return cam_det_bboxes.transpose(1, 0).cpu().detach().numpy(), cam_is_valids.transpose(1, 0).cpu().detach().numpy(), cam_reid_scores.cpu().detach().numpy(), cam_det_cls, cam_det_scores.cpu().detach().numpy() 

                is_draw_gt_target = True
                #is_draw_gt_target = 0
                if is_draw_gt_target : 
                    draw_idx = target_idx
                    save_name = 'gt_target' 
                    cur_save_dir = os.path.join(save_dir, save_name)
                    cam_det_bboxes, cam_is_valids, cam_reid_scores, cam_det_cls, cam_det_scores = result_from_idx(draw_idx)
                    #cam_is_valids[:] = 1
                    result = (scene_id, gt_bbox, gt_is_valid, cam_gt_cls, cam_det_bboxes, cam_is_valids, cam_det_cls, cam_reid_scores)
                    show_result_mtv2d(data_root, cur_save_dir, result, 0) 

                #is_draw_pred = True
                is_draw_pred = 0
                if is_draw_pred :
                    draw_thresh = .1
                    draw_idx = torch.where(reid_scores.sigmoid()>draw_thresh)[0] #(num_pred, )
                    save_name = 'pred_thresh%.2f'%(draw_thresh) 
                    cur_save_dir = os.path.join(save_dir, save_name)
                    cam_det_bboxes, cam_is_valids, cam_reid_scores, cam_det_cls, cam_det_scores = result_from_idx(draw_idx)
                    result = (scene_id, gt_bbox, gt_is_valid, cam_gt_cls, cam_det_bboxes, cam_is_valids, cam_det_cls, cam_reid_scores)
                    show_result_mtv2d(data_root, cur_save_dir, result, 0) 

                    draw_thresh = .3
                    draw_idx = torch.where(reid_scores.sigmoid()>draw_thresh)[0] #(num_pred, )
                    save_name = 'pred_thresh%.2f'%(draw_thresh) 
                    cur_save_dir = os.path.join(save_dir, save_name)
                    cam_det_bboxes, cam_is_valids, cam_reid_scores, cam_det_cls, cam_det_scores = result_from_idx(draw_idx)
                    result = (scene_id, gt_bbox, gt_is_valid, cam_gt_cls, cam_det_bboxes, cam_is_valids, cam_det_cls, cam_reid_scores)
                    show_result_mtv2d(data_root, cur_save_dir, result, 0) 

                #is_draw_gt = True
                is_draw_gt = 0
                if is_draw_gt : 
                    draw_idx = target_idx
                    save_name = 'gt' 
                    cur_save_dir = os.path.join(save_dir, save_name)
                    cam_det_bboxes, cam_is_valids, cam_reid_scores, cam_det_cls, cam_det_scores = result_from_idx(draw_idx)
                    result = (scene_id, gt_bbox, gt_is_valid, cam_gt_cls, cam_det_bboxes, cam_is_valids, cam_det_cls, cam_reid_scores)
                    show_result_mtv2d(data_root, cur_save_dir, result, 0, show_pred=False, show_gt=True) 


                '''
                img_list = []
                imgs = self.img_metas[0]['img'][0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                is_draw_all_gt_all_target = True
                if is_draw_all_gt_all_target : 
                    for i in range(self.num_decode_views) :
                        img = imgs[i]
                        cam_gt_bboxes = gt_bbox[:, i] #(num_gt, 4) #(cx cy w h)
                        cam_det_bboxes = show_bbox_preds[:, i] #(num_pred, 4) #(cx cy w h)

                        img = imshow_gt_det_bboxes(img, cam_gt_bboxes, cam_gt_cls, cam_det_bboxes, cam_det_cls, cam_det_score, 1)
                        img_list.append(img)
                    img = np.concatenate(img_list, axis=0)

                    result_path = os.path.join(cur_save_dir, '%s.jpg'%(scene_name))
                    mmcv.imwrite(img, result_path)

                    #draw_inst_by_inst
                    for jj in range(len(gt_bbox)) :
                        img_list = []
                        for i in range(self.num_decode_views) :
                            img = imgs[i]
                            cam_gt_bboxes = gt_bbox[jj:jj+1, i] #(num_gt, 4) #(cx cy w h)
                            cam_det_bboxes = show_bbox_preds[jj:jj+1, i] #(num_pred, 4) #(cx cy w h)

                            img = img.cpu().numpy().astype(np.uint8)
                            img = imshow_gt_det_bboxes(img, cam_gt_bboxes, cam_gt_cls, cam_det_bboxes, cam_det_cls, cam_det_score, 1)
                            img_list.append(img)
                        img = np.concatenate(img_list, axis=0)

                        result_path = os.path.join(cur_save_dir, '%s_%02d.jpg'%(scene_name, jj))
                        mmcv.imwrite(img, result_path)

                reid_valid_idx = torch.where(reid_scores>-5)[0] #(num_pred, )

                reid_valid_pred_labels = pred_labels[reid_valid_idx]
                reid_valid_pred_idx = pred_idx[reid_valid_idx]

                reid_valid_show_bbox_preds = get_box_form_pred_idx(self.pred_box[0], reid_valid_pred_idx, self.num_decode_views)

                reid_valid_cam_det_cls = reid_valid_pred_labels.cpu().numpy().astype('str') #(num_pred,) #str
                reid_valid_cam_det_score = cls_scores_sig[reid_valid_idx][(torch.arange(len(reid_valid_pred_labels)), reid_valid_pred_labels)] #(num_pred,) #float

                is_draw_all_pred = False
                if is_draw_all_pred : 
                    all_bbox_preds = denormalize_bbox(bbox_preds, self.pc_range)[:, 9:]
                    for jj in range(len(all_bbox_preds)) :
                        img_list = []
                        imgs = self.img_metas[0]['img'][0].permute(0, 2, 3, 1)
                        for i in range(self.num_decode_views) :
                            img = imgs[i]
                            cam_gt_bboxes = gt_bbox[0:1, (i*9, i*9+1, i*9+3, i*9+5)] #(num_gt, 4) #(cx cy w h)
                            cam_det_bboxes = all_bbox_preds[jj:jj+1, (i*9, i*9+1, i*9+3, i*9+5)] #(num_pred, 4) #(cx cy w h)

                            cam_det_bboxes[:, 0] = cam_det_bboxes[:, 0] * input_img_w
                            cam_det_bboxes[:, 1] = cam_det_bboxes[:, 1] * input_img_h
                            cam_gt_bboxes[:, 0] = cam_gt_bboxes[:, 0] * input_img_w
                            cam_gt_bboxes[:, 1] = cam_gt_bboxes[:, 1] * input_img_h

                            img = img.cpu().numpy().astype(np.uint8)
                            img = imshow_gt_det_bboxes(img, cam_gt_bboxes, cam_gt_cls, cam_det_bboxes, cam_det_cls, cam_det_score, 1)
                            img_list.append(img)
                        img = np.concatenate(img_list, axis=0)

                        result_path = os.path.join(cur_save_dir, '%s_all_pred_%d.jpg'%(scene_name, jj))
                        mmcv.imwrite(img, result_path)


                is_draw_each_inst = False
                if is_draw_each_inst : 
                    pred_cls_scores, pred_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
                    show_bbox_preds = denormalize_bbox(bbox_preds, self.pc_range)[:, 9:]
                    for k in range(len(gt_labels)) :
                        same_cls_idx =  (gt_labels[k] == pred_labels)
                        cur_gt_bbox = gt_bbox[k:k+1]
                        cur_show_bbox_preds = show_bbox_preds[same_cls_idx]

                        cam_det_cls = pred_labels[same_cls_idx].cpu().numpy().astype('str') #(num_pred,) #str
                        cam_det_score = cls_scores_sig[same_cls_idx][:, gt_labels[k]] #(num_pred,) #float

                        cam_gt_cls = gt_labels[k:k+1].cpu().numpy().astype('str') #(num_gt, ) #str

                        img_list = []
                        imgs = self.img_metas[0]['img'][0].permute(0, 2, 3, 1)
                        for i in range(self.num_decode_views) :
                            img = imgs[i]
                            cam_gt_bboxes = cur_gt_bbox[:, (i*9, i*9+1, i*9+3, i*9+5)] #(num_gt, 4) #(cx cy w h)

                            cam_det_bboxes = cur_show_bbox_preds[:, (i*9, i*9+1, i*9+3, i*9+5)] #(num_pred, 4) #(cx cy w h)

                            cam_det_bboxes[:, 0] = cam_det_bboxes[:, 0] * input_img_w
                            cam_det_bboxes[:, 1] = cam_det_bboxes[:, 1] * input_img_h
                            cam_gt_bboxes[:, 0] = cam_gt_bboxes[:, 0] * input_img_w
                            cam_gt_bboxes[:, 1] = cam_gt_bboxes[:, 1] * input_img_h

                            img = img.cpu().numpy().astype(np.uint8)
                            img = imshow_gt_det_bboxes(img, cam_gt_bboxes, cam_gt_cls, cam_det_bboxes, cam_det_cls, cam_det_score, 1)
                            img_list.append(img)
                        img = np.concatenate(img_list, axis=0)

                        result_path = os.path.join(cur_save_dir, '%s_cls_%d.jpg'%(scene_name, gt_labels[k].item()))
                        mmcv.imwrite(img, result_path)

                        for j in range(len(cur_show_bbox_preds)) :
                            img_list = []
                            imgs = self.img_metas[0]['img'][0].permute(0, 2, 3, 1)
                            for i in range(self.num_decode_views) :
                                img = imgs[i]
                                cam_gt_bboxes = cur_gt_bbox[:, (i*9, i*9+1, i*9+3, i*9+5)] #(num_gt, 4) #(cx cy w h)

                                cam_det_bboxes = cur_show_bbox_preds[:, (i*9, i*9+1, i*9+3, i*9+5)] #(num_pred, 4) #(cx cy w h)

                                cam_det_bboxes[:, 0] = cam_det_bboxes[:, 0] * input_img_w
                                cam_det_bboxes[:, 1] = cam_det_bboxes[:, 1] * input_img_h
                                cam_gt_bboxes[:, 0] = cam_gt_bboxes[:, 0] * input_img_w
                                cam_gt_bboxes[:, 1] = cam_gt_bboxes[:, 1] * input_img_h

                                img = img.cpu().numpy().astype(np.uint8)
                                img = imshow_gt_det_bboxes(img, cam_gt_bboxes, cam_gt_cls, cam_det_bboxes[j:j+1], cam_det_cls[j:j+1], cam_det_score[j:j+1], 1)
                                img_list.append(img)
                            img = np.concatenate(img_list, axis=0)

                            result_path = os.path.join(cur_save_dir, '%s_cls_%d_%d.jpg'%(scene_name, gt_labels[k].item(), j))
                            mmcv.imwrite(img, result_path)

                is_calc_diff = False
                if is_calc_diff :
                    show_bbox_preds = bbox_preds[target_idx]
                    show_bbox_preds = denormalize_bbox(show_bbox_preds, self.pc_range)[:, 9:]

                    gt_labels = labels[target_idx]
                    gt_bbox = bbox_targets[target_idx][:, 9:]

                    norm_target = normalized_bbox_targets[target_idx]
                    norm_pred = bbox_preds[target_idx]
                    wgt = bbox_weights[target_idx]

                    diff = norm_target - norm_pred
                    wgted_diff = diff * wgt
                    wgted_diff = wgted_diff[:, 10:]
                    wgted_diff = wgted_diff[:, (0, 1, 2, 5, 10, 11, 12, 15, 20, 21, 22, 25)] #x,y,w,h
                    wgted_diff = torch.reshape(wgted_diff, (-1, 3, 4))

                    diff = diff[:, 10:]
                    diff = diff[:, (0, 1, 2, 5, 10, 11, 12, 15, 20, 21, 22, 25)] #x,y,w,h
                    diff = torch.reshape(diff, (-1, 3, 4))

                    print_label = pred_labels[target_idx]
                
                    target_bbox_coord_idx = []
                    for i in range(1, self.num_decode_views+1) :
                        target_bbox_coord_idx.extend([i*10, i*10+1, i*10+2, i*10+5])

                    print(wgted_diff)
                    print(wgted_diff.abs().mean())

                    #target_idx = torch.unique(torch.where(bbox_targets!=0)[0])
                '''

            '''
            loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan],
                normalized_bbox_targets[isnotnan],
                bbox_weights[isnotnan],
                avg_factor=num_total_pos)
            '''

            loss_cls = torch.nan_to_num(loss_cls)
            loss_visible = torch.nan_to_num(loss_visible)
            loss_reid = torch.nan_to_num(loss_reid)
            loss_idx = torch.nan_to_num(loss_idx)
            loss_bbox = 0
            #loss_bbox = torch.nan_to_num(loss_bbox)
            #print(torch.abs(bbox_preds[isnotnan] - normalized_bbox_targets[isnotnan]))
            #print('')
            #print('loss_cls', loss_cls, 'loss_visible', loss_visible, 'loss_bbox', loss_bbox)
            #print('')

        if seg_preds is not None:
            loss_seg = self.loss_seg(seg_preds, gt_seg_list[0])
            loss_seg = torch.nan_to_num(loss_seg)

        return loss_cls, loss_visible, loss_reid, loss_idx, loss_bbox, loss_seg

    @force_fp32(apply_to=('preds_dicts'))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_segs,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #self.debug=True
        loss_dict = dict()
        if self.loss_cls is not None:
            assert gt_bboxes_ignore is None, \
                f'{self.__class__.__name__} only supports ' \
                f'for gt_bboxes_ignore setting to None.'

            all_cls_scores = preds_dicts['all_cls_scores']
            all_visible_scores = preds_dicts['all_visible_scores']
            all_reid_scores = preds_dicts['all_reid_scores']
            all_idx_scores = preds_dicts['all_idx_scores']
            all_bbox_preds = preds_dicts['all_bbox_preds']
            enc_cls_scores = preds_dicts['enc_cls_scores']
            enc_bbox_preds = preds_dicts['enc_bbox_preds']
            all_seg_preds = preds_dicts['all_seg_preds']

            num_dec_layers = len(all_cls_scores) if all_cls_scores is not None else len(all_seg_preds)
            all_gt_bboxes_list = [None] * num_dec_layers
            all_gt_labels_list = [None] * num_dec_layers
            all_gt_visibles_list = [None] * num_dec_layers
            all_gt_idx_list = [None] * num_dec_layers
            all_gt_bboxes_ignore_list = [None] * num_dec_layers
            all_gt_seg_list = [None] * num_dec_layers

            if all_cls_scores is not None:
                device = gt_labels_list[0].device

                gt_visibles_list = [
                    gt_bboxes.mtv_visibility.to(device) 
                    for gt_bboxes in gt_bboxes_list
                ]

                gt_idx_list = [ gt_bboxes.mtv_targets_idx.to(device) for gt_bboxes in gt_bboxes_list ]
                gt_proj_cxcy_list = [ gt_bboxes.mtv_targets_proj_cxcy.to(device) for gt_bboxes in gt_bboxes_list ]
                gt_bboxes_list = [ gt_bboxes.mtv_targets.to(device) for gt_bboxes in gt_bboxes_list ]

                all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
                all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
                all_gt_visibles_list = [gt_visibles_list for _ in range(num_dec_layers)]
                all_gt_idx_list = [gt_idx_list for _ in range(num_dec_layers)]
                all_gt_proj_cxcy_list = [gt_proj_cxcy_list for _ in range(num_dec_layers)]
                all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

                all_bbox_preds = [None] * num_dec_layers
            else:
                all_cls_scores = [None] * num_dec_layers
                all_visible_scores = [None] * num_dec_layers
                all_bbox_preds = [None] * num_dec_layers

            if all_seg_preds is not None:
                all_gt_seg_list = [[gt_segs] for _ in range(num_dec_layers)]
            else:
                all_seg_preds = [None] * num_dec_layers

            if self.debug : 
                losses_cls, losses_visible, losses_reid, losses_idx, losses_bbox, losses_seg = multi_apply(self.loss_single, all_cls_scores[-1:], all_visible_scores[-1:], all_reid_scores[-1:], all_idx_scores[-1:], all_bbox_preds[-1:], all_seg_preds[-1:], all_gt_bboxes_list[-1:], all_gt_labels_list[-1:], all_gt_visibles_list[-1:], all_gt_idx_list[-1:], all_gt_proj_cxcy_list[-1:], all_gt_seg_list[-1:], all_gt_bboxes_ignore_list[-1:])
            else : 
                losses_cls, losses_visible, losses_reid, losses_idx, losses_bbox, losses_seg = multi_apply(self.loss_single, all_cls_scores, all_visible_scores, all_reid_scores, all_idx_scores, all_bbox_preds, all_seg_preds, all_gt_bboxes_list, all_gt_labels_list, all_gt_visibles_list, all_gt_idx_list, all_gt_proj_cxcy_list, all_gt_seg_list, all_gt_bboxes_ignore_list)

            # loss of proposal generated from encode feature map.
            if enc_cls_scores is not None:
                binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(all_gt_labels_list))]
                enc_loss_cls, enc_losses_bbox = \
                    self.loss_single(enc_cls_scores, enc_bbox_preds,
                                    gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
                loss_dict['enc_loss_cls'] = enc_loss_cls
                loss_dict['enc_loss_bbox'] = enc_losses_bbox

            if losses_cls is not None:
                # loss from the last decoder layer
                loss_dict['loss_cls'] = losses_cls[-1]
                loss_dict['loss_visible'] = losses_visible[-1]
                loss_dict['loss_reid'] = losses_reid[-1]
                loss_dict['loss_idx'] = losses_idx[-1]
                #loss_dict['loss_bbox'] = losses_bbox[-1]

                # loss from other decoder layers
                num_dec_layer = 0
                #for loss_cls_i, loss_visible_i, loss_bbox_i in zip(losses_cls[:-1], losses_visible[:-1], losses_bbox[:-1]):
                for loss_cls_i, loss_visible_i, loss_reid_i, loss_idx_i in zip(losses_cls[:-1], losses_visible[:-1], losses_reid[:-1], losses_idx[:-1]):
                    loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
                    loss_dict[f'd{num_dec_layer}.loss_visible'] = loss_visible_i
                    loss_dict[f'd{num_dec_layer}.loss_reid'] = loss_reid_i
                    loss_dict[f'd{num_dec_layer}.loss_idx'] = loss_idx_i
                    #loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
                    num_dec_layer += 1

            if losses_seg[0] is not None:
                # loss from the last decoder layer
                loss_dict['loss_seg'] = losses_seg[-1]

                # loss from other decoder layers
                num_dec_layer = 0
                for loss_seg_i in losses_seg[:-1]:
                    loss_dict[f'd{num_dec_layer}.loss_seg'] = loss_seg_i
                    num_dec_layer += 1

        return loss_dict

    def get_mtv_points_img(self, init_det_points_mtv, img_metas):
        #if not self.training:
        #    return None

        if len(img_metas[0].get('dec_extrinsics', [])) == 0:
            return None

        intrinsics = init_det_points_mtv.new_tensor([img_meta['intrinsics'] for img_meta in img_metas]) #(1, 9, 4, 4)
        B, V, M = init_det_points_mtv.shape[:3] # = 1, 3, 900 #(1, 3, 900, 3)
        intrinsics = intrinsics[:, :V, :3, :3] #(1, 3, 3, 3)

        # bring back to metric values
        rg = self.pc_range
        divider = torch.tensor([rg[3] - rg[0], rg[4] - rg[1], rg[5] - rg[2]], device=intrinsics.device) #=[2.0, 2.0, .75]
        subtract = torch.tensor([rg[0], rg[1], rg[2]], device=intrinsics.device) #=[-1.0, -1.0, -.25]
        init_det_points_mtv = init_det_points_mtv * divider + subtract

        # (B, N, M, 3)
        #init_det_points_mtv = init_det_points_mtv.repeat(1, N, 1, 1)
        #Rt = extrinsics[:, :, None, :3, :3].transpose(-1, -2).repeat(1, 1, M, 1, 1)
        K = intrinsics[:, :, None, :, :].repeat(1, 1, M, 1, 1) #(1, 3, 900, 3, 3)
        init_det_points_mtv = torch.matmul(K, init_det_points_mtv[..., None]).squeeze(-1)
        init_det_points_mtv = init_det_points_mtv / init_det_points_mtv[:,:,:,-1:] #(1, 3, 900, 3)
        init_det_points_mtv = init_det_points_mtv[..., :-1] #(1, 3, 900, 2)

        # bring back to metric values
        pad_shape = img_metas[0].get('pad_shape', [])[:V] 
        H, W, _ = pad_shape
        img_divider = torch.Tensor([W,H]).to(intrinsics.device) #WH

        # normalize
        init_det_points_mtv_denorm = init_det_points_mtv
        init_det_points_mtv_norm = init_det_points_mtv_denorm / img_divider

        return init_det_points_mtv_norm, init_det_points_mtv_denorm

    def get_mtv_points_local(self, init_det_points, img_metas):
        # Same as vedet_haed.py except for the below two rows.

        #if not self.training:
        #    return None

        if len(img_metas[0].get('dec_extrinsics', [])) == 0:
            return None

        extrinsics = init_det_points.new_tensor([img_meta['dec_extrinsics'] for img_meta in img_metas])
        B, N = extrinsics.shape[:2]
        M = init_det_points.shape[2]

        # bring back to metric values
        rg = self.pc_range
        divider = torch.tensor([rg[3] - rg[0], rg[4] - rg[1], rg[5] - rg[2]], device=extrinsics.device)
        subtract = torch.tensor([rg[0], rg[1], rg[2]], device=extrinsics.device)
        init_det_points = init_det_points * divider + subtract

        #init_det_points[:,:,:7] = init_det_points.new_tensor([[ 0.29542732, -0.03341508, -0.00442334], [ 0.09282182,  0.16310196,  0.08339175], [ 0.20363398,  0.16747573,  0.02623144], [-0.05534008,  0.14034814,  0.01114271], [ 0.13638118, -0.04308339, -0.02122111], [-0.09600962,  0.00557075,  0.04734374], [-0.03392284, -0.11034748,  0.05279527]])

        # (B, N, M, 3)
        init_det_points_mtv = init_det_points.repeat(1, N, 1, 1)
        init_det_points_mtv = init_det_points_mtv - extrinsics[:, :, None, :3, 3]
        Rt = extrinsics[:, :, None, :3, :3].transpose(-1, -2).repeat(1, 1, M, 1, 1)
        init_det_points_mtv = torch.matmul(Rt, init_det_points_mtv[..., None]).squeeze(-1)

        # normalize
        init_det_points_mtv = (init_det_points_mtv - subtract) / (divider + 1e-6)

        return init_det_points_mtv, init_det_points

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        #preds_dicts['all_bbox_preds'] = preds_dicts['all_bbox_preds'][..., :10]
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            #bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            #bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            if rescale :
                code_size = bboxes.shape[-1] // (self.num_decode_views + 1)
                scale_factor = img_metas[0]['scale_factor'][0] 
                bboxes[:, 0::code_size] /= scale_factor[0]
                bboxes[:, 1::code_size] /= scale_factor[1]
                bboxes[:, 3::code_size] /= scale_factor[2]
                bboxes[:, 5::code_size] /= scale_factor[3]

            visibles = preds['visibles']
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, visibles, scores, labels])
        return ret_list

    def add_pose_info(self, init_det_points, init_det_points_mtv, img_metas):
        imgH, imgW, _, _ = img_metas[0]['ori_shape']
        # add identity pose to ego queries and make V copies of queries with viewing poses as mtv queries
        B = init_det_points.shape[0] #1
        identity_quat = init_det_points.new_zeros(B, 1, self.num_query, 4) #(1, 1, 900, 4)
        identity_quat[..., 0] = 1 
        # (B, 1, M, 10)
        #init_det_points = torch.cat([init_det_points, identity_quat, torch.zeros_like(init_det_points)], dim=-1)
        init_det_points = torch.cat([init_det_points, identity_quat, torch.zeros_like(init_det_points), torch.zeros_like(identity_quat)], dim=-1) #(1, 1, 900, 14)

        if init_det_points_mtv is None:
            return init_det_points, None

        dec_extrinsics = init_det_points.new_tensor([img_meta['dec_extrinsics'] for img_meta in img_metas])
        dec_quat = tfms.matrix_to_quaternion(dec_extrinsics[..., :3, :3].transpose(-1, -2))
        dec_tvec = dec_extrinsics[..., :3, 3]
        # (B, V, M, 7)
        dec_pose = torch.cat([dec_quat, dec_tvec], dim=-1).unsqueeze(2).repeat(1, 1, self.num_query, 1) #(1, 3, 900, 7)

        intrinsics = init_det_points.new_tensor([img_meta['intrinsics'] for img_meta in img_metas]) #(1, 9, 4, 4)
        intrinsics = intrinsics[:, :self.num_decode_views, :3, :3] #(1, 2, 3, 3)
        fx = intrinsics[:, :, 0, 0] / imgW #(1, 2, 1)
        fy = intrinsics[:, :, 1, 1] / imgH 
        x0 = intrinsics[:, :, 0, 2] / imgW 
        y0 = intrinsics[:, :, 1, 2] / imgH 
        intrinsics = torch.stack([fx, fy, x0, y0], dim=-1).unsqueeze(2).repeat(1, 1, self.num_query, 1) #(1, 2, 900, 4)

        # (B, V, M, 10)
        #init_det_points_mtv = torch.cat([init_det_points_mtv, dec_pose], dim=-1)
        init_det_points_mtv = torch.cat([init_det_points_mtv, dec_pose, intrinsics], dim=-1) #(1, 2, 900, 13) 

        return init_det_points, init_det_points_mtv

    def generate_rays(self, pred_feats, pred_box, img_metas):
        B, N, _, H, W = pred_feats.shape #(1,3, 300, 1)

        extrinsics = []
        for img_meta in img_metas:
            for i in range(N):
                extrinsics.append(img_meta['extrinsics'][i])
        extrinsics = coords.new_tensor(np.asarray(extrinsics)).view(B, N, 1, 1, 4, 4)
        extrinsics = extrinsics.repeat(1, 1, H, W, 1, 1) #(1, 3, 300, 1, 4, 4)
        #(1, 3, 300, 1, 4)

        return rays, extrinsics

        inv_intrinsics, extrinsics = [], []
        for img_meta in img_metas:
            for i in range(N):
                #inv_intrinsics.append(np.linalg.inv(img_meta['intrinsics'][i]))
                extrinsics.append(img_meta['extrinsics'][i])
        #inv_intrinsics = coords.new_tensor(np.asarray(inv_intrinsics)).view(B, N, 1, 1, 4, 4)
        extrinsics = coords.new_tensor(np.asarray(extrinsics)).view(B, N, 1, 1, 4, 4)

        #inv_intrinsics = inv_intrinsics.repeat(1, 1, H, W, 1, 1)
        extrinsics = extrinsics.repeat(1, 1, H, W, 1, 1)

        #coords3d = torch.matmul(inv_intrinsics, coords)
        #rays = F.normalize(coords3d[..., :3, :], dim=-2)

        #rays = torch.matmul(extrinsics[..., :3, :3], rays).squeeze(-1)

        return rays, extrinsics


    def position_embedding(self, pred_box, img_metas, depth_maps=None, use_cache_depth=False):
        imgH, imgW, _, _ = img_metas[0]['ori_shape']
        B, N, H, W, _ = pred_box.shape #(1, 3, 300, 1)

        extrinsics = []
        for img_meta in img_metas:
            for i in range(N):
                extrinsics.append(img_meta['extrinsics'][i])
        extrinsics = pred_box.new_tensor(np.asarray(extrinsics)).view(B, N, 1, 1, 4, 4)
        extrinsics = extrinsics.repeat(1, 1, H, W, 1, 1) #(1, 3, 300, 1, 4, 4)

        rg = self.pc_range
        divider = torch.tensor([rg[3] - rg[0], rg[4] - rg[1], rg[5] - rg[2]], device=extrinsics.device)
        subtract = torch.tensor([rg[0], rg[1], rg[2]], device=extrinsics.device)

        ctrs = extrinsics[..., :3, 3]
        ctrs = (ctrs - subtract) / divider

        # pytorch3d uses row-major, so transpose the R first
        quats = tfms.matrix_to_quaternion(extrinsics[..., :3, :3].transpose(-1, -2))
        ctrs = torch.cat([ctrs, quats], dim=-1) #(1, 3, 300, 1, 7)

        intrinsics = ctrs.new_tensor([img_meta['intrinsics'] for img_meta in img_metas]) #(1, 9, 4, 4)
        intrinsics = intrinsics[:, :N, :3, :3] #(1, 3, 3, 3)
        fx = intrinsics[:, :, 0, 0] / imgW #(1, 3, 1)
        fy = intrinsics[:, :, 1, 1] / imgH 
        x0 = intrinsics[:, :, 0, 2] / imgW 
        y0 = intrinsics[:, :, 1, 2] / imgH 
        intrinsics = torch.stack([fx, fy, x0, y0], dim=-1).unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W, 1) #(1, 3, 300, 1, 4)
        ctrs = torch.cat([ctrs, intrinsics], dim=-1) #(1, 3, 300, 1, 11)

        pred_box_norm = deepcopy(pred_box)
        pred_box_norm[..., [0,2]] /= imgW # cx w
        pred_box_norm[..., [1,3]] /= imgH # cy h

        geometry = torch.cat([pred_box_norm, ctrs, ], dim=-1) #(1, 3, 300, 1, 4+11)
        camera_embedding = self.input_ray_encoding(geometry) #(1, 3, 300, 1, 256)

        return camera_embedding

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        #preds_dicts['all_bbox_preds'] = preds_dicts['all_bbox_preds'][..., :10]
        preds_dicts = self.bbox_coder.decode(preds_dicts, self.pred_box[0])
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            visibles = preds['visibles']
            reid_scores = preds['reid_scores']
            cls_scores = preds['cls_scores']
            labels = preds['labels']
            ret_list.append([bboxes, visibles, reid_scores, cls_scores, labels])
        return ret_list


