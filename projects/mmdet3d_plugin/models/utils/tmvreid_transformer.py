# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule

from .vedet_transformer import VETransformer

@TRANSFORMER.register_module()
class TMvReidTransformer(VETransformer):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """
    def forward(self,
                x,
                mask,
                x_pos,
                init_det_points,
                init_det_points_mtv,
                init_seg_points,
                pos_encoder,
                pos_seg_encoder,
                reg_branch=None,
                num_decode_views=2,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, hw, c = x.shape #1, 3, 300*1, 256
        x = x.reshape(bs, n * hw, c) #(1, 3*300*1, 256)
        x_pos = x_pos.reshape(bs, n * hw, -1) #(1, 3*300*1, 256)

        mask = mask.view(bs, -1)  # [bs, n, h*w] -> [bs, n*h*w] #(1, 3*300*1)

        # segmentation decoders
        seg_outputs = []
        if self.seg_decoders is not None:
            query_points = init_seg_points.flatten(1, -2)
            # query_embeds = pos_encoder(query_points)
            query_embeds = pos_seg_encoder(query_points)
            query = torch.zeros_like(query_embeds)

            seg_outputs = self.seg_decoders(
                query=query.transpose(0, 1),
                key=x.transpose(0, 1),
                value=x.transpose(0, 1),
                key_pos=None,
                query_pos=query_embeds.transpose(0, 1),
                key_padding_mask=None,
                reg_branch=None)
            seg_outputs = seg_outputs.transpose(1, 2)
            seg_outputs = torch.nan_to_num(seg_outputs)

        # detection decoders
        det_outputs, regs = [], []
        if self.det_decoders is not None:
            memory = x.transpose(0, 1) #(3*300*1, 1, 256)
            attn_masks = [None, None]
            num_query = init_det_points.shape[-2] #900
            total_num = num_query * num_decode_views #900*3 
            self_attn_mask = memory.new_ones((total_num, total_num)) #(2700, 2700)
            for i in range(num_decode_views):
                self_attn_mask[i * num_query:(i + 1) * num_query, i * num_query:(i + 1) * num_query] = 0
            attn_masks[0] = self_attn_mask
            det_outputs, regs = self.decode_bboxes(init_det_points, init_det_points_mtv, memory, x_pos.transpose(0, 1),
                                                   mask, attn_masks, pos_encoder, reg_branch, num_decode_views) #(6, 1, 3, 900, 256), []

        return det_outputs, regs, seg_outputs



    def decode_bboxes(self, init_det_points, init_det_points_mtv, memory, key_pos, mask, attn_masks, pos_encoder,
                      reg_branch, num_decode_views):
        pos3d_encoder, pos2d_encoder = pos_encoder
        B, V, Q, C = init_det_points_mtv.shape #1, 3, 900, 13

        query_points = init_det_points_mtv.flatten(1, 2) #(1, 3*900, 13)
        query_embeds = pos2d_encoder(query_points) #(1, 3*900, 256)

        query = torch.zeros_like(query_embeds) #(1, 3*900, 256)

        regs = []
        # output from layers' won't update next's layer's ref points
        det_outputs = self.det_decoders(
            query=query.transpose(0, 1), #(1, 3*900, 256)
            key=memory, #(3*300*1, 1, 256)
            value=memory, #(3*300*1, 1, 256)
            key_pos=key_pos, #(3*300*1, 1, 256)
            query_pos=query_embeds.transpose(0, 1), #(3*900, 1, 256)
            key_padding_mask=mask, #(1, 3*300*1)
            attn_masks=attn_masks,  #[(2700, 2700), None]
            reg_branch=reg_branch)
        det_outputs = det_outputs.transpose(1, 2) #(6, 1, 2700, 256)
        det_outputs = torch.nan_to_num(det_outputs) #(6, 1, 2700, 256)

        '''
        for reg_brch, output in zip(reg_branch, det_outputs):

            reg = reg_brch(output)
            reference = inverse_sigmoid(query_points[..., :3].clone())
            reg[..., 0:2] += reference[..., 0:2] #cx, cy
            reg[..., 0:2] = reg[..., 0:2].sigmoid()
            reg[..., 4:5] += reference[..., 2:3] #cz
            reg[..., 4:5] = reg[..., 4:5].sigmoid()

            regs.append(reg)
        '''

        L, B, _, C = det_outputs.shape #6, 1, 256
        # (L, B, V, M, C)
        det_outputs = det_outputs.reshape(L, B, num_decode_views, -1, C) #(6, 1, 3, 900, 256)
        # (L, B, V + 1, M, 10)
        #regs = torch.stack(regs).reshape(L, B, num_decode_views + 1, init_det_points.shape[-2], -1)

        # ego decode + mtv center decode, (L, B, M, V * 10)
        #regs = regs.permute(0, 1, 3, 2, 4).flatten(-2)

        return det_outputs, regs
