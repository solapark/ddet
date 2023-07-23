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
class TMvdetTransformer(VETransformer):
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

    def decode_bboxes(self, init_det_points, init_det_points_mtv, memory, key_pos, mask, attn_masks, pos3d_encoder, pos2d_encoder,
                      reg_branch, num_decode_views):
        B, V, Q, C = init_det_points_mtv.shape
        if init_det_points_mtv is not None:
            # append queries from virtual views
            #query_points = torch.cat([init_det_points, init_det_points_mtv], dim=1).flatten(1, 2)
            query_points_3d = init_det_points.flatten(1, 2) #(1, 900, 10)
            query_points_2d = init_det_points_mtv.flatten(1, 2) #(1, 900*num_views, 10)
            query_embeds3d = pos3d_encoder(query_points_3d).reshape(B, 1, Q, -1) #(1, 1, 900, 256)
            query_embeds2d = pos2d_encoder(query_points_2d).reshape(B, V, Q, -1)#(1, num_views, 900, 256)

            query_embeds = torch.cat([query_embeds3d, query_embeds2d], dim=1).flatten(1, 2) #(1, num_views*900, 256)
        else:
            query_points = init_det_points.flatten(1, 2)
            query_embeds = pos3d_encoder(query_points)

        #query_embeds3d = pos3d_encoder(query_points_3d)
        #query_embeds2d = pos2d_encoder(query_points_2d)
        query = torch.zeros_like(query_embeds)

        regs = []
        # output from layers' won't update next's layer's ref points
        det_outputs = self.det_decoders(
            query=query.transpose(0, 1),
            key=memory,
            value=memory,
            key_pos=key_pos,
            query_pos=query_embeds.transpose(0, 1),
            key_padding_mask=mask,
            attn_masks=attn_masks,
            reg_branch=reg_branch)
        det_outputs = det_outputs.transpose(1, 2)
        det_outputs = torch.nan_to_num(det_outputs)

        for reg_brch, output in zip(reg_branch, det_outputs):

            reg = reg_brch(output)
            reference = inverse_sigmoid(query_points[..., :3].clone())
            reg[..., 0:2] += reference[..., 0:2] #cx, cy
            reg[..., 0:2] = reg[..., 0:2].sigmoid()
            reg[..., 4:5] += reference[..., 2:3] #cz
            reg[..., 4:5] = reg[..., 4:5].sigmoid()

            regs.append(reg)

        L, B, _, C = det_outputs.shape
        # (L, B, V + 1, M, C)
        det_outputs = det_outputs.reshape(L, B, num_decode_views + 1, -1, C)
        # (L, B, V + 1, M, 10)
        regs = torch.stack(regs).reshape(L, B, num_decode_views + 1, init_det_points.shape[-2], -1)

        # ego decode + mtv center decode, (L, B, M, V * 10)
        regs = regs.permute(0, 1, 3, 2, 4).flatten(-2)

        return det_outputs, regs
