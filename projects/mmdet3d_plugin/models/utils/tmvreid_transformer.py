# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import warnings
import torch.nn as nn

from mmcv.cnn.bricks.transformer import (BaseTransformerLayer, TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import (build_activation_layer, build_conv_layer, build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning, to_2tuple)
import torch.utils.checkpoint as cp

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
    def __init__(self,
                 det_decoder=None,
                 seg_decoder=None,
                 use_iterative_refinement=False,
                 reduction='ego',
                 init_cfg=None,
                 query_is_3d_emb=False):
        super(TMvReidTransformer, self).__init__(init_cfg=init_cfg)

        self.det_decoders = None
        if det_decoder is not None:
            self.det_decoders = build_transformer_layer_sequence(det_decoder)

        self.seg_decoders = None
        if seg_decoder is not None:
            self.seg_decoders = build_transformer_layer_sequence(seg_decoder)

        assert reduction in {'ego', 'mean'}
        self.reduction = reduction
        self.use_iterative_refinement = use_iterative_refinement

        self.query_is_3d_emb = query_is_3d_emb

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

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
                include_attn_map=False,
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
            num_query = init_det_points_mtv.shape[-2] #900
            total_num = num_query * num_decode_views #900*3 
            self_attn_mask = memory.new_ones((total_num, total_num)) #(2700, 2700)
            for i in range(num_decode_views):
                self_attn_mask[i * num_query:(i + 1) * num_query, i * num_query:(i + 1) * num_query] = 0
            attn_masks[0] = self_attn_mask
            #det_outputs, regs, attn = self.decode_bboxes(init_det_points, init_det_points_mtv, memory, x_pos.transpose(0, 1),
            det_outputs, regs, self_attn_map, cross_attn_map = self.decode_bboxes(init_det_points, init_det_points_mtv, memory, x_pos.transpose(0, 1),
                                                   mask, attn_masks, pos_encoder, reg_branch, num_decode_views, include_attn_map) #(6, 1, 3, 900, 256), []

        return det_outputs, regs, seg_outputs, self_attn_map, cross_attn_map



    def decode_bboxes(self, init_det_points, init_det_points_mtv, memory, key_pos, mask, attn_masks, pos_encoder,
                      reg_branch, num_decode_views, include_attn_map):
        pos3d_encoder, pos2d_encoder = pos_encoder
        B, V, Q, C = init_det_points_mtv.shape #1, 3, 900, 13

        query_points = init_det_points_mtv.flatten(1, 2) #(1, 3*900, 13)
        query_embeds = pos2d_encoder(query_points) #(1, 3*900, 256)

        if self.query_is_3d_emb :
            query = init_det_points.repeat(1, V, 1, 1) #(1, 3, 900, 3)
            query = query.flatten(1, 2) #(1, 3*900, 3)
            query = pos3d_encoder(query) #(1, 3*900, 256)
        else :
            query = torch.zeros_like(query_embeds) #(1, 3*900, 256)

        regs = []
        attn_map = None
        # output from layers' won't update next's layer's ref points
        #det_outputs, attn = self.det_decoders(
        det_outputs = self.det_decoders(
            query=query.transpose(0, 1), #(1, 3*900, 256)
            key=memory, #(3*300*1, 1, 256)
            value=memory, #(3*300*1, 1, 256)
            key_pos=key_pos, #(3*300*1, 1, 256)
            query_pos=query_embeds.transpose(0, 1), #(3*900, 1, 256)
            key_padding_mask=mask, #(1, 3*300*1)
            attn_masks=attn_masks,  #[(2700, 2700), None]
            reg_branch=reg_branch)

        self_attn_map, cross_attn_map = None, None
        if include_attn_map :
            det_outputs, self_attn_map, cross_attn_map = det_outputs

        det_outputs = det_outputs.transpose(1, 2) #(6, 1, 2700, 256)
        det_outputs = torch.nan_to_num(det_outputs) #(6, 1, 2700, 256)
        L, B, _, C = det_outputs.shape #6, 1, 256
        # (L, B, V, M, C)
        det_outputs = det_outputs.reshape(L, B, num_decode_views, -1, C) #(6, 1, 3, 900, 256)
        return det_outputs, regs, self_attn_map, cross_attn_map 

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TMVReidTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 use_intermediate_feat=False,
                 **kwargs):

        super(TMVReidTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.use_intermediate_feat = use_intermediate_feat
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate_query, intermediate_self_attn_map, intermediate_cross_attn_map  = [], [], []
        key = kwargs.pop('key')
        value = kwargs.pop('value')
        for l, layer in enumerate(self.layers):
            if self.use_intermediate_feat:
                query, self_attn_map, cross_attn_map = layer(query, *args, key=key[l], value=value[l], **kwargs)
            else:
                query, self_attn_map, cross_attn_map = layer(query, *args, key=key, value=value, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate_query.append(self.post_norm(query))
                else:
                    intermediate_query.append(query)
                intermediate_self_attn_map.append(self_attn_map)
                intermediate_cross_attn_map.append(cross_attn_map)
        return torch.stack(intermediate_query), torch.stack(intermediate_self_attn_map), torch.stack(intermediate_cross_attn_map)

    def forward_single_layer(self, query, idx, *args, **kwargs):
        key = kwargs.pop('key')
        value = kwargs.pop('value')

        if self.use_intermediate_feat:
            query = self.layers[idx](query, *args, key=key[idx], value=value[idx], **kwargs)
        else:
            query = self.layers[idx](query, *args, key=key, value=value, **kwargs)

        output = query
        if self.post_norm is not None:
            output = self.post_norm(query)

        return query, output

@TRANSFORMER_LAYER.register_module()
class TMVReidTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 with_cp=True,
                 **kwargs):
        super(TMVReidTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.use_checkpoint = with_cp

    def _forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs
    ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query, self_attn_map = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query, cross_attn_map = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query, self_attn_map, cross_attn_map

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward,
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
            )
        else:
            x = self._forward(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask)
        return x


@ATTENTION.register_module()
class TMVReidMultiheadSelfAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(TMVReidMultiheadSelfAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, attn_map = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out)), attn_map

@ATTENTION.register_module()
class TMVReidMultiheadCrossAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 kdim=None,
                 vdim=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(TMVReidMultiheadCrossAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        if kdim is None:
            kdim = embed_dims
        if vdim is None:
            vdim = kdim

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, kdim=kdim, vdim=vdim, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'}, cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, attn_map = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out)), attn_map



