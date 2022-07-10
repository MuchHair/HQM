# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
from models.guass_mask_att import multi_head_attention_forward_gaussian
from models.hard_mask_att import HardMaskMultiheadAttention
from models.hard_mask_att_each import HardMaskMultiheadAttentionEach
from models.hard_mask_att_each2 import HardMaskMultiheadAttentionEach2


class HOITransformer117Q(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, num_verb=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        decoder_norm1 = nn.LayerNorm(d_model)
        self.decoder1 = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm1,
                                           return_intermediate=return_intermediate_dec)

        self.hoi_cls = nn.Linear(d_model, num_verb)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, query_embed_verb, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        ################################################
        query_embed_verb = query_embed_verb.unsqueeze(1).repeat(1, bs, 1)
        tgt_verb = torch.zeros_like(query_embed_verb)
        hs_verb = self.decoder1(tgt_verb, memory, memory_key_padding_mask=mask,
                                pos=pos_embed, query_pos=query_embed_verb)
        ################################################
        pred_verb_cls = self.hoi_cls(hs_verb.transpose(1, 2))
        num_verb_classes = pred_verb_cls.size()[-1]

        mask_verb = torch.eye(num_verb_classes).cuda()
        mask_verb = mask_verb.expand((pred_verb_cls.shape[0],
                                      pred_verb_cls.shape[1], num_verb_classes, num_verb_classes))
        pred_verb_cls = pred_verb_cls.sigmoid()
        pred_verb_cls = pred_verb_cls * mask_verb
        pred_verb_cls = torch.sum(pred_verb_cls, -1)
        ################################################
        verb_weight = pred_verb_cls[-1].transpose(0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        query_embed_verb1 = query_embed_verb * verb_weight[:, :, None]
        query_embed_verb1 = torch.sum(query_embed_verb1, dim=0, keepdim=False)
        # todo
        query_embed = query_embed + query_embed_verb1[None, :, :]
        ################################################
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), pred_verb_cls


class HOITransformer2Q(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm1 = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm1,
                                          return_intermediate=return_intermediate_dec)
        ##################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer1 = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                 dropout, activation, normalize_before)
        self.decoder1 = TransformerDecoder(decoder_layer1, num_decoder_layers, decoder_norm,
                                           return_intermediate=return_intermediate_dec)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        hs1 = self.decoder(tgt1, memory, memory_key_padding_mask=mask,
                           pos=pos_embed, query_pos=query_embed1)
        #####################################################################
        if query_embed2 is not None:
            tgt2 = torch.zeros_like(query_embed2)
            hs2 = self.decoder1(tgt2, memory, tgt_key_padding_mask=query_embed2_mask,
                                memory_key_padding_mask=mask,
                                pos=pos_embed, query_pos=query_embed2)
            return hs1.transpose(1, 2), hs2.transpose(1, 2), \
                   memory.permute(1, 2, 0).view(bs, c, h, w)
        #####################################################################
        return hs1.transpose(1, 2), None, \
               memory.permute(1, 2, 0).view(bs, c, h, w)


class HOITransformer2QAllShare(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        hs1 = self.decoder(tgt1, memory, memory_key_padding_mask=mask,
                           pos=pos_embed, query_pos=query_embed1)
        #####################################################################
        if query_embed2 is not None:
            tgt2 = torch.zeros_like(query_embed2)
            hs2 = self.decoder(tgt2, memory, tgt_key_padding_mask=query_embed2_mask,
                               memory_key_padding_mask=mask,
                               pos=pos_embed, query_pos=query_embed2)
            return hs1.transpose(1, 2), hs2.transpose(1, 2), \
                   memory.permute(1, 2, 0).view(bs, c, h, w)
        #####################################################################
        return hs1.transpose(1, 2), None, \
               memory.permute(1, 2, 0).view(bs, c, h, w)


class HOITransformerTS(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTS(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec,
                                            begin_l=begin_l)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        hs1 = self.decoder(tgt1, memory, memory_key_padding_mask=mask,
                           pos=pos_embed, query_pos=query_embed1)
        #####################################################################
        if query_embed2 is not None:
            tgt2 = torch.zeros_like(query_embed2)
            hs2 = self.decoder.forwardt(tgt2, memory, tgt_key_padding_mask=query_embed2_mask,
                                        memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=query_embed2)
            return hs1.transpose(1, 2), hs2.transpose(1, 2), \
                   memory.permute(1, 2, 0).view(bs, c, h, w)
        #####################################################################
        return hs1.transpose(1, 2), None, \
               memory.permute(1, 2, 0).view(bs, c, h, w)


class HOITransformerTSOffset(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffset(decoder_layer, num_decoder_layers, decoder_norm,
                                                  return_intermediate=return_intermediate_dec,
                                                  begin_l=begin_l, matcher=matcher,
                                                  num_obj_classes=num_obj_classes, num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPosObj(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosObj(decoder_layer, num_decoder_layers, decoder_norm,
                                                   return_intermediate=return_intermediate_dec,
                                                   begin_l=begin_l, matcher=matcher,
                                                   num_obj_classes=num_obj_classes, num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            tgt2 = torch.zeros([query_embed2.size()[0], query_embed2.size()[1], 256]).type_as(query_embed2)
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, tgt2, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffsetTCDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetTCDN(decoder_layer, num_decoder_layers, decoder_norm,
                                                      return_intermediate=return_intermediate_dec,
                                                      begin_l=begin_l, matcher=matcher,
                                                      num_obj_classes=num_obj_classes,
                                                      num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2_verb=None, query_embed2_obj=None,
                query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2_verb is not None and query_embed2_obj is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t, hs_t_verb = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t_verb=query_embed2_verb,
                                      query_pos_t_obj=query_embed2_obj,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2), hs_t_verb.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJ(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobj(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec,
                                                    begin_l=begin_l,
                                                    num_obj_classes=num_obj_classes,
                                                    num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJ1(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_layer1 = TransformerDecoderLayer1(d_model, nhead, dim_feedforward,
                                                  dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobj1(decoder_layer1, decoder_layer, num_decoder_layers, decoder_norm,
                                                     return_intermediate=return_intermediate_dec,
                                                     begin_l=begin_l,
                                                     num_obj_classes=num_obj_classes,
                                                     num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)



class TransformerDecoderTSQPosEobjAttOccShiftCheckShift(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 occ_embed_q: Optional[Tensor] = None,
                 occ_embed_e: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_occ, obj_boxes_occ, verb_class_occ, obj_class_occ = [], [], [], []
        hs, hs_t, hs_occ = [], [], []
        atts_all, attt_all, attocc_all = [], [], []
        att_select_all = []

        # shift
        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask_t,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        #learnable

        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                att_select_all.append(att_select)
        # occ
        '''output_occ = occ_embed_e
        for layer in self.layers:
            output_occ, att_occ, _ = layer(output_occ, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=occ_embed_q,
                                       attn_select=att_select_all[i], use_hard_mask=True)

            if self.return_intermediate:
                output_n = self.norm(output_occ)
                hs_occ.append(output_n)
                attocc_all.append(att_occ)
                sub_boxes_occ.append(self.sub_bbox_embed(output_occ).sigmoid())
                obj_boxes_occ.append(self.obj_bbox_embed(output_occ).sigmoid())
                verb_class_occ.append(self.verb_class_embed(output_occ))
                obj_class_occ.append(self.obj_class_embed(output_occ))'''
        # torch.stack(sub_boxes_occ), torch.stack(obj_boxes_occ), torch.stack(obj_class_occ), torch.stack(
        #             verb_class_occ),
        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               sub_boxes_occ, obj_boxes_occ, verb_class_occ, obj_class_occ, \
               torch.stack(atts_all), torch.stack(attt_all), attocc_all, \
               torch.stack(hs), torch.stack(hs_t), hs_occ


class TransformerDecoderShiftOccPass(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None):
        # assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        atts_all, attt_all = [], []
        att_select_all = []

        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                att_select_all.append(att_select)

        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask_t,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q,
                                       attn_select=att_select_all[i], use_hard_mask=True)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(atts_all), torch.stack(attt_all), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSQPosEobjAttOccShiftPass(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 occ_embed_q: Optional[Tensor] = None,
                 occ_embed_e: Optional[Tensor] = None,
                 shift_pass=True, occ_pass=False):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_occ, obj_boxes_occ, verb_class_occ, obj_class_occ = [], [], [], []
        hs, hs_t, hs_occ = [], [], []
        atts_all, attt_all, attocc_all = [], [], []
        att_select_all = []

        # learnable

        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                att_select_all.append(att_select)

        # shift
        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask_t,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q, use_hard_mask=occ_pass)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        # occ
        if occ_pass:
            output_occ = occ_embed_e
            for i in range(len(self.layers)):
                layer = self.layers[i]
                output_occ, att_occ, _ = layer(output_occ, memory, tgt_mask=tgt_mask_t,
                                               memory_mask=memory_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask,
                                               memory_key_padding_mask=memory_key_padding_mask,
                                               pos=pos, query_pos=occ_embed_q,
                                               attn_select=att_select_all[i], use_hard_mask=True)
                if self.return_intermediate:
                    output_n = self.norm(output_occ)
                    hs_occ.append(output_n)
                    attocc_all.append(att_occ)
                    sub_boxes_occ.append(self.sub_bbox_embed(output_occ).sigmoid())
                    obj_boxes_occ.append(self.obj_bbox_embed(output_occ).sigmoid())
                    verb_class_occ.append(self.verb_class_embed(output_occ))
                    obj_class_occ.append(self.obj_class_embed(output_occ))
        else:

            hs_occ = hs
            attocc_all = atts_all
            sub_boxes_occ = sub_boxes
            obj_boxes_occ = obj_boxes
            verb_class_occ = verb_class
            obj_class_occ = obj_class

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(sub_boxes_occ), torch.stack(obj_boxes_occ), torch.stack(obj_class_occ), torch.stack(verb_class_occ), \
               torch.stack(atts_all), torch.stack(attt_all), torch.stack(attocc_all), \
               torch.stack(hs), torch.stack(hs_t), torch.stack(hs_occ)



class TransformerDecoderTSQPosEobjAttOccShift(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 occ_embed_q: Optional[Tensor] = None,
                 occ_embed_e: Optional[Tensor] = None,
                 shift_pass=True, occ_pass=False):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_occ, obj_boxes_occ, verb_class_occ, obj_class_occ = [], [], [], []
        hs, hs_t, hs_occ = [], [], []
        atts_all, attt_all, attocc_all = [], [], []
        att_select_all = []

        # learnable

        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                att_select_all.append(att_select)

        # shift
        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask_t,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        # occ
        output_occ = occ_embed_e
        for i in range(len(self.layers)):
            layer = self.layers[i]
            output_occ, att_occ, _ = layer(output_occ, memory, tgt_mask=tgt_mask_t,
                                           memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=occ_embed_q,
                                           attn_select=att_select_all[i], use_hard_mask=True)
            if self.return_intermediate:
                output_n = self.norm(output_occ)
                hs_occ.append(output_n)
                attocc_all.append(att_occ)
                sub_boxes_occ.append(self.sub_bbox_embed(output_occ).sigmoid())
                obj_boxes_occ.append(self.obj_bbox_embed(output_occ).sigmoid())
                verb_class_occ.append(self.verb_class_embed(output_occ))
                obj_class_occ.append(self.obj_class_embed(output_occ))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(sub_boxes_occ), torch.stack(obj_boxes_occ), torch.stack(obj_class_occ), torch.stack(verb_class_occ), \
               torch.stack(atts_all), torch.stack(attt_all), torch.stack(attocc_all), \
               torch.stack(hs), torch.stack(hs_t), torch.stack(hs_occ)


class TransformerDecoderTSQPosEobjAttOccShiftCheckOcc(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 occ_embed_q: Optional[Tensor] = None,
                 occ_embed_e: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_occ, obj_boxes_occ, verb_class_occ, obj_class_occ = [], [], [], []
        hs, hs_t, hs_occ = [], [], []
        atts_all, attt_all, attocc_all = [], [], []
        att_select_all = []

        # shift
        '''output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask_t,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))'''

        #learnable

        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                att_select_all.append(att_select)
        # occ
        output_occ = occ_embed_e
        for i in range(len(self.layers)):
            layer = self.layers[i]
            output_occ, att_occ, _ = layer(output_occ, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=occ_embed_q,
                                       attn_select=att_select_all[i], use_hard_mask=True)

            if self.return_intermediate:
                output_n = self.norm(output_occ)
                hs_occ.append(output_n)
                attocc_all.append(att_occ)
                sub_boxes_occ.append(self.sub_bbox_embed(output_occ).sigmoid())
                obj_boxes_occ.append(self.obj_bbox_embed(output_occ).sigmoid())
                verb_class_occ.append(self.verb_class_embed(output_occ))
                obj_class_occ.append(self.obj_class_embed(output_occ))
        #
        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_occ), torch.stack(obj_boxes_occ), torch.stack(obj_class_occ), torch.stack(verb_class_occ), \
               torch.stack(sub_boxes_occ), torch.stack(obj_boxes_occ), torch.stack(obj_class_occ), torch.stack(verb_class_occ), \
               torch.stack(atts_all), torch.stack(attocc_all), torch.stack(attocc_all), \
               torch.stack(hs), torch.stack(hs_occ), torch.stack(hs_occ)


class HOITransformerTSQPOSEOBJAttOccCheckOcc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnOccShift(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttOccShiftCheckOcc(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None,
                occ_embed_q=None, occ_embed_e=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)

        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, sub_boxes_occ, obj_boxes_occ, obj_class_occ, verb_class_occ,\
            att_s, att_t, att_occ, hs, hs_t, hs_occ = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      occ_embed_q=occ_embed_q, occ_embed_e=occ_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_occ.transpose(1, 2), obj_boxes_occ.transpose(1, 2), \
                   obj_class_occ.transpose(1, 2), verb_class_occ.transpose(1, 2),  \
                   att_s, att_occ, att_occ, hs.transpose(1, 2), hs_occ.transpose(1, 2),  hs_occ.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJAttOccPass(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnE(d_model, nhead, dim_feedforward,
                                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttHardmAttnE(decoder_layer, num_decoder_layers, decoder_norm,
                                                                 return_intermediate=return_intermediate_dec,
                                                                 begin_l=begin_l,
                                                                 num_obj_classes=num_obj_classes,
                                                                 num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None, occ_pass=False):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_s, att_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e, occ_pass=occ_pass)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


    '''
class HOITransformerTSQPOSEOBJAttOccPass(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnE(d_model, nhead, dim_feedforward,
                                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderShiftOccPass(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None,
                occ_embed_q=None, occ_embed_e=None,
                shift_pass=True, occ_pass=True):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)

        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, sub_boxes_occ, obj_boxes_occ, obj_class_occ, verb_class_occ,\
            att_s, att_t, att_occ, hs, hs_t, hs_occ = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      occ_embed_q=occ_embed_q, occ_embed_e=occ_embed_e,
                                      shift_pass=shift_pass, occ_pass=occ_pass)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_occ.transpose(1, 2), obj_boxes_occ.transpose(1, 2), \
                   obj_class_occ.transpose(1, 2), verb_class_occ.transpose(1, 2),   \
                   att_s, att_t, att_occ, hs.transpose(1, 2), hs_t.transpose(1, 2),  hs_occ.transpose(1, 2)

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)
    '''

class HOITransformerTSQPOSEOBJAttOcc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnOccShift(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttOccShift(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None,
                occ_embed_q=None, occ_embed_e=None,
                shift_pass=True, occ_pass=True):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)

        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, sub_boxes_occ, obj_boxes_occ, obj_class_occ, verb_class_occ,\
            att_s, att_t, att_occ, hs, hs_t, hs_occ = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      occ_embed_q=occ_embed_q, occ_embed_e=occ_embed_e,
                                      shift_pass=shift_pass, occ_pass=occ_pass)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_occ.transpose(1, 2), obj_boxes_occ.transpose(1, 2), \
                   obj_class_occ.transpose(1, 2), verb_class_occ.transpose(1, 2),   \
                   att_s, att_t, att_occ, hs.transpose(1, 2), hs_t.transpose(1, 2),  hs_occ.transpose(1, 2)

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJAttOccCheckShift(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnOccShift(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttOccShiftCheckShift(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None,
                occ_embed_q=None, occ_embed_e=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)

        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, sub_boxes_occ, obj_boxes_occ, obj_class_occ, verb_class_occ,\
            att_s, att_t, att_occ, hs, hs_t, hs_occ = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      occ_embed_q=occ_embed_q, occ_embed_e=occ_embed_e)
            '''sub_boxes_occ.transpose(1, 2), obj_boxes_occ.transpose(1, 2), \
                   obj_class_occ.transpose(1, 2), verb_class_occ.transpose(1, 2), '''
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2),         \
                   att_s, att_t, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),  hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJAtt(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMap(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAtt(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_s, att_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJAttSOcc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapSOcc(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttSOcc(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_s, att_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


'''class HOITransformerTSQPOSEOBJAttOcc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapOcc(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAtt(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_s, att_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)

'''

class Test(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMap(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAtt(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed,
                oracle_query=None, initial_emeddings=None,
                oracle_query_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        sub_boxes, obj_boxes, obj_class, verb_class, hs, att_s = self.decoder(tgt, memory,
                                                                              memory_key_padding_mask=mask,
                                                                              pos=pos_embed,
                                                                              query_pos=query_embed)
        #####################################################################
        if oracle_query is not None:
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_t, hs_t = \
                self.decoder(initial_emeddings, memory,
                             memory_key_padding_mask=mask,
                             tgt_key_padding_mask=oracle_query_mask,
                             pos=pos_embed,
                             query_pos=oracle_query)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),
        else:
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2)


class HOITransformerTSQPOSEOBJVerb(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjVerb(decoder_layer, num_decoder_layers, decoder_norm,
                                                        return_intermediate=return_intermediate_dec,
                                                        begin_l=begin_l,
                                                        num_obj_classes=num_obj_classes,
                                                        num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None,
                query_embed_e_obj=None, query_embed_e_verb=None,
                query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e_obj=query_embed_e_obj,
                                      query_embed_e_verb=query_embed_e_verb,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJWithS(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjWithS(decoder_layer, num_decoder_layers, decoder_norm,
                                                         return_intermediate=return_intermediate_dec,
                                                         begin_l=begin_l, matcher=matcher,
                                                         num_obj_classes=num_obj_classes,
                                                         num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffsetDQ(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetDQ(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec,
                                                    begin_l=begin_l, matcher=matcher,
                                                    num_obj_classes=num_obj_classes, num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs_o, hs_v, hs_to, hs_tv = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs_o.transpose(1, 2), hs_v.transpose(1, 2), hs_to.transpose(1, 2), \
                   hs_tv.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class = self.decoder(tgt1, memory,
                                                                       memory_key_padding_mask=mask,
                                                                       pos=pos_embed,
                                                                       query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2),


class HOITransformerTSOffsetImageLevel(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetImageLevel(decoder_layer, num_decoder_layers, decoder_norm,
                                                            return_intermediate=return_intermediate_dec,
                                                            begin_l=begin_l, matcher=matcher,
                                                            num_obj_classes=num_obj_classes,
                                                            num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                query_embed_image_level=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            query_embed_image_level = query_embed_image_level.unsqueeze(1).repeat(1, bs, 1)
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t, verb_class_i = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_image_level=query_embed_image_level,
                                      query_pos_t=query_embed2,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2), verb_class_i.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffsetQM(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetQM(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec,
                                                    begin_l=begin_l, matcher=matcher,
                                                    num_obj_classes=num_obj_classes, num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffsetSMask(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerMask(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetSMask(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l, matcher=matcher,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target, h=h, w=w)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffsetTMask(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerMask(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetTMask(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l, matcher=matcher,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None, wh=None, inter_point=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            ###################################################
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            grid = torch.stack((grid_x, grid_y), 2).float()
            grid.requires_grad = False
            grid = grid.type_as(src)
            grid = grid.unsqueeze(0).permute(0, 3, 1, 2).flatten(2).permute(2, 0, 1)
            grid = grid.repeat(1, bs, 1)
            ###################################################

            inter_point_ref = (wh - 0) * inter_point / 32
            distance = (inter_point_ref.unsqueeze(1) - grid.unsqueeze(0).unsqueeze(-2)) ** 2
            scale = 1
            smooth = 8
            distance = distance.sum(-1) * scale
            gaussian = -(distance - 0).abs() / smooth
            if len(gaussian) > 0:
                gaussian = torch.min(gaussian, dim=-1)[0]
            else:
                gaussian = torch.empty(gaussian.size()[:-1]).type_as(src)
            ###################################################
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target,
                                      gaussian=gaussian)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffsetTMask1(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerMask(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetTMask1(decoder_layer, num_decoder_layers, decoder_norm,
                                                        return_intermediate=return_intermediate_dec,
                                                        begin_l=begin_l, matcher=matcher,
                                                        num_obj_classes=num_obj_classes,
                                                        num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None, src_mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target,
                                      src_mask=src_mask)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffset2T(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffset2T(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec,
                                                    begin_l=begin_l, matcher=matcher,
                                                    num_obj_classes=num_obj_classes, num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSOffsetPrune(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetPrune(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l, matcher=matcher)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            tgt2 = torch.zeros_like(query_embed2)

            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, hs, hs_t, query_mask = \
                self.decoder.forwardt(tgt1, tgt2, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_pos_t=query_embed2,
                                      target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2), query_mask.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs, query_mask = \
                self.decoder(tgt1, memory,
                             memory_key_padding_mask=mask,
                             pos=pos_embed,
                             query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2), query_mask.transpose(1, 2)


class HOITransformerTSObjVerb(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSObjVerb(decoder_layer, num_decoder_layers, decoder_norm,
                                                   return_intermediate=return_intermediate_dec,
                                                   begin_l=begin_l, matcher=matcher)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed3=None,
                query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, \
            sub_boxes_t1, obj_boxes_t1, obj_class_t1, verb_class_t1, \
            hs, hs1, hs_t, hs_t1 = self.decoder.forwardt(tgt1, memory,
                                                         memory_key_padding_mask=mask,
                                                         tgt_key_padding_mask_t=query_embed2_mask,
                                                         pos=pos_embed,
                                                         query_pos=query_embed1,
                                                         query_pos_t=query_embed2,
                                                         query_pos_t1=query_embed3,
                                                         target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_t1.transpose(1, 2), obj_boxes_t1.transpose(1, 2), \
                   obj_class_t1.transpose(1, 2), verb_class_t1.transpose(1, 2), \
                   hs.transpose(1, 2), hs1.transpose(1, 2), hs_t.transpose(1, 2), hs_t1.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSNotShare(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSNotShare(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        hs1 = self.decoder(tgt1, memory, memory_key_padding_mask=mask,
                           pos=pos_embed, query_pos=query_embed1)
        #####################################################################
        if query_embed2 is not None:
            tgt2 = torch.zeros_like(query_embed2)
            hs2 = self.decoder.forwardt(tgt2, memory, tgt_key_padding_mask=query_embed2_mask,
                                        memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=query_embed2)
            return hs1.transpose(1, 2), hs2.transpose(1, 2), \
                   memory.permute(1, 2, 0).view(bs, c, h, w)
        #####################################################################
        return hs1.transpose(1, 2), None, \
               memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerQMTS(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderQMTS(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2_pos=None, query_embed2_word=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        hs1 = self.decoder(tgt1, memory, memory_key_padding_mask=mask,
                           pos=pos_embed, query_pos=query_embed1)
        #####################################################################
        if query_embed2_pos is not None:
            hs2 = self.decoder.forwardT(memory, tgt_key_padding_mask=query_embed2_mask,
                                        memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=query_embed2_pos, query_word=query_embed2_word)
            return hs1[0].transpose(1, 2), hs1[1].transpose(1, 2), hs1[2].transpose(1, 2), hs1[3].transpose(1, 2), \
                   hs2[0].transpose(1, 2), hs2[1].transpose(1, 2), hs2[2].transpose(1, 2), hs2[3].transpose(1, 2), \
                   hs1[4].transpose(1, 2), hs2[4].transpose(1, 2)
        #####################################################################
        return hs1[0].transpose(1, 2), hs1[1].transpose(1, 2), hs1[2].transpose(1, 2), hs1[3].transpose(1, 2)


class HOITransformert(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_norm = nn.LayerNorm(d_model)
        ###################################################################################
        decoder_layer1 = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                 dropout, activation, normalize_before)
        self.decoder1 = TransformerDecoder(decoder_layer1, num_decoder_layers, decoder_norm,
                                           return_intermediate=return_intermediate_dec)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        tgt2 = torch.zeros_like(query_embed2)
        hs2 = self.decoder1(tgt2, memory, tgt_key_padding_mask=query_embed2_mask,
                            memory_key_padding_mask=mask,
                            pos=pos_embed, query_pos=query_embed2)
        return hs2.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)




class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerOcc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayerOcc(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerMask(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayerMask(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderMask(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord = self.decoder(tgt, memory,
                                                                                                   memory_key_padding_mask=mask,
                                                                                                   pos=pos_embed,
                                                                                                   query_pos=query_embed,
                                                                                                   h=h, w=w)
        return outputs_obj_class.transpose(1, 2), outputs_verb_class.transpose(1, 2), \
               outputs_sub_coord.transpose(1, 2), outputs_obj_coord.transpose(1, 2),


class TransformerROI(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderROI(decoder_layer, num_decoder_layers, decoder_norm,
                                             return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord = self.decoder(tgt, memory,
                                                                                                   memory_key_padding_mask=mask,
                                                                                                   pos=pos_embed,
                                                                                                   query_pos=query_embed,
                                                                                                   h=h, w=w)
        return outputs_obj_class.transpose(1, 2), outputs_verb_class.transpose(1, 2), \
               outputs_sub_coord.transpose(1, 2), outputs_obj_coord.transpose(1, 2),


class TransformerGuide(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderGuide(decoder_layer, num_decoder_layers, decoder_norm,
                                               return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerPosGuide(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderPosGuide(decoder_layer, num_decoder_layers, decoder_norm,
                                                  return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        sub_boxes, obj_boxes, obj_class, verb_class = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                                   pos=pos_embed, query_pos=query_embed)
        return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), obj_class.transpose(1, 2), verb_class.transpose(1,
                                                                                                                     2)


class TransformerCDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm1 = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers // 2, decoder_norm1,
                                          return_intermediate=return_intermediate_dec)
        decoder_norm2 = nn.LayerNorm(d_model)
        self.decoder_verb = TransformerDecoder(decoder_layer, num_decoder_layers // 2, decoder_norm2,
                                               return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs_obj = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)
        ##################################################################
        tgt1 = torch.zeros_like(hs_obj[-1])
        hs_verb = self.decoder_verb(tgt1, memory, memory_key_padding_mask=mask,
                                    pos=pos_embed, query_pos=hs_obj[-1])
        return hs_obj.transpose(1, 2), hs_verb.transpose(1, 2)


class TransformerAddVerbAsQ(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, query_embed_verb, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        num_q = query_embed.size()[0]
        query_embed_verb = query_embed_verb.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)
        query_embed = torch.cat([query_embed, query_embed_verb], dim=0)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs[:, 0:num_q, :, :].transpose(1, 2), hs[:, num_q:, :, :].transpose(1, 2)


class TransformerPruning(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderPruning(decoder_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, obj_class_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, query_mask = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed, obj_class_embed=obj_class_embed,
                                      )
        return hs.transpose(1, 2), query_mask.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerPruningE2E(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderPruningE2E(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, obj_class_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, query_mask = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed, obj_class_embed=obj_class_embed,
                                      )
        return hs.transpose(1, 2), query_mask.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerPruningGumble(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderPruningGumbelSoftmax(decoder_layer, num_decoder_layers, decoder_norm,
                                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, prune_scores = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                        pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), prune_scores.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerGS(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderGS(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed, w=w, h=h)
        return hs[0].transpose(1, 2), hs[1].transpose(1, 2), hs[2].transpose(1, 2), \
               hs[3].transpose(1, 2), hs[4].transpose(1, 2)


class TransformerFFN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFN(decoder_layer, num_decoder_layers, decoder_norm,
                                             return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                                 pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), \
               output_human.transpose(0, 1), output_obj.transpose(0, 1), output_verb.transpose(0, 1)


class TransformerFFN7l(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFN7l(decoder_layer, num_decoder_layers, decoder_norm,
                                               return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


# 2l
class TransformerFFNAll(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFNAll(decoder_layer, num_decoder_layers, decoder_norm,
                                                return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


class TransformerFFNAllRec(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=3):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFNAllRec(decoder_layer, num_decoder_layers, decoder_norm,
                                                   return_intermediate=return_intermediate_dec, begin_l=begin_l)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        outputs_sub_coord, outputs_obj_coord, outputs_obj_class, outputs_verb_class = \
            self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return outputs_sub_coord.transpose(1, 2), outputs_obj_coord.transpose(1, 2), \
               outputs_obj_class.transpose(1, 2), outputs_verb_class.transpose(1, 2)


class TransformerPartSum(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderPartSum(decoder_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb, output_hoi = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                                         pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), \
               output_verb.transpose(1, 2), output_hoi.transpose(1, 2)


class TransformerFFN2lQG(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder2lQG(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


# 2l object word embedding to verb classfier
class TransformerFFNAllOV(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFNAllOV(decoder_layer, num_decoder_layers, decoder_norm,
                                                  return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        sub_boxes, obj_boxes, obj_cates, verbs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                              pos=pos_embed, query_pos=query_embed)
        return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), obj_cates.transpose(1, 2), verbs.transpose(1, 2)


class TransformerFFNAllRR(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFN2lRR(decoder_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


class TransformerFFN2lTs(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=3):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFNAllTS(decoder_layer, num_decoder_layers, decoder_norm,
                                                  return_intermediate=return_intermediate_dec,
                                                  begin_l=begin_l)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        if query_embed2 is not None:
            tgt2 = torch.zeros_like(query_embed2)
            output_human_gt, output_obj_gt, output_verb_gt = \
                self.decoder.forwardt(tgt2, memory,
                                      tgt_key_padding_mask=query_embed2_mask,
                                      memory_key_padding_mask=mask,
                                      pos=pos_embed,
                                      query_pos=query_embed2)
            return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2), \
                   output_human_gt.transpose(1, 2), output_obj_gt.transpose(1, 2), output_verb_gt.transpose(1, 2)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


# 2l 3l
class TransformerFFN2l3Q(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFN2l3q(decoder_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb, output_hoi = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                                         pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1,
                                                                                               2), output_hoi.transpose(
            1, 2)


class TransformerFFNAllEMul(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFNAllEMul(decoder_layer, num_decoder_layers, decoder_norm,
                                                    return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.encoder_apart = nn.Linear(d_model, 3)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        memory_weight = self.encoder_apart(memory).softmax(-1)
        memory1 = memory * memory_weight[:, :, 0:1]
        memory2 = memory * memory_weight[:, :, 1:2]
        memory3 = memory * memory_weight[:, :, 2:3]

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory1, memory2, memory3,
                                                             memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


# 2l + inter
class TransformerFFN2lInter(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        # branch aggregation: instance-aware attention
        interaction_layer = InteractionLayer(d_model, d_model, dropout)
        self.decoder = TransformerDecoder2lInter(decoder_layer, interaction_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=return_intermediate_dec)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


# 2l + v+o-->hoi
class TransformerFFN2LCom(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder2lCom(decoder_layer, num_decoder_layers, decoder_norm,
                                               return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb, output_hoi = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                                         pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1,
                                                                                               2), output_hoi.transpose(
            1, 2)


# 2l = h+o-->hoi
class TransformerFFN2LHOCom(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder2lHOCom(decoder_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb, output_hoi = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                                         pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1,
                                                                                               2), output_hoi.transpose(
            1, 2)


# 1l
class TransformerFFNAll1l(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFNAll1l(decoder_layer, num_decoder_layers, decoder_norm,
                                                  return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


class TransformerFFN2liq(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderFFN2lIq(decoder_layer, num_decoder_layers, decoder_norm,
                                                 return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, image_embed_obj_verb, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        image_embed_obj_verb = image_embed_obj_verb.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt1 = torch.zeros_like(query_embed)
        tgt2 = torch.zeros_like(image_embed_obj_verb)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        output_human, output_obj, output_verb = self.decoder(tgt1, tgt2, memory, memory_key_padding_mask=mask,
                                                             pos=pos_embed, query_pos=query_embed,
                                                             query_obj_verb=image_embed_obj_verb)
        return output_human.transpose(1, 2), output_obj.transpose(1, 2), output_verb.transpose(1, 2)


class TransformerQM(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderQM(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs[0].transpose(1, 2), hs[1].transpose(1, 2), hs[2].transpose(1, 2), hs[3].transpose(1, 2)


class HOITransformer2QAllShareC(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        if query_embed2 is None:
            query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
            tgt1 = torch.zeros_like(query_embed1)
            hs1 = self.decoder(tgt1, memory, memory_key_padding_mask=mask,
                               pos=pos_embed, query_pos=query_embed1)
            return hs1.transpose(1, 2), None, \
                   memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
            query_embed = torch.cat([query_embed1, query_embed2], dim=0)
            tgt = torch.zeros_like(query_embed)
            query_embed2_mask_ = torch.zeros([bs, len(query_embed1)]).type_as(query_embed2_mask)
            query_embed2_mask = torch.cat([query_embed2_mask_, query_embed2_mask], dim=1)
            hs = self.decoder(tgt, memory, tgt_key_padding_mask=query_embed2_mask,
                              memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)
            return hs[:, 0: len(query_embed1), :, :].transpose(1, 2), \
                   hs[:, len(query_embed1):, :, :].transpose(1, 2), \
                   memory.permute(1, 2, 0).view(bs, c, h, w)
        #####################################################################


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderMask(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        hidden_dim = 256
        num_obj_classes = 80
        num_verb_classes = 117
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                w=None, h=None):
        output = tgt

        intermediate = []
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []

        for layer in self.layers:
            if len(sub_boxes) == 0:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            else:
                sub_box, obj_box = sub_boxes[-1].clone(), obj_boxes[-1].clone()
                num_q, bs, _ = sub_box.size()
                src_mask = np.zeros((bs, num_q, h, w), dtype=np.float32)
                for i in range(bs):
                    sub_box_batch = sub_box[:, i, :]
                    obj_box_batch = obj_box[:, i, :]

                    sub_box_batch[:, 0] = sub_box_batch[:, 0] * w
                    sub_box_batch[:, 1] = sub_box_batch[:, 1] * h
                    sub_box_batch[:, 2] = sub_box_batch[:, 2] * w
                    sub_box_batch[:, 3] = sub_box_batch[:, 3] * h

                    obj_box_batch[:, 0] = obj_box_batch[:, 0] * w
                    obj_box_batch[:, 1] = obj_box_batch[:, 1] * h
                    obj_box_batch[:, 2] = obj_box_batch[:, 2] * w
                    obj_box_batch[:, 3] = obj_box_batch[:, 3] * h

                    sub_box_batch = torch.cat([
                        sub_box_batch[:, 0:1] - sub_box_batch[:, 2:3] / 2,
                        sub_box_batch[:, 1:2] - sub_box_batch[:, 3:4] / 2,
                        sub_box_batch[:, 0:1] + sub_box_batch[:, 2:3] / 2,
                        sub_box_batch[:, 1:2] + sub_box_batch[:, 3:4] / 2
                    ], dim=-1)

                    obj_box_batch = torch.cat([
                        obj_box_batch[:, 0:1] - obj_box_batch[:, 2:3] / 2,
                        obj_box_batch[:, 1:2] - obj_box_batch[:, 3:4] / 2,
                        obj_box_batch[:, 0:1] + obj_box_batch[:, 2:3] / 2,
                        obj_box_batch[:, 1:2] + obj_box_batch[:, 3:4] / 2
                    ], dim=-1)

                    union_box_batch = torch.cat([
                        torch.clamp(torch.min(obj_box_batch[:, 0:1], sub_box_batch[:, 0:1]), min=0, max=w),
                        torch.clamp(torch.min(obj_box_batch[:, 1:2], sub_box_batch[:, 1:2]), min=0, max=h),
                        torch.clamp(torch.max(obj_box_batch[:, 2:3], sub_box_batch[:, 2:3]), min=0, max=w),
                        torch.clamp(torch.max(obj_box_batch[:, 3:4], sub_box_batch[:, 3:4]), min=0, max=h),
                    ], dim=-1)

                    for j in range(num_q):
                        union_box = union_box_batch[j]
                        radius_u = self.gaussian_radius((math.ceil(union_box[3] - union_box[1]),
                                                         math.ceil(union_box[2] - union_box[0])))
                        radius_u = max(0, int(radius_u))
                        self.draw_gaussian(src_mask[i][j], (int((union_box[2] + union_box[0]) / 2),
                                                            int((union_box[1] + union_box[3]) / 2)), radius_u)

                src_mask = torch.from_numpy(src_mask).type_as(memory)
                src_mask = src_mask.flatten(2)

                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, gaussian=src_mask)

            if self.return_intermediate:
                output_n = self.norm(output)
                intermediate.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        return torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes), torch.stack(obj_boxes)

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def draw_gaussian(self, heatmap, center, radius, k=1, factor=6):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / factor)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h


class TransformerDecoderROI(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        hidden_dim = 256
        num_obj_classes = 80
        num_verb_classes = 117
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                w=None, h=None):
        output = tgt

        intermediate = []
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []

        for layer in self.layers:
            if len(sub_boxes) == 0:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            else:
                sub_box, obj_box = sub_boxes[-1].clone(), obj_boxes[-1].clone()
                num_q, bs, _ = sub_box.size()
                src_mask = np.ones((bs, w, h), dtype=np.float32)
                for i in range(bs):
                    sub_box_batch = sub_box[:, i, :]
                    obj_box_batch = obj_box[:, i, :]

                    sub_box_batch[:, 0] = sub_box_batch[:, 0] * w
                    sub_box_batch[:, 1] = sub_box_batch[:, 1] * h
                    sub_box_batch[:, 2] = sub_box_batch[:, 2] * w
                    sub_box_batch[:, 3] = sub_box_batch[:, 3] * h

                    obj_box_batch[:, 0] = obj_box_batch[:, 0] * w
                    obj_box_batch[:, 1] = obj_box_batch[:, 1] * h
                    obj_box_batch[:, 2] = obj_box_batch[:, 2] * w
                    obj_box_batch[:, 3] = obj_box_batch[:, 3] * h

                    sub_box_batch = torch.cat([
                        sub_box_batch[:, 0:1] - sub_box_batch[:, 2:3] / 2,
                        sub_box_batch[:, 1:2] - sub_box_batch[:, 3:4] / 2,
                        sub_box_batch[:, 0:1] + sub_box_batch[:, 2:3] / 2,
                        sub_box_batch[:, 1:2] + sub_box_batch[:, 3:4] / 2
                    ], dim=-1)

                    obj_box_batch = torch.cat([
                        obj_box_batch[:, 0:1] - obj_box_batch[:, 2:3] / 2,
                        obj_box_batch[:, 1:2] - obj_box_batch[:, 3:4] / 2,
                        obj_box_batch[:, 0:1] + obj_box_batch[:, 2:3] / 2,
                        obj_box_batch[:, 1:2] + obj_box_batch[:, 3:4] / 2
                    ], dim=-1)

                    union_box_batch = torch.cat([
                        torch.clamp(torch.min(obj_box_batch[:, 0:1], sub_box_batch[:, 0:1]), min=0, max=w),
                        torch.clamp(torch.min(obj_box_batch[:, 1:2], sub_box_batch[:, 1:2]), min=0, max=h),
                        torch.clamp(torch.max(obj_box_batch[:, 2:3], sub_box_batch[:, 2:3]), min=0, max=w),
                        torch.clamp(torch.max(obj_box_batch[:, 3:4], sub_box_batch[:, 3:4]), min=0, max=h),
                    ], dim=-1)

                    for j in range(num_q):
                        union_box = union_box_batch[j]
                        src_mask[i][int(union_box[0]):int(union_box[2]), int(union_box[1]):int(union_box[3])] = 0

                src_mask = torch.from_numpy(src_mask).type_as(memory)
                src_mask = src_mask.transpose(1, 2).flatten(1)
                # src_mask[memory_key_padding_mask == 1] = 1
                src_mask = src_mask.type_as(memory_key_padding_mask)

                ###################################################################################
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=src_mask,
                               pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                output_n = self.norm(output)
                intermediate.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        return torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes), torch.stack(obj_boxes)


class TransformerDecoderGuide(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.encoded_pos = nn.Linear(256, 256)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                query_pos = query_pos + self.encoded_pos(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderPosGuide(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        hidden_dim = 256
        num_obj_classes = 80
        num_verb_classes = 117
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.encoded_pos_fc = nn.Linear(hidden_dim, hidden_dim)

    def encoded_pos(self, sub_box, obj_box):
        dim = 32
        dim_t = torch.arange(dim, dtype=torch.float32, device=obj_box.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / dim)
        pos_s = sub_box[:, :, :, None] / dim_t
        pos_o = obj_box[:, :, :, None] / dim_t
        pos_s = torch.stack((pos_s[:, :, :, 0::2].sin(), pos_s[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_o = torch.stack((pos_o[:, :, :, 0::2].sin(), pos_o[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_s, pos_o), dim=3)
        pos = pos.flatten(2)
        pos = self.encoded_pos_fc(pos)
        return pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                sub_box = self.sub_bbox_embed(output_n).sigmoid()
                sub_boxes.append(sub_box)
                obj_box = self.obj_bbox_embed(output_n).sigmoid()
                obj_boxes.append(obj_box)
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                # todo offset
                query_pos = query_pos + self.encoded_pos(sub_box, obj_box)
                print(query_pos.size())
                assert False

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class)


class TransformerDecoderPruning(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                obj_class_embed=None):
        output = tgt

        intermediate = []
        tgt_key_padding_masks = []
        num_q, bs, _ = output.size()

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 2 or i == 4:
                outputs_obj_class = F.softmax(obj_class_embed(intermediate[-1]), -1)
                bg_scores = outputs_obj_class[:, :, -1]

                if len(tgt_key_padding_masks) == 0:
                    _, top_index = torch.topk(bg_scores, k=25, dim=0)
                    tgt_key_padding_mask = torch.zeros([num_q, bs]).type_as(output).type(torch.bool)
                    for bx in range(bs):
                        tgt_key_padding_mask[top_index[:, bx], bx] = 1
                else:
                    tgt_key_padding_mask = tgt_key_padding_masks[-1].clone()
                    bg_scores[tgt_key_padding_mask] = 0
                    _, top_index = torch.topk(bg_scores, k=25, dim=0)
                    for bx in range(bs):
                        tgt_key_padding_mask[top_index[:, bx], bx] = 1

                tgt_key_padding_masks.append(tgt_key_padding_mask)
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask.transpose(0, 1),
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            else:
                if len(tgt_key_padding_masks) > 0:
                    tgt_key_padding_masks.append(tgt_key_padding_masks[-1])
                    output = layer(output, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_masks[-1].transpose(0, 1),
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos)
                else:
                    output = layer(output, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=None,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        return torch.stack(intermediate), torch.stack(tgt_key_padding_masks)


class TransformerDecoderPruningE2E(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        # self.binary_fc = nn.Linear(256, 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                obj_class_embed=None):
        output = tgt

        intermediate = []
        tgt_key_padding_masks = []
        num_q, bs, _ = output.size()
        prev_keeps = []

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i >= 3:
                ############################################################################################
                # todo gumbel_softmax is better?????
                bg_score = F.gumbel_softmax(obj_class_embed(intermediate[-1]), dim=-1)[:, :, -1]
                # 0-keep 1-bg
                if len(prev_keeps) > 0:
                    bg_score = bg_score + prev_keeps[-1]
                    prune_score_hard = prev_keeps[-1].clone()
                else:
                    prune_score_hard = torch.ones_like(bg_score, memory_format=torch.legacy_contiguous_format)
                _, top_index = torch.topk(bg_score, k=25, dim=0)

                prune_score_hard = prune_score_hard.scatter_(0, top_index, 0.0)
                prev_keeps.append(prune_score_hard)

                tgt_key_padding_mask = (1 - prune_score_hard).bool()
                tgt_key_padding_masks.append(tgt_key_padding_mask)
                #########################################################
                prune_score_hard = prune_score_hard - bg_score.detach() + bg_score
                output = output * prune_score_hard[:, :, None]
                # begin ############################################################################################
                # prune_score = F.gumbel_softmax(self.binary_fc(intermediate[-1]), hard=False, dim=-1)[:, :, 0]
                # # 0-keep 1-bg
                # if len(prev_keeps) > 0:
                #     prune_score = prune_score - prev_keeps[-1]
                #     prune_score_hard = prev_keeps[-1].clone()
                # else:
                #     prune_score_hard = torch.ones_like(prune_score, memory_format=torch.legacy_contiguous_format)
                # _, top_index = torch.topk(-prune_score, k=25, dim=0)
                #
                # prune_score_hard = prune_score_hard.scatter_(0, top_index, 0.0)
                # prev_keeps.append(prune_score_hard)
                #
                # tgt_key_padding_mask = (1 - prune_score_hard).bool()
                # tgt_key_padding_masks.append(tgt_key_padding_mask)
                # #########################################################
                # prune_score_hard = prune_score_hard - prune_score.detach() + prune_score
                # output = output * prune_score_hard[:, :, None]
                # end #########################################################
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask.transpose(0, 1),
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            else:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=None,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        return torch.stack(intermediate), torch.stack(tgt_key_padding_masks)


class TransformerDecoderPruningGumbelSoftmax(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.binary_fc = nn.Linear(256, 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        assert tgt_key_padding_mask is None
        output = tgt

        intermediate = []
        prev_keeps = []
        prune_scores = []
        num_q, bs, _ = output.size()

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i >= 3:
                prune_score = self.binary_fc(intermediate[-1])
                prune_scores.append(prune_score)
                prune_hard_labels = F.gumbel_softmax(prune_score, hard=True, dim=-1)

                # 0-keep 1-bg
                if len(prev_keeps) > 0:
                    cur_keep = prune_hard_labels[:, :, 0] * prev_keeps[-1]
                else:
                    cur_keep = prune_hard_labels[:, :, 0]
                # todo add to
                # tgt_key_padding_mask = (1 - cur_keep).bool().transpose(0, 1)
                # backward
                output = output * cur_keep[:, :, None]
                prev_keeps.append(prune_hard_labels[:, :, 0])
                ########################################################################
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        return torch.stack(intermediate), torch.stack(prune_scores)


class TransformerDecoderQG(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.l_num = 1

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        output1 = tgt

        intermediate = []
        for layer in self.layers[0:self.l_num]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        for layer in self.layers[self.l_num:]:
            output1 = layer(output1, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=output + query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output1))

        if self.norm is not None:
            output1 = self.norm(output1)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output1)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderTS(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        assert tgt_key_padding_mask is not None
        for layer in self.layers[self.begin_l:]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        return torch.stack(intermediate)


class TransformerDecoderTSNotShare(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers_t = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            intermediate.append(self.norm(output))

        return torch.stack(intermediate)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        assert tgt_key_padding_mask is not None
        for layer in self.layers_t:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            intermediate.append(self.norm(output))

        return torch.stack(intermediate)


class TransformerDecoderTSOffset(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)

            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSQPosObj(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher
        self.query_embed_image = nn.ModuleList(MLP(300 + 100, hidden_dim, hidden_dim, 2) for _ in range(num_layers))

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, tgt1, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = tgt1
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=torch.tanh(self.query_embed_image[i](query_pos_t)))

            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSOffsetTCDN(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t_verb: Optional[Tensor] = None,
                 query_pos_t_obj: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t, hs_t_verb = [], [], []
        #########################################################
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
        #########################################################
        output_t = query_pos_t_obj
        output_t_verb = query_pos_t_verb
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t_obj, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)
            output_t_verb = layer(output_t_verb, memory, tgt_mask=tgt_mask_t,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask_t,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=select_q)

            if self.return_intermediate:
                output_n = self.norm(output_t)
                output_n_verb = self.norm(output_t_verb)
                hs_t.append(output_n)
                hs_t_verb.append(output_n_verb)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                obj_class_t.append(self.obj_class_embed(output_n))
                verb_class_t.append(self.verb_class_embed(output_n_verb))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t), torch.stack(hs_t_verb)


class TransformerDecoderTSQPosEobj(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSQPosEobj1(nn.Module):

    def __init__(self, decoder_layer1, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer1, 1)
        self.layers.extend(_get_clones(decoder_layer, num_layers - 1))
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))
        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSQPosEobjAtt(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []

        hs, hs_t  = [], []
        atts_all, attt_all = [], []

        for layer in self.layers:
            output, att_s = layer(output, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask_t,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(atts_all), torch.stack(attt_all), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSQPosEobjAttSOcc(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []

        hs, hs_t  = [], []
        atts_all, attt_all = [], []

        # learnable
        for layer in self.layers:
            output, att_s = layer(output, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        # shift + selfocc
        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask_t,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_embed_q, use_hard_mask=True)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(atts_all), torch.stack(attt_all), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSQPosEobjVerb(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e_obj: Optional[Tensor] = None,
                 query_embed_e_verb: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_embed_e_verb
        for i in range(self.begin_l, 3):
            layer = self.layers[i]
            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        output_t = output_t + query_embed_e_obj
        for i in range(3, len(self.layers)):
            layer = self.layers[i]
            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSQPosEobjWithS(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

        self.query_embed_sp = MLP(8, hidden_dim, hidden_dim, 2)
        self.query_embed_obj = MLP(300, hidden_dim, hidden_dim, 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

                sub_bbox = sub_boxes[-1]
                obj_bbox = obj_boxes[-1]
                sp = self.query_embed_sp(torch.cat([sub_bbox, obj_bbox], dim=-1))
                query_pos = query_pos + sp

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        query_pos_all = [query_pos]

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

                sub_bbox = sub_boxes[-1]
                obj_bbox = obj_boxes[-1]
                sp = self.query_embed_sp(torch.cat([sub_bbox, obj_bbox], dim=-1))
                query_pos = query_pos + sp
                query_pos_all.append(query_pos)
        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_embed_e, query_pos_all[i])

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_embed_q + select_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSOffsetDQ(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher
        self.fc_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_verb = nn.Linear(hidden_dim, hidden_dim)

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        query_pos_v, query_pos_o, output_o, output_v = query_pos, query_pos, tgt, tgt

        for layer in self.layers:
            output_o = layer(output_o, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_pos_o)
            output_v = layer(output_v, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_pos_v)
            if self.return_intermediate:
                output_on = self.norm(output_o)
                output_vn = self.norm(output_v)
                sub_boxes.append(self.sub_bbox_embed(output_on).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_on).sigmoid())
                verb_class.append(self.verb_class_embed(output_vn))
                obj_class.append(self.obj_class_embed(output_on))

                query_pos_o = query_pos_o + self.fc_obj(output_on)
                query_pos_v = query_pos_v + self.fc_verb(output_vn)

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs_o, hs_v, hs_t_o, hs_t_v = [], [], [], []

        query_pos_v, query_pos_o, output_o, output_v = query_pos, query_pos, tgt, tgt
        query_pos_v_all = [query_pos_v]
        query_pos_o_all = [query_pos_o]

        for layer in self.layers:
            output_o = layer(output_o, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_pos_o)
            output_v = layer(output_v, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_pos_v)
            if self.return_intermediate:
                output_on = self.norm(output_o)
                output_vn = self.norm(output_v)
                hs_o.append(output_on)
                hs_v.append(output_vn)

                sub_boxes.append(self.sub_bbox_embed(output_on).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_on).sigmoid())
                verb_class.append(self.verb_class_embed(output_vn))
                obj_class.append(self.obj_class_embed(output_on))

                query_pos_o = query_pos_o + self.fc_obj(output_on)
                query_pos_v = query_pos_v + self.fc_verb(output_vn)
                query_pos_v_all.append(query_pos_v)
                query_pos_o_all.append(query_pos_o)

        output_t_o, output_t_v = query_pos_t, query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q_o = self.select(indices, query_pos_t, query_pos_o_all[i])
            select_q_v = self.select(indices, query_pos_t, query_pos_v_all[i])

            output_t_o = layer(output_t_o, memory, tgt_mask=tgt_mask_t,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask_t,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=select_q_o)
            output_t_v = layer(output_t_v, memory, tgt_mask=tgt_mask_t,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask_t,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=select_q_v)
            if self.return_intermediate:
                output_n_to = self.norm(output_t_o)
                output_n_tv = self.norm(output_t_v)
                hs_t_o.append(output_n_to)
                hs_t_v.append(output_n_tv)

                sub_boxes_t.append(self.sub_bbox_embed(output_n_to).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n_to).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n_tv))
                obj_class_t.append(self.obj_class_embed(output_n_to))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs_o), torch.stack(hs_v), torch.stack(hs_t_o), torch.stack(hs_t_v)


class TransformerDecoderTSOffsetImageLevel(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 query_pos_image_level: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None

        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        verb_class_i = []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        #################################################################################
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
        #################################################################################
        output_i = torch.zeros_like(query_pos_image_level)
        for layer in self.layers:
            output_i = layer(output_i, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=query_pos_image_level)
            if self.return_intermediate:
                verb_class_i.append(self.verb_class_embed(self.norm(output_i)))
        #################################################################################
        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t), torch.stack(verb_class_i)


class TransformerDecoderTSOffsetQM(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

        self.query_embed_sp = nn.Linear(8, 100)
        self.query_embed_image = nn.Linear(400, hidden_dim)
        self.coco_80 = torch.from_numpy(np.load('data/hico_20160224_det/annotations/coco_80.npy')).cuda()

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for i in range(len(self.layers)):
            layer = self.layers[i]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

                sub_bbox = sub_boxes[-1]
                obj_bbox = obj_boxes[-1]
                obj_prob = F.softmax(obj_class[-1], -1)
                _, obj_labels = obj_prob[..., :-1].max(-1)
                sp = self.query_embed_sp(torch.cat([sub_bbox, obj_bbox], dim=-1))
                num_q, bs = obj_labels.size()
                obj_labels1 = obj_labels.view(-1)
                word_vec = self.coco_80[obj_labels1]
                word_vec = word_vec.view(num_q, bs, 300).type_as(sp)
                query_embed_vec = self.query_embed_image(torch.cat([word_vec, sp], dim=-1))
                query_pos = query_pos + query_embed_vec

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        query_pos_all = [query_pos]

        for i in range(len(self.layers)):
            layer = self.layers[i]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

                sub_bbox = sub_boxes[-1]
                obj_bbox = obj_boxes[-1]
                obj_prob = F.softmax(obj_class[-1], -1)
                _, obj_labels = obj_prob[..., :-1].max(-1)
                sp = self.query_embed_sp(torch.cat([sub_bbox, obj_bbox], dim=-1))
                num_q, bs = obj_labels.size()
                obj_labels1 = obj_labels.view(-1)
                word_vec = self.coco_80[obj_labels1]
                word_vec = word_vec.view(num_q, bs, 300).type_as(sp)
                query_embed_vec = self.query_embed_image(torch.cat([word_vec, sp], dim=-1))
                query_pos = query_pos + query_embed_vec
                query_pos_all.append(query_pos)

        ################################################################
        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos_all[i])

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSOffsetSMask(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None, h=None, w=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for i in range(len(self.layers) // 2):
            layer = self.layers[i]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
        ##############################################################################
        sub_box, obj_box = sub_boxes[-1], obj_boxes[-1]
        num_q, bs, _ = sub_box.size()

        src_mask = np.zeros((bs, num_q, w, h), dtype=np.float32)
        for i in range(bs):
            h1, w1 = target[i]["size"]
            sub_box_batch = sub_box[:, i, :]
            obj_box_batch = obj_box[:, i, :]

            for j in range(num_q):
                sub_box_batch_j = sub_box_batch[j]
                obj_box_batch_j = obj_box_batch[j]

                sub_box_resize = [sub_box_batch_j[0] * w1 / 32, sub_box_batch_j[1] * h1 / 32,
                                  sub_box_batch_j[2] * w1 / 32, sub_box_batch_j[3] * h1 / 32]
                obj_box_resize = [obj_box_batch_j[0] * w1 / 32, obj_box_batch_j[1] * h1 / 32,
                                  obj_box_batch_j[2] * w1 / 32, obj_box_batch_j[3] * h1 / 32]

                radius_s = gaussian_radius((math.ceil(sub_box_resize[3]), math.ceil(sub_box_resize[2])))
                radius_s = max(0, int(radius_s))
                radius_o = gaussian_radius((math.ceil(obj_box_resize[3]), math.ceil(obj_box_resize[2])))
                radius_o = max(0, int(radius_o))
                draw_gaussian(src_mask[i][j], (int(sub_box_resize[0]), int(sub_box_resize[1])), radius_s)
                draw_gaussian(src_mask[i][j], (int(obj_box_resize[0]), int(obj_box_resize[1])), radius_o)

        src_mask = torch.from_numpy(src_mask).type_as(memory)
        src_mask = src_mask.transpose(2, 3).flatten(2).permute(1, 2, 0)
        ##############################################################################
        for i in range(len(self.layers) // 2, len(self.layers)):
            layer = self.layers[i]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, gaussian=src_mask)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        ##############################################################################
        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSOffsetTMask(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None, gaussian=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q, gaussian=gaussian)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSOffsetTMask1(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None,
                 src_mask=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q, gaussian=src_mask)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSOffset2T(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers1)):
            layer = self.layers1[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderTSOffsetPrune(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        num_obj_classes = 80
        num_verb_classes = 117
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher
        self.topk = 25

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target, mask):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target, mask)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        tgt_key_padding_masks = []
        num_q, bs, _ = output.size()

        for i in range(len(self.layers)):
            layer = self.layers[i]
            # todo
            if i >= 3:
                outputs_obj_class = F.softmax(obj_class[-1], -1)
                bg_scores = outputs_obj_class[:, :, -1]

                if len(tgt_key_padding_masks) == 0:
                    _, top_index = torch.topk(bg_scores, k=self.topk, dim=0)
                    tgt_key_padding_mask = torch.zeros([num_q, bs]).type_as(output).type(torch.bool)
                    for bx in range(bs):
                        tgt_key_padding_mask[top_index[:, bx], bx] = 1
                else:
                    tgt_key_padding_mask = tgt_key_padding_masks[-1].clone()
                    bg_scores[tgt_key_padding_mask] = 0
                    _, top_index = torch.topk(bg_scores, k=self.topk, dim=0)
                    for bx in range(bs):
                        tgt_key_padding_mask[top_index[:, bx], bx] = 1

                tgt_key_padding_masks.append(tgt_key_padding_mask)

                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask.transpose(0, 1),
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            else:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=None,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs), torch.stack(tgt_key_padding_masks)

    def forwardt(self, tgt, tgt_t, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        tgt_key_padding_masks = []
        num_q, bs, _ = output.size()

        for i in range(len(self.layers)):
            layer = self.layers[i]
            # todo
            if i >= 3:
                outputs_obj_class = F.softmax(obj_class[-1], -1)
                bg_scores = outputs_obj_class[:, :, -1]

                if len(tgt_key_padding_masks) == 0:
                    _, top_index = torch.topk(bg_scores, k=self.topk, dim=0)
                    tgt_key_padding_mask = torch.zeros([num_q, bs]).type_as(output).type(torch.bool)
                    for bx in range(bs):
                        tgt_key_padding_mask[top_index[:, bx], bx] = 1
                else:
                    tgt_key_padding_mask = tgt_key_padding_masks[-1].clone()
                    bg_scores[tgt_key_padding_mask] = 0
                    _, top_index = torch.topk(bg_scores, k=self.topk, dim=0)
                    for bx in range(bs):
                        tgt_key_padding_mask[top_index[:, bx], bx] = 1

                tgt_key_padding_masks.append(tgt_key_padding_mask)
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask.transpose(0, 1),
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            else:
                assert tgt_key_padding_mask is None
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
        #########################################################################
        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]

            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            # todo
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target,
                                     tgt_key_padding_masks[i - 3].transpose(0, 1))
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)

            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))
        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t), torch.stack(tgt_key_padding_masks)


class TransformerDecoderTSObjVerb(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        num_obj_classes = 80
        num_verb_classes = 117
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_verb_classes)
        )
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1),
               }
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 query_pos_t1: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_t1, obj_boxes_t1, verb_class_t1, obj_class_t1 = [], [], [], []
        hs, hs1, hs_t, hs_t1 = [], [], [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_temp = self.verb_class_embed[0:2](output_n)
                hs1.append(verb_temp)
                verb_class.append(self.verb_class_embed[2:](verb_temp))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_pos_t
        output_t1 = query_pos_t1
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)

            output_t1 = layer(output_t1, memory, tgt_mask=tgt_mask_t,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask_t,
                              memory_key_padding_mask=memory_key_padding_mask,
                              pos=pos, query_pos=select_q)

            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

                output_n = self.norm(output_t1)
                sub_boxes_t1.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t1.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_temp = self.verb_class_embed[1:2](output_n)
                hs_t1.append(verb_temp)
                verb_class_t1.append(self.verb_class_embed[2:](verb_temp))
                obj_class_t1.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(sub_boxes_t1), torch.stack(obj_boxes_t1), torch.stack(obj_class_t1), torch.stack(
            verb_class_t1), \
               torch.stack(hs), torch.stack(hs1), torch.stack(hs_t), torch.stack(hs_t1)


class TransformerDecoderGS(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, n_points=4):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        #################################################################
        self.obj_class_embed = nn.Linear(256, 80 + 1)
        self.verb_class_embed = nn.Linear(256, 117)
        self.verb_class_embed1 = nn.Linear(256, 117)
        self.sub_bbox_embed = MLP(256, 256, 4, 3)
        self.obj_bbox_embed = MLP(256, 256, 4, 3)
        #################################################################
        self.reference_points = nn.Linear(256, 2)
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(256, n_points * 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                w=None, h=None):
        output = tgt
        wh, bs, c = memory.size()
        memory_sample = memory.permute(1, 2, 0).view(bs, c, h, w)

        intermediate_sub_box = []
        intermediate_obj_box = []
        intermediate_verb_cls = []
        intermediate_obj_cls = []

        intermediate_verb_cls1 = []

        reference_points = self.reference_points(query_pos).sigmoid()

        for layer in self.layers:
            sampling_offsets = self.sampling_offsets(query_pos + output)
            reference_points_repeat = reference_points.repeat(1, 1, self.n_points)
            sampling_offsets[:, :, 0::2] = sampling_offsets[:, :, 0::2] / w
            sampling_offsets[:, :, 1::2] = sampling_offsets[:, :, 1::2] / h
            reference_points_repeat += sampling_offsets
            # n, c, w, h and n, w, h, 2
            reference_points_repeat = reference_points_repeat.reshape(bs, -1, self.n_points, 2)
            sample_feature = F.grid_sample(memory_sample, reference_points_repeat, padding_mode="border")
            sample_feature = sample_feature.permute(2, 0, 1, 3).mean(-1)
            intermediate_verb_cls1.append(self.verb_class_embed1(sample_feature))
            ###################################################################################
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            output1 = self.norm(output)
            intermediate_verb_cls.append(self.verb_class_embed(output1))
            intermediate_obj_cls.append(self.obj_class_embed(output1))
            sub_boxes = self.sub_bbox_embed(output1)
            obj_boxes = self.obj_bbox_embed(output1)
            intermediate_sub_box.append(sub_boxes.sigmoid())
            intermediate_obj_box.append(obj_boxes.sigmoid())
            ###################################################################################
            # update
            # center_point_x = (sub_boxes[:, :, 0:1] + obj_boxes[:, :, 0:1]) / 2
            # center_point_y = (sub_boxes[:, :, 1:2] + obj_boxes[:, :, 1:2]) / 2
            # center_point = torch.cat([center_point_x, center_point_y], dim=-1)
            # new_reference_points = center_point + inverse_sigmoid(reference_points)
            # new_reference_points = new_reference_points.sigmoid()

        ###################################################################################
        return torch.stack(intermediate_obj_box), torch.stack(intermediate_sub_box), \
               torch.stack(intermediate_verb_cls), torch.stack(intermediate_obj_cls), \
               torch.stack(intermediate_verb_cls1)


class TransformerDecoderFFN(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers + 2)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers[:5]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        output_human = self.layers[5](output, memory, tgt_mask=tgt_mask,
                                      memory_mask=memory_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask,
                                      pos=pos, query_pos=query_pos)
        output_obj = self.layers[6](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_pos)
        output_verb = self.layers[7](output, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate), output_human, output_obj, output_verb

        return output


class TransformerDecoderLayerAttMapHardmAttnOccShift2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttentionEach2(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     attn_select: Optional[Tensor] = None,
                     use_hard_mask: bool = False,
                     rate=0.4):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map, att_select = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                        key=self.with_pos_embed(memory, pos),
                                                        value=memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask,
                                                        attn_select=attn_select, use_hard_mask=use_hard_mask, rate=rate)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map, att_select

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    attn_select: Optional[Tensor] = None,
                    use_hard_mask: bool = False,
                    rate=0.4):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   use_hard_mask=use_hard_mask, rate=rate)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_select: Optional[Tensor] = None,
                use_hard_mask: bool = False,
                rate=0.4):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask, rate)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask, rate)


class TransformerDecoderTSQPosEobjAttOccShift2(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 occ_embed_q: Optional[Tensor] = None,
                 occ_embed_e: Optional[Tensor] = None,
                 shift_occ_pass=False, occ_pass=False):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_occ, obj_boxes_occ, verb_class_occ, obj_class_occ = [], [], [], []
        hs, hs_t, hs_occ = [], [], []
        atts_all, attt_all, attocc_all = [], [], []
        att_select_all = []

        # learnable
        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                att_select_all.append(att_select)

        # shift
        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask_t,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q, use_hard_mask=shift_occ_pass, rate=0.2)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        # occ
        output_occ = occ_embed_e
        for i in range(len(self.layers)):
            layer = self.layers[i]
            output_occ, att_occ, _ = layer(output_occ, memory, tgt_mask=tgt_mask_t,
                                           memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=occ_embed_q,
                                           attn_select=att_select_all[i], use_hard_mask=True)
            if self.return_intermediate:
                output_n = self.norm(output_occ)
                hs_occ.append(output_n)
                attocc_all.append(att_occ)
                sub_boxes_occ.append(self.sub_bbox_embed(output_occ).sigmoid())
                obj_boxes_occ.append(self.obj_bbox_embed(output_occ).sigmoid())
                verb_class_occ.append(self.verb_class_embed(output_occ))
                obj_class_occ.append(self.obj_class_embed(output_occ))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(sub_boxes_occ), torch.stack(obj_boxes_occ), torch.stack(obj_class_occ), torch.stack(verb_class_occ), \
               torch.stack(atts_all), torch.stack(attt_all), torch.stack(attocc_all), \
               torch.stack(hs), torch.stack(hs_t), torch.stack(hs_occ)


class HOITransformerTSQPOSEOBJAttOccShiftOcc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnOccShift2(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttOccShift2(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None,
                occ_embed_q=None, occ_embed_e=None,
                shift_occ_pass=False, occ_pass=True):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)

        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, sub_boxes_occ, obj_boxes_occ, obj_class_occ, verb_class_occ,\
            att_s, att_t, att_occ, hs, hs_t, hs_occ = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      occ_embed_q=occ_embed_q, occ_embed_e=occ_embed_e,
                                      shift_occ_pass=shift_occ_pass, occ_pass=occ_pass)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_occ.transpose(1, 2), obj_boxes_occ.transpose(1, 2), \
                   obj_class_occ.transpose(1, 2), verb_class_occ.transpose(1, 2),   \
                   att_s, att_t, att_occ, hs.transpose(1, 2), hs_t.transpose(1, 2),  hs_occ.transpose(1, 2)

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class TransformerDecoderFFN7l(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, 2)
        self.layers2 = _get_clones(decoder_layer, 2)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []

        for layer in self.layers[:-2]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate_h.append(self.norm(output))
                intermediate_o.append(self.norm(output))
                intermediate_v.append(self.norm(output))

        output_h = output
        output_o = output
        output_v = output

        for i in range(5, 7):
            output_h = self.layers[i](output_h, memory, tgt_mask=tgt_mask,
                                      memory_mask=memory_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask,
                                      pos=pos, query_pos=query_pos)
            output_o = self.layers1[i - 5](output_o, memory, tgt_mask=tgt_mask,
                                           memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=query_pos)
            output_v = self.layers2[i - 5](output_v, memory, tgt_mask=tgt_mask,
                                           memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask,
                                           pos=pos, query_pos=query_pos)
            intermediate_h.append(self.norm(output_h))
            intermediate_o.append(self.norm(output_o))
            intermediate_v.append(self.norm(output_v))

        return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)


class TransformerDecoderFFNAll(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human


class TransformerDecoderFFNAllRec(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 begin_l=3):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

        hidden_dim = 256
        num_obj_classes = 11
        num_verb_classes = 10
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.query_embed_sp = nn.Linear(8, 100)
        self.query_embed_image = nn.Linear(300 + 100, hidden_dim)
        import numpy as np
        self.coco_80 = torch.from_numpy(np.load('data/hico_20160224_det/annotations/coco_80.npy')).cuda()

        self.begin_l = begin_l

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_sub_boxes = []
        intermediate_obj_boxes = []
        intermediate_obj_cate = []
        intermediate_verb = []
        intermediate_h = []
        intermediate_o = []
        intermediate_v = []

        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

                intermediate_sub_boxes.append(self.sub_bbox_embed(self.norm(output_human)).sigmoid())
                intermediate_obj_boxes.append(self.obj_bbox_embed(self.norm(output_obj)).sigmoid())
                intermediate_obj_cate.append(self.obj_class_embed(self.norm(output_obj)))
                intermediate_verb.append(self.verb_class_embed(self.norm(output_verb)))
        ###################################################################################
        sub_bbox = intermediate_sub_boxes[-1]
        obj_bbox = intermediate_obj_boxes[-1]
        obj_prob = F.softmax(intermediate_obj_boxes[-1], -1)

        _, obj_labels = obj_prob[..., :-1].max(-1)

        sp = self.query_embed_sp(torch.cat([sub_bbox, obj_bbox], dim=-1))
        num_q, bs = obj_labels.size()
        obj_labels1 = obj_labels.view(-1)
        word_vec = self.coco_80[obj_labels1]
        word_vec = word_vec.view(num_q, bs, 300).type_as(sp)
        query_embed_vec = self.query_embed_image(torch.cat([word_vec, sp], dim=-1))
        ###################################################################################
        # todo
        output_verb1 = output_verb.detach()
        output_obj1 = output_obj.detach()
        output_human1 = output_human.detach()

        for i in range(self.begin_l, self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human1 = layer_ho(output_human1, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     pos=pos, query_pos=self.fc1(query_embed_vec))
            output_obj1 = layer_ho(output_obj1, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=self.fc2(query_embed_vec))
            output_verb1 = layer_v(output_verb1, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=self.fc3(query_embed_vec))
            if self.return_intermediate:
                intermediate_sub_boxes.append(self.sub_bbox_embed(self.norm(output_human1)).sigmoid())
                intermediate_obj_boxes.append(self.obj_bbox_embed(self.norm(output_obj1)).sigmoid())
                intermediate_obj_cate.append(self.obj_class_embed(self.norm(output_obj1)))
                intermediate_verb.append(self.verb_class_embed(self.norm(output_verb1)))
        return torch.stack(intermediate_sub_boxes), torch.stack(intermediate_obj_boxes), \
               torch.stack(intermediate_obj_cate), torch.stack(intermediate_verb)


class TransformerDecoderFFNAllTS(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=3):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.begin_l = begin_l

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []

        for i in range(self.begin_l, self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human


class TransformerDecoder2lQG(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.l_num = 1

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        output_verb1 = tgt
        output_obj1 = tgt
        output_human1 = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(0, self.l_num):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        for i in range(self.l_num, self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human1 = layer_ho(output_human1, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     pos=pos, query_pos=self.fc1(output_human + query_pos))
            output_obj1 = layer_ho(output_obj1, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=self.fc2(output_obj + query_pos))
            output_verb1 = layer_v(output_verb1, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=self.fc3(output_verb + query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human1))
                intermediate_o.append(self.norm(output_obj1))
                intermediate_v.append(self.norm(output_verb1))

        if self.norm is not None:
            output_human1 = self.norm(output_human1)
            output_obj1 = self.norm(output_obj1)
            output_verb1 = self.norm(output_verb1)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human1)
                intermediate_o.append(output_obj1)
                intermediate_v.append(output_verb1)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human


class TransformerDecoderFFNAllOV(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

        self.obj_class_embed = nn.Linear(256, 80 + 1)
        self.sub_bbox_embed = MLP(256, 256, 4, 3)
        self.obj_bbox_embed = MLP(256, 256, 4, 3)

        # self.verb_class_embed = nn.Linear(256, 117)
        self.verb_to_verb_class = nn.Linear(600, 256)

        import numpy as np
        coco_80 = torch.from_numpy(np.load('data/hico_20160224_det/annotations/coco_80.npy')).cuda()
        verb_117 = torch.from_numpy(np.load('data/hico_20160224_det/annotations/verb_117.npy')).cuda()

        coco_80 = coco_80.unsqueeze(1).repeat(1, 117, 1)
        verb_117 = verb_117.unsqueeze(0).repeat(80, 1, 1)
        self.obj_verb = torch.cat([coco_80, verb_117], dim=-1)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        sub_boxes = []
        obj_boxes = []
        obj_cates = []
        verbs = []

        for i in range(self.num_layers - 1):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                output_human1 = self.norm(output_human)
                output_obj1 = self.norm(output_obj)
                output_verb1 = self.norm(output_verb)

                sub_boxes.append(self.sub_bbox_embed(output_human1).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_obj1).sigmoid())
                obj_cates.append(self.obj_class_embed(output_obj1))
                verbs.append(self.verb_class_embed(output_verb1))

                if i == self.num_layers - 2:
                    obj_cate = obj_cates[-1].detach()
                    obj_prob = F.softmax(obj_cate, -1)
                    _, obj_labels = obj_prob[..., :-1].max(-1)
                    N, bs, c = output_obj1.size()
                    a1 = output_obj1.unsqueeze(-1).repeat(1, 1, 1, 81)
                    a2 = self.obj_class_embed.weight.t().unsqueeze(0).unsqueeze(0).repeat(N, bs, 1, 1)
                    out2 = a1 * a2
                    obj_labels_index = obj_labels.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, c, 1)
                    q_weight = out2.gather(index=obj_labels_index, dim=-1).squeeze(-1)

        for i in range(self.num_layers - 1, self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            # todo
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=q_weight * self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                output_human1 = self.norm(output_human)
                output_obj1 = self.norm(output_obj)
                output_verb1 = self.norm(output_verb)

                sub_boxes.append(self.sub_bbox_embed(output_human1).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_obj1).sigmoid())
                obj_cates.append(self.obj_class_embed(output_obj1))
                verbs.append(self.verb_class_embed(output_verb1))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_cates), torch.stack(verbs)


class TransformerDecoderFFN2lRR(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256 * 2, 256)
        self.fc4 = nn.Linear(256, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))

            # todo
            output_verb = layer_v(output_verb, memory=self.fc3(torch.cat([output_human, output_obj], dim=-1)),
                                  query_pos=self.fc4(query_pos))

            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human


class TransformerDecoderFFN2l3q(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        # self.layers2 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt
        output_hoi = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        intermediate_hoi = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            # layer_t = self.layers2[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            # todo
            output_hoi = layer_v(output_hoi, memory=output_human + output_obj,
                                 query_pos=query_pos)

            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))
                intermediate_hoi.append(self.norm(output_hoi))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            output_hoi = self.norm(output_hoi)

            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()
                intermediate_hoi.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)
                intermediate_hoi.append(output_hoi)

        return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v), torch.stack(
            intermediate_hoi)


class TransformerDecoder2lVerbApart(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.layers2 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.verb_fc = nn.Linear(256, 117 * 256)
        self.verb_class_embed = nn.Linear(256, 117)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                image_action_labels=None,
                training=False):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt
        output_verb1 = None

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        intermediate_v_apart = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            q3 = self.fc3(query_pos)
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=q3)
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))
            ######################################################################
            layer_v1 = self.layers2[i]
            num_q, bs, c = q3.size()
            q_v = self.verb_fc(q3).reshape(num_q, bs, 117, 256)

            if training:
                image_action_labels = image_action_labels.squeeze(1)

                max_verb_num = 0
                for index in range(bs):
                    s = sum(image_action_labels[index] > 0)
                    max_verb_num = max(max_verb_num, s)
                #####################################################################################
                query_repeat = torch.zeros(num_q, bs, max_verb_num, c).type_as(q3)
                query_repeat_mask = torch.ones([num_q, bs, max_verb_num]).type_as(q3).type(torch.bool)
                verb_mask = torch.zeros([num_q, bs, max_verb_num, 117]).cuda()

                for index in range(bs):
                    mask = image_action_labels[index] > 0
                    q_v1 = torch.stack([q_v[j, index][mask] for j in range(num_q)])
                    if len(q_v1) == 0:
                        query_repeat_mask[:, index, :] = False
                    else:
                        query_repeat_mask[:, index, 0:q_v1.size()[1]] = False
                    query_repeat[:, index, :q_v1.size()[1], :] = q_v1
                    verb_mask[:, index, :q_v1.size()[1], mask] = 1

                query_repeat_mask = query_repeat_mask.transpose(0, 1)
                #####################################################################################
                if output_verb1 is None:
                    output_verb1 = torch.zeros_like(query_repeat)

                for index in range(max_verb_num):
                    output_verb1[:, :, index, :] = layer_v1(output_verb1[:, :, index, :], memory, tgt_mask=tgt_mask,
                                                            memory_mask=memory_mask,
                                                            tgt_key_padding_mask=query_repeat_mask[:, :, index],
                                                            memory_key_padding_mask=memory_key_padding_mask,
                                                            pos=pos, query_pos=query_repeat[:, :, index, :])
                output_verb1[np.isnan(output_verb1.detach().cpu().numpy())] = 0
                output_verb_feature = self.verb_class_embed(self.norm(output_verb1))
                output_verb_feature = (output_verb_feature.sigmoid()) * verb_mask
                output_verb_feature = torch.sum(output_verb_feature, dim=-2)
                intermediate_v_apart.append(output_verb_feature)
            else:
                if output_verb1 is None:
                    output_verb1 = torch.zeros_like(q_v)
                for index in range(117):
                    output_verb1[:, :, index, :] = layer_v1(output_verb1[:, :, index:], memory, tgt_mask=tgt_mask,
                                                            memory_mask=memory_mask,
                                                            tgt_key_padding_mask=None,
                                                            memory_key_padding_mask=memory_key_padding_mask,
                                                            pos=pos, query_pos=q_v[:, :, index, :])
                output_verb_feature = self.verb_class_embed(self.norm(output_verb1))
                verb_mask = torch.eye(117)
                verb_mask = verb_mask.unsqueeze(0).unsqueeze(0).repeat(num_q, bs, 1, 1).cuda()
                output_verb_feature = output_verb_feature.sigmoid() * verb_mask
                output_verb_feature = torch.sum(output_verb_feature, dim=-2)
                intermediate_v_apart.append(output_verb_feature)

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v), \
                   torch.stack(intermediate_v_apart)

        return output_human


class TransformerDecoderFFNAllEMul(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, tgt, memory1, memory2, memory3,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory1, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory2, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory3, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human


class TransformerDecoder2lInter(nn.Module):

    def __init__(self, decoder_layer, interaction_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.inter_layers = _get_clones(interaction_layer, num_layers)

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            # todo
            _, output_verb = self.inter_layers[i](
                torch.cat([output_human, output_obj], dim=-1), output_verb
            )
            intermediate_h.append(self.norm(output_human))
            intermediate_o.append(self.norm(output_obj))
            intermediate_v.append(self.norm(output_verb))

        return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)


class TransformerDecoderPartSum(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        # self.layers2 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt
        output_hoi = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        intermediate_hoi = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            # layer_t = self.layers2[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            ##############################################################################
            output_hoi = layer_v(output_hoi, memory, tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))
                intermediate_hoi.append(self.norm(output_hoi))

        return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v), torch.stack(
            intermediate_hoi)


class TransformerDecoder2lCom(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256 * 2, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt
        output_hoi = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        intermediate_hoi = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            ##############################################################################
            output_hoi = layer_v(output_hoi, memory, tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 pos=pos, query_pos=self.fc4(torch.cat([output_obj, output_verb], dim=-1)))

            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))
                intermediate_hoi.append(self.norm(output_hoi))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            output_hoi = self.norm(output_hoi)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()
                intermediate_hoi.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)
                intermediate_hoi.append(output_hoi)

        return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v), torch.stack(
            intermediate_hoi)


class TransformerDecoder2lHOCom(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256 * 2, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt
        output_hoi = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        intermediate_hoi = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos))
            ##############################################################################
            output_hoi = layer_v(output_hoi, memory, tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 pos=pos, query_pos=self.fc4(torch.cat([output_human, output_obj], dim=-1)))

            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))
                intermediate_hoi.append(self.norm(output_hoi))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            output_hoi = self.norm(output_hoi)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()
                intermediate_hoi.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)
                intermediate_hoi.append(output_hoi)

        return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v), torch.stack(
            intermediate_hoi)


class TransformerDecoderFFNAll1l(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output_verb = tgt
        output_obj = tgt
        output_human = tgt

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            output_human = layer(output_human, memory, tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask,
                                 pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer(output_obj, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=self.fc2(query_pos))
            output_verb = layer(output_verb, memory, tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                pos=pos, query_pos=self.fc3(query_pos))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human


class TransformerDecoderFFN2lIq(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers1 = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, tgt1, tgt2, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_obj_verb: Optional[Tensor] = None):
        output_human = tgt1
        output_obj = torch.cat([tgt1, tgt2[0:1, :, :]], dim=0)
        output_verb = torch.cat([tgt1, tgt2[1:2, :, :]], dim=0)
        query_pos_o = torch.cat([query_pos, query_obj_verb[0:1, :, :]], dim=0)
        query_pos_v = torch.cat([query_pos, query_obj_verb[1:2, :, :]], dim=0)

        intermediate_h = []
        intermediate_o = []
        intermediate_v = []
        for i in range(self.num_layers):
            layer_ho = self.layers[i]
            layer_v = self.layers1[i]
            output_human = layer_ho(output_human, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=self.fc1(query_pos))
            output_obj = layer_ho(output_obj, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc2(query_pos_o))
            output_verb = layer_v(output_verb, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=self.fc3(query_pos_v))
            if self.return_intermediate:
                intermediate_h.append(self.norm(output_human))
                intermediate_o.append(self.norm(output_obj))
                intermediate_v.append(self.norm(output_verb))

        if self.norm is not None:
            output_human = self.norm(output_human)
            output_obj = self.norm(output_obj)
            output_verb = self.norm(output_verb)
            if self.return_intermediate:
                intermediate_h.pop()
                intermediate_o.pop()
                intermediate_v.pop()

                intermediate_h.append(output_human)
                intermediate_o.append(output_obj)
                intermediate_v.append(output_verb)

        if self.return_intermediate:
            return torch.stack(intermediate_h), torch.stack(intermediate_o), torch.stack(intermediate_v)

        return output_human


class TransformerDecoderQM(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.l_num = 1
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        hidden_dim = 256
        num_obj_classes = 80
        num_verb_classes = 117
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.query_embed_sp = nn.Linear(8, 100)
        self.query_embed_image = nn.Linear(100 + 300, hidden_dim)

        import numpy as np
        self.coco_80 = torch.from_numpy(np.load('data/hico_20160224_det/annotations/coco_80.npy')).cuda()
        ################################################################

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        output1 = tgt

        intermediate_sub_box = []
        intermediate_obj_box = []
        intermediate_verb_cls = []
        intermediate_obj_cls = []

        for layer in self.layers[0:self.l_num]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                intermediate_sub_box.append(self.sub_bbox_embed(output_n).sigmoid())
                intermediate_obj_box.append(self.obj_bbox_embed(output_n).sigmoid())
                intermediate_verb_cls.append(self.verb_class_embed(output_n))
                intermediate_obj_cls.append(self.obj_class_embed(output_n))

        # ################################################################################
        sub_bbox = intermediate_sub_box[-1]
        obj_bbox = intermediate_obj_box[-1]
        obj_prob = F.softmax(intermediate_obj_cls[-1], -1)

        _, obj_labels = obj_prob[..., :-1].max(-1)

        sp = self.query_embed_sp(torch.cat([sub_bbox, obj_bbox], dim=-1))
        num_q, bs = obj_labels.size()
        obj_labels1 = obj_labels.view(-1)
        word_vec = self.coco_80[obj_labels1]
        word_vec = word_vec.view(num_q, bs, 300).type_as(sp)
        query_embed_vec = self.query_embed_image(torch.cat([word_vec, sp], dim=-1))
        ###################################################################################
        for layer in self.layers[self.l_num:]:
            output1 = layer(output1, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_embed_vec)

            if self.return_intermediate:
                output_n = self.norm(output1)
                intermediate_sub_box.append(self.sub_bbox_embed(output_n).sigmoid())
                intermediate_obj_box.append(self.obj_bbox_embed(output_n).sigmoid())
                intermediate_verb_cls.append(self.verb_class_embed(output_n))
                intermediate_obj_cls.append(self.obj_class_embed(output_n))

        return torch.stack(intermediate_obj_box), torch.stack(intermediate_sub_box), \
               torch.stack(intermediate_verb_cls), torch.stack(intermediate_obj_cls)


class TransformerDecoderQMTS(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        #################################################################
        self.obj_class_embed = nn.Linear(256, 80 + 1)
        self.verb_class_embed = nn.Linear(256, 117)
        self.sub_bbox_embed = MLP(256, 256, 4, 3)
        self.obj_bbox_embed = MLP(256, 256, 4, 3)

        self.query_embed_sp = nn.Linear(8, 100)
        self.query_embed_image = nn.Linear(300 + 100, 256)
        import numpy as np
        self.coco_80 = torch.from_numpy(np.load('data/hico_20160224_det/annotations/coco_80.npy')).cuda()
        #################################################################

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        intermediate_sub_box = []
        intermediate_obj_box = []
        intermediate_verb_cls = []
        intermediate_obj_cls = []

        for layer in self.layers[0:self.num_layers // 2]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output = self.norm(output)
                intermediate_sub_box.append(self.sub_bbox_embed(output).sigmoid())
                intermediate_obj_box.append(self.obj_bbox_embed(output).sigmoid())
                intermediate_verb_cls.append(self.verb_class_embed(output))
                intermediate_obj_cls.append(self.obj_class_embed(output))

        ################################################################################
        sub_bbox = intermediate_sub_box[-1]
        obj_bbox = intermediate_obj_box[-1]
        obj_prob = F.softmax(intermediate_obj_cls[-1], -1)
        _, obj_labels = obj_prob[..., :-1].max(-1)

        sp = self.query_embed_sp(torch.cat([sub_bbox, obj_bbox], dim=-1))
        num_q, bs = obj_labels.size()
        obj_labels1 = obj_labels.view(-1)
        word_vec = self.coco_80[obj_labels1]
        word_vec = word_vec.view(num_q, bs, 300).type_as(sp)
        query_embed_vec = self.query_embed_image(torch.cat([word_vec, sp], dim=-1))

        ###################################################################################
        for layer in self.layers[self.num_layers // 2:]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_embed_vec)
            if self.return_intermediate:
                output = self.norm(output)
                intermediate.append(output)
                intermediate_sub_box.append(self.sub_bbox_embed(output).sigmoid())
                intermediate_obj_box.append(self.obj_bbox_embed(output).sigmoid())
                intermediate_verb_cls.append(self.verb_class_embed(output))
                intermediate_obj_cls.append(self.obj_class_embed(output))
        ###################################################################################
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate_sub_box.pop()
                intermediate_obj_box.pop()
                intermediate_verb_cls.pop()
                intermediate_obj_cls.pop()
                intermediate.pop()

                intermediate.append(output)
                intermediate_sub_box.append(self.sub_bbox_embed(output).sigmoid())
                intermediate_obj_box.append(self.obj_bbox_embed(output).sigmoid())
                intermediate_verb_cls.append(self.verb_class_embed(output))
                intermediate_obj_cls.append(self.obj_class_embed(output))

        if self.return_intermediate:
            return torch.stack(intermediate_obj_box), torch.stack(intermediate_sub_box), \
                   torch.stack(intermediate_verb_cls), torch.stack(intermediate_obj_cls), \
                   torch.stack(intermediate)

        return output

    def forwardT(self, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_word: Optional[Tensor] = None):

        sp = self.query_embed_sp(query_pos)
        query_embed_vec = self.query_embed_image(torch.cat([query_word, sp], dim=-1))
        output = torch.zeros_like(query_embed_vec)
        ##########################################################
        intermediate = []
        intermediate_sub_box = []
        intermediate_obj_box = []
        intermediate_verb_cls = []
        intermediate_obj_cls = []
        for layer in self.layers[self.num_layers // 2:]:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_embed_vec)
            if self.return_intermediate:
                output = self.norm(output)
                intermediate.append(output)
                intermediate_sub_box.append(self.sub_bbox_embed(output).sigmoid())
                intermediate_obj_box.append(self.obj_bbox_embed(output).sigmoid())
                intermediate_verb_cls.append(self.verb_class_embed(output))
                intermediate_obj_cls.append(self.obj_class_embed(output))
        ###################################################################################
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate_sub_box.pop()
                intermediate_obj_box.pop()
                intermediate_verb_cls.pop()
                intermediate_obj_cls.pop()
                intermediate.pop()

                intermediate.append(output)
                intermediate_sub_box.append(self.sub_bbox_embed(output).sigmoid())
                intermediate_obj_box.append(self.obj_bbox_embed(output).sigmoid())
                intermediate_verb_cls.append(self.verb_class_embed(output))
                intermediate_obj_cls.append(self.obj_class_embed(output))

        if self.return_intermediate:
            return torch.stack(intermediate_obj_box), torch.stack(intermediate_sub_box), \
                   torch.stack(intermediate_verb_cls), torch.stack(intermediate_obj_cls), \
                   torch.stack(intermediate)

        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderPos(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        self.sub_bbox_embed = MLP(256, 256, 4, 3)
        self.obj_bbox_embed = MLP(256, 256, 4, 3)
        self.sp = MLP(8, 100, 256, 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        intermediate_obj, intermediate_sub = [], []
        pos_feature = None

        for layer in self.layers:
            if pos_feature is None:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            else:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=pos_feature + query_pos)

            output = self.norm(output)
            outputs_sub_coord = self.sub_bbox_embed(output).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(output).sigmoid()

            outputs_coord = torch.cat([outputs_sub_coord, outputs_obj_coord], dim=-1)
            pos_feature = self.sp(outputs_coord)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_sub.append(outputs_sub_coord)
                intermediate_obj.append(outputs_obj_coord)

        if self.norm is not None:
            output = self.norm(output)
            outputs_sub_coord = self.sub_bbox_embed(output).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(output).sigmoid()
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                intermediate_sub.pop()
                intermediate_sub.append(outputs_sub_coord)

                intermediate_obj.pop()
                intermediate_obj.append(outputs_obj_coord)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_sub), \
                   torch.stack(intermediate_obj)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayerOcc(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, use_hard_mask=True)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)




class TransformerDecoderLayer1(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



class TransformerDecoderLayerAttMapOcc(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttentionEach(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                            key=self.with_pos_embed(memory, pos),
                                            value=memory, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayerAttMap(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                            key=self.with_pos_embed(memory, pos),
                                            value=memory, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayerAttMapSOcc(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     use_hard_mask: bool = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                            key=self.with_pos_embed(memory, pos),
                                            value=memory, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask,
                                            use_hard_mask=use_hard_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    use_hard_mask: bool = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   use_hard_mask=use_hard_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                use_hard_mask: bool = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, use_hard_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, use_hard_mask)


'''
class TransformerDecoderLayerAttMapOcc(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttentionOcc(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                            key=self.with_pos_embed(memory, pos),
                                            value=memory, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
'''

class GaussianMultiheadAttention(nn.MultiheadAttention):
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(GaussianMultiheadAttention, self).__init__(embed_dim, num_heads, dropout=dropout, bias=bias,
                                                         add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                                         kdim=kdim, vdim=vdim)
        self.gaussian = True

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, gaussian=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward_gaussian(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, gaussian=gaussian)
        else:
            return multi_head_attention_forward_gaussian(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, gaussian=gaussian)


class TransformerDecoderLayerMask(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = GaussianMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     gaussian=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   gaussian=gaussian)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                gaussian=None):
        if self.normalize_before:
            assert False
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, gaussian)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class InteractionLayer(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        self.d_feature = d_feature

        self.det_tfm = nn.Linear(d_model * 2, d_feature)
        self.rel_tfm = nn.Linear(d_model, d_feature)
        self.det_value_tfm = nn.Linear(d_model * 2, d_feature)

        self.rel_norm = nn.LayerNorm(d_model)

        if dropout is not None:
            self.dropout = dropout
            self.det_dropout = nn.Dropout(dropout)
            self.rel_add_dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, det_in, rel_in):
        det_attn_in = self.det_tfm(det_in)
        rel_attn_in = self.rel_tfm(rel_in)
        det_value = self.det_value_tfm(det_in)
        import math
        scores = torch.matmul(det_attn_in.transpose(0, 1),
                              rel_attn_in.permute(1, 2, 0)) / math.sqrt(self.d_feature)
        det_weight = F.softmax(scores.transpose(1, 2), dim=-1)
        if self.dropout is not None:
            det_weight = self.det_dropout(det_weight)
        rel_add = torch.matmul(det_weight, det_value.transpose(0, 1))
        rel_out = self.rel_add_dropout(rel_add) + rel_in.transpose(0, 1)
        rel_out = self.rel_norm(rel_out)

        return det_in, rel_out.transpose(0, 1)


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def build_transformer_occ(args):
    return TransformerOcc(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def build_transformer_mask(args):
    return TransformerMask(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_roi(args):
    return TransformerROI(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_cdn(args):
    return TransformerCDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_pos_guide(args):
    return TransformerPosGuide(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_guide(args):
    return TransformerGuide(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_add_verb_as_query(args):
    return TransformerAddVerbAsQ(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_grid_sample(args):
    return TransformerGS(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_pruning(args):
    return TransformerPruning(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_pruning_e2e(args):
    return TransformerPruningE2E(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_pruning_gumble_softmax(args):
    return TransformerPruningGumble(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_7l(args):
    return TransformerFFN7l(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn(args):
    return TransformerFFN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all(args):
    return TransformerFFNAll1l(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l(args):
    return TransformerFFNAll(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_rec(args, begin_l):
    return TransformerFFNAllRec(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l
    )


def build_transformer_ffn_2l_qg(args):
    return TransformerFFN2lQG(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l_obj_verb(args):
    return TransformerFFNAllOV(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l_rr(args):
    return TransformerFFNAllRR(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l_ts(args, begin_l):
    return TransformerFFN2lTs(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l
    )


def build_transformer_ffn_all_2l_e_mul(args):
    return TransformerFFNAllEMul(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l_ho_com(args):
    return TransformerFFN2LHOCom(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_part_sum(args):
    return TransformerPartSum(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l_3q(args):
    return TransformerFFN2l3Q(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


class TransformerDecoderTSQPosEobjAttHardSample(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 query_embed_q_h: Optional[Tensor] = None,
                 query_embed_e_h: Optional[Tensor] = None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_h, obj_boxes_h, verb_class_h, obj_class_h = [], [], [], []
        hs, hs_t, hs_hard = [], [], []
        atts_all, attt_all, atth_all = [], [], []

        for layer in self.layers:
            output, att_s = layer(output, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_embed_e
        output_hard = query_embed_e_h
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask_t,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_embed_q)
            output_hard, att_hard = layer(output_hard, memory, tgt_mask=tgt_mask_t,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask_t,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_embed_q_h)

            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

                output_n = self.norm(output_hard)
                atth_all.append(att_hard)
                hs_hard.append(output_n)
                sub_boxes_h.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_h.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_h.append(self.verb_class_embed(output_n))
                obj_class_h.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(sub_boxes_h), torch.stack(obj_boxes_h), torch.stack(obj_class_h), torch.stack(verb_class_h), \
        torch.stack(atts_all), torch.stack(attt_all), torch.stack(atth_all), \
               torch.stack(hs), torch.stack(hs_t), torch.stack(hs_hard)


class TransformerDecoderLayerAttMapHardm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     use_hard_mask: bool = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                            key=self.with_pos_embed(memory, pos),
                                            value=memory, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask,
                                            use_hard_mask=use_hard_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    use_hard_mask: bool = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   use_hard_mask=use_hard_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                use_hard_mask: bool = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, use_hard_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, use_hard_mask)


class TransformerDecoderTSQPosEobjAttHardm(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 use_learnable_embedding = False):
        # assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        atts_all, attt_all = [], []
        if use_learnable_embedding:
            output_tts = []
        for layer in self.layers:
            if use_learnable_embedding:
                output_tts.append(output)
            output, att_s = layer(output, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            if use_learnable_embedding:
                output_tt = output_tts[i]
            else:
                output_tt = output_t
            output_t, att_t = layer(output_tt, memory, tgt_mask=tgt_mask_t,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask_t,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_embed_q, use_hard_mask=True)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(atts_all), torch.stack(attt_all), \
               torch.stack(hs), torch.stack(hs_t)







class HOITransformerTSQPOSEOBJAttHardm(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardm(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttHardm(decoder_layer, num_decoder_layers, decoder_norm,
                                                            return_intermediate=return_intermediate_dec,
                                                            begin_l=begin_l,
                                                            num_obj_classes=num_obj_classes,
                                                            num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_s, att_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)



class TransformerDecoderTSQPosEobjAttHardmAttnES(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                 query_embed_sh: Optional[Tensor] = None,
                 query_sh: Optional[Tensor] = None
                 ):
        # assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        sub_boxes_sh, obj_boxes_sh, verb_class_sh, obj_class_sh = [], [], [], []
        hs, hs_t, hs_sh = [], [], []
        atts_all, attt_all, attsh_all = [], [], []
        att_select_all = []

        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos)



            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
                att_select_all.append(att_select)

        output_t = query_embed_e
        output_shift = query_embed_sh

        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]

            output_shift, att_shift, _ = layer(output_shift, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask_t,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_sh)

            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q,
                                       attn_select=att_select_all[i], use_hard_mask=True)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

                output_shift = self.norm(output_shift)
                attsh_all.append(att_shift)
                hs_sh.append(output_shift)
                sub_boxes_sh.append(self.sub_bbox_embed(output_shift).sigmoid())
                obj_boxes_sh.append(self.obj_bbox_embed(output_shift).sigmoid())
                verb_class_sh.append(self.verb_class_embed(output_shift))
                obj_class_sh.append(self.obj_class_embed(output_shift))


        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(sub_boxes_sh), torch.stack(obj_boxes_sh), torch.stack(obj_class_sh), torch.stack(verb_class_sh), \
               torch.stack(atts_all), torch.stack(attt_all), torch.stack(attsh_all), \
               torch.stack(hs), torch.stack(hs_t),  torch.stack(hs_sh)


class TransformerDecoderTSQPosEobjAttHardmAttnE(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardocc(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None,
                att_select_all=None,
                att_select_shift_all=None):

        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        atts_all, attt_all = [], []

        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output, att_s, _ = layer(output, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     pos=pos, query_pos=query_pos, attn_select=att_select_all[i], use_hard_mask=False)
            output_t, att_t, _ = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask_t,
                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                 pos=pos, query_pos=query_embed_q,
                                                 attn_select=att_select_shift_all[i], use_hard_mask=False)

            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(atts_all), torch.stack(attt_all), \
               torch.stack(hs), torch.stack(hs_t)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_embed_q: Optional[Tensor] = None,
                 query_embed_e: Optional[Tensor] = None, occ_pass=False):
        # assert tgt_key_padding_mask_t is not None

        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []
        atts_all, attt_all = [], []
        att_select_all = []
        att_select_shift_all = []

        for layer in self.layers:
            output, att_s, att_select = layer(output, memory, tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              pos=pos, query_pos=query_pos, use_hard_mask=occ_pass)
            att_select_all.append(att_select)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        output_t = query_embed_e
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            output_t, att_t, att_select2 = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask_t,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       pos=pos, query_pos=query_embed_q, use_hard_mask=occ_pass)
            att_select_shift_all.append(att_select2)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(atts_all), torch.stack(attt_all), \
               torch.stack(hs), torch.stack(hs_t)


class TransformerDecoderLayerAttMapHardmAttnOccShift(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttentionEach(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     attn_select: Optional[Tensor] = None,
                     use_hard_mask: bool = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map, att_select = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                        key=self.with_pos_embed(memory, pos),
                                                        value=memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask,
                                                        attn_select=attn_select, use_hard_mask=use_hard_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map, att_select

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    attn_select: Optional[Tensor] = None,
                    use_hard_mask: bool = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   use_hard_mask=use_hard_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_select: Optional[Tensor] = None,
                use_hard_mask: bool = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask)


class TransformerDecoderLayerAttMapHardmAttnE(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttentionEach(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     attn_select: Optional[Tensor] = None,
                     use_hard_mask: bool = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map, att_select = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                        key=self.with_pos_embed(memory, pos),
                                                        value=memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask,
                                                        attn_select=attn_select, use_hard_mask=use_hard_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map, att_select

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    attn_select: Optional[Tensor] = None,
                    use_hard_mask: bool = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   use_hard_mask=use_hard_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_select: Optional[Tensor] = None,
                use_hard_mask: bool = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask)


class TransformerDecoderLayerAttMapHardmAttnET(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = HardMaskMultiheadAttentionEachT(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     attn_select: Optional[Tensor] = None,
                     use_hard_mask: bool = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map, att_select = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                        key=self.with_pos_embed(memory, pos),
                                                        value=memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask,
                                                        attn_select=attn_select, use_hard_mask=use_hard_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map, att_select

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    attn_select: Optional[Tensor] = None,
                    use_hard_mask: bool = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   use_hard_mask=use_hard_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_select: Optional[Tensor] = None,
                use_hard_mask: bool = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, attn_select, use_hard_mask)


class HOITransformerTSQPOSEOBJAttHardmAttnE(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnE(d_model, nhead, dim_feedforward,
                                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttHardmAttnE(decoder_layer, num_decoder_layers, decoder_norm,
                                                                 return_intermediate=return_intermediate_dec,
                                                                 begin_l=begin_l,
                                                                 num_obj_classes=num_obj_classes,
                                                                 num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_s, att_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJAttHardmAttnES(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnE(d_model, nhead, dim_feedforward,
                                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttHardmAttnES(decoder_layer, num_decoder_layers, decoder_norm,
                                                                 return_intermediate=return_intermediate_dec,
                                                                 begin_l=begin_l,
                                                                 num_obj_classes=num_obj_classes,
                                                                 num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None, query_shift=None,
                query_embed_shift=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)

        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t,  \
            sub_boxes_sh, obj_boxes_sh, obj_class_sh, verb_class_sh, att_s, att_t, att_sh, hs, hs_t, hs_sh = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      query_embed_sh=query_shift,
                                      query_sh=query_embed_shift
                                      )

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_sh.transpose(1, 2), obj_boxes_sh.transpose(1, 2), \
                   obj_class_sh.transpose(1, 2), verb_class_sh.transpose(1, 2), \
                   att_s, att_t, att_sh,  hs.transpose(1, 2), hs_t.transpose(1, 2), hs_sh.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


class HOITransformerTSQPOSEOBJAttHardmAttnET(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMapHardmAttnET(d_model, nhead, dim_feedforward,
                                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttHardmAttnE(decoder_layer, num_decoder_layers, decoder_norm,
                                                                 return_intermediate=return_intermediate_dec,
                                                                 begin_l=begin_l,
                                                                 num_obj_classes=num_obj_classes,
                                                                 num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, att_s, att_t, hs, hs_t = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)




class HOITransformerTSQPOSEOBJAttHardSample(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1,
                 num_obj_classes=80, num_verb_classes=117):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMap(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSQPosEobjAttHardSample(decoder_layer, num_decoder_layers, decoder_norm,
                                                       return_intermediate=return_intermediate_dec,
                                                       begin_l=begin_l,
                                                       num_obj_classes=num_obj_classes,
                                                       num_verb_classes=num_verb_classes)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None,
                query_embed_q_h=None, query_embed_e_h=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed_q is not None:
            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, sub_boxes_h, obj_boxes_h, obj_class_h, verb_class_h, \
            att_s, att_t, att_h, hs, hs_t, hs_h = \
                self.decoder.forwardt(tgt1, memory,
                                      memory_key_padding_mask=mask,
                                      tgt_key_padding_mask_t=query_embed2_mask,
                                      pos=pos_embed,
                                      query_pos=query_embed1,
                                      query_embed_q=query_embed_q,
                                      query_embed_e=query_embed_e,
                                      query_embed_q_h=query_embed_q_h, query_embed_e_h=query_embed_e_h)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   sub_boxes_h.transpose(1, 2), obj_boxes_h.transpose(1, 2), \
                   obj_class_h.transpose(1, 2), verb_class_h.transpose(1, 2), \
                   att_s, att_t, att_h, hs.transpose(1, 2), hs_t.transpose(1, 2), hs_h.transpose(1, 2),

        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


def build_hoi_transformer_ts_qpos_eobj_attention_map_hard_occ(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttHardm(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj_attention_map_hard_occ_one_flow(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttHardm(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )




def build_hoi_transformer_ts_qpos_eobj_attention_map_hard_sample(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttHardSample(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )

def build_transformer_ffn_all_2l_com(args):
    return TransformerFFN2LCom(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l_inter(args):
    return TransformerFFN2lInter(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_ffn_all_2l_iq(args):
    return TransformerFFN2liq(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


#######################TODO###############################
def build_transformer_qm(args):
    return TransformerQM(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_qm_ts(args):
    return TransformerQMTS(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_hoi_transformer_2q_all_share_c(args):
    return HOITransformer2QAllShareC(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_hoi_transformer_2q(args):
    return HOITransformer2Q(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_hoi_transformer_2q_all_share(args):
    return HOITransformer2QAllShare(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_hoi_transformer_ts(args, begin_l):
    return HOITransformerTS(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l
    )


def build_hoi_transformer_ts_not_share(args):
    return HOITransformerTSNotShare(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_hoi_transformer_ts_offset(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffset(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qposob(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPosObj(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_tcdn(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffsetTCDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJ(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj1(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJ1(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )



def build_hoi_transformer_learnable_shiftbbox_occ(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttOcc(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_learnable_shiftbbox_occ_check_occ(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttOccCheckOcc(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_learnable_shiftbbox_occ_occ_pass(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttOccPass(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_learnable_shiftbbox_occ(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttOcc(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_learnable_shiftbbox_occ_pass(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttOccPass(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_learnable_shiftbbox_occ_shiftocc(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttOccShiftOcc(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj_attention_map(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAtt(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj_attention_map_selfocc(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttSOcc(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj_attention_map_occ(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttOcc(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )



def build_hoi_transformer_ts_qpos_eobjverb(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJVerb(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj_withs(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJWithS(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_diff_q(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffsetDQ(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_image_level(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffsetImageLevel(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_qm(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffsetQM(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_smask(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffsetSMask(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_tmask(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffsetTMask(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_tmask1(args, begin_l, matcher, num_obj_classes, num_verb_classes):
    return HOITransformerTSOffsetTMask1(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_2t(args, begin_l, matcher,
                                       num_obj_classes, num_verb_classes):
    return HOITransformerTSOffset2T(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_offset_prune(args, begin_l, matcher):
    return HOITransformerTSOffsetPrune(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher
    )


def build_hoi_transformer_ts_obj_verb(args, begin_l, matcher):
    return HOITransformerTSObjVerb(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher
    )


def build_hoi_transformer_ts_qpos_eobj_attention_map_hardm_attn_each(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttHardmAttnE(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj_attention_map_hardm_attn_each_shiftbbox(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttHardmAttnES(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_ts_qpos_eobj_attention_map_hardm_attn_each_t(args, begin_l, num_obj_classes, num_verb_classes):
    return HOITransformerTSQPOSEOBJAttHardmAttnET(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        num_obj_classes=num_obj_classes,
        num_verb_classes=num_verb_classes
    )


def build_hoi_transformer_t(args):
    return HOITransformert(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_hoi_transformer_117q(args):
    return HOITransformer117Q(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


class TransformerDecoderTSOffsetRefine(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, begin_l=1,
                 matcher=None, add_decoder_num=6):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.begin_l = begin_l

        hidden_dim = 256
        num_obj_classes = 80
        num_verb_classes = 117
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.matcher = matcher

        self.query_embed_sp = nn.Linear(8, 100)
        self.query_embed_image = nn.Linear(400, hidden_dim)
        self.query_embed_sp_s = nn.Linear(8, 100)
        self.query_embed_image_s = nn.Linear(400, hidden_dim)

        self.add_decoder_num = add_decoder_num
        import numpy as np
        self.coco_80 = torch.from_numpy(np.load('data/hico_20160224_det/annotations/coco_80.npy')).cuda()

    def match_fun(self, sub_box_s, obj_box_s, verb_class_s, obj_class_s, target):
        out = {'pred_obj_logits': obj_class_s.transpose(0, 1), 'pred_verb_logits': verb_class_s.transpose(0, 1),
               'pred_sub_boxes': sub_box_s.transpose(0, 1), 'pred_obj_boxes': obj_box_s.transpose(0, 1)}
        indices = self.matcher(out, target)
        return indices

    def select(self, indices, query_pos_t, query_pos):
        select_q = torch.zeros_like(query_pos_t)
        for batch_idx, (i, j) in enumerate(indices):
            select_q[j, batch_idx, :] = query_pos[i, batch_idx, :]
        return select_q

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        sub_bbox = sub_boxes[-1]
        obj_bbox = obj_boxes[-1]
        obj_prob = F.softmax(obj_class[-1], -1)
        _, obj_labels = obj_prob[..., :-1].max(-1)

        sp = self.query_embed_sp_s(torch.cat([sub_bbox, obj_bbox], dim=-1))
        num_q, bs = obj_labels.size()
        obj_labels1 = obj_labels.view(-1)
        word_vec = self.coco_80[obj_labels1]
        word_vec = word_vec.view(num_q, bs, 300).type_as(sp)
        query_embed_vec = self.query_embed_image_s(torch.cat([word_vec, sp], dim=-1))
        output = output + query_embed_vec

        for i in range(len(self.layers) - self.add_decoder_num, len(self.layers)):
            layer = self.layers[i]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 tgt_mask_t: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask_t: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 query_pos_t: Optional[Tensor] = None,
                 target=None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        sub_boxes, obj_boxes, verb_class, obj_class = [], [], [], []
        sub_boxes_t, obj_boxes_t, verb_class_t, obj_class_t = [], [], [], []
        hs, hs_t = [], []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
        ####################################################################
        sub_bbox = sub_boxes[-1]
        obj_bbox = obj_boxes[-1]
        obj_prob = F.softmax(obj_class[-1], -1)
        _, obj_labels = obj_prob[..., :-1].max(-1)
        # todo detach() hs to 0  query_offset
        sp = self.query_embed_sp_s(torch.cat([sub_bbox, obj_bbox], dim=-1))
        num_q, bs = obj_labels.size()
        obj_labels1 = obj_labels.view(-1)
        word_vec = self.coco_80[obj_labels1]
        word_vec = word_vec.view(num_q, bs, 300).type_as(sp)
        query_embed_vec = self.query_embed_image_s(torch.cat([word_vec, sp], dim=-1))
        # todo withhout deteach
        output = output + query_embed_vec

        for i in range(len(self.layers) - self.add_decoder_num, len(self.layers)):
            layer = self.layers[i]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                sub_boxes.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class.append(self.verb_class_embed(output_n))
                obj_class.append(self.obj_class_embed(output_n))
        ####################################################################
        output_t = query_pos_t
        for i in range(self.begin_l, len(self.layers)):
            layer = self.layers[i]
            sub_box_s = sub_boxes[i]
            obj_box_s = obj_boxes[i]
            verb_class_s = verb_class[i]
            obj_class_s = obj_class[i]
            indices = self.match_fun(sub_box_s, obj_box_s, verb_class_s, obj_class_s, target)
            select_q = self.select(indices, query_pos_t, query_pos)

            output_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask_t,
                             memory_key_padding_mask=memory_key_padding_mask,
                             pos=pos, query_pos=select_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                hs_t.append(output_n)
                sub_boxes_t.append(self.sub_bbox_embed(output_n).sigmoid())
                obj_boxes_t.append(self.obj_bbox_embed(output_n).sigmoid())
                verb_class_t.append(self.verb_class_embed(output_n))
                obj_class_t.append(self.obj_class_embed(output_n))

        return torch.stack(sub_boxes), torch.stack(obj_boxes), torch.stack(obj_class), torch.stack(verb_class), \
               torch.stack(sub_boxes_t), torch.stack(obj_boxes_t), torch.stack(obj_class_t), torch.stack(verb_class_t), \
               torch.stack(hs), torch.stack(hs_t)


class HOITransformerTSOffsetRefine(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, begin_l=1, matcher=None,
                 add_decoder_num=6):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTSOffsetRefine(decoder_layer, num_decoder_layers, decoder_norm,
                                                        return_intermediate=return_intermediate_dec,
                                                        begin_l=begin_l, matcher=matcher,
                                                        add_decoder_num=add_decoder_num)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed2=None, query_embed2_mask=None,
                pos_embed=None, target=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = torch.zeros_like(query_embed1)
        if query_embed2 is not None:

            sub_boxes, obj_boxes, obj_class, verb_class, \
            sub_boxes_t, obj_boxes_t, obj_class_t, verb_class_t, \
            hs, hs_t = self.decoder.forwardt(tgt1, memory,
                                             memory_key_padding_mask=mask,
                                             tgt_key_padding_mask_t=query_embed2_mask,
                                             pos=pos_embed,
                                             query_pos=query_embed1,
                                             query_pos_t=query_embed2,
                                             target=target)

            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   sub_boxes_t.transpose(1, 2), obj_boxes_t.transpose(1, 2), \
                   obj_class_t.transpose(1, 2), verb_class_t.transpose(1, 2), \
                   hs.transpose(1, 2), hs_t.transpose(1, 2)
        else:
            sub_boxes, obj_boxes, obj_class, verb_class, hs = self.decoder(tgt1, memory,
                                                                           memory_key_padding_mask=mask,
                                                                           pos=pos_embed,
                                                                           query_pos=query_embed1)
            return sub_boxes.transpose(1, 2), obj_boxes.transpose(1, 2), \
                   obj_class.transpose(1, 2), verb_class.transpose(1, 2), \
                   hs.transpose(1, 2)


def build_hoi_transformer_ts_offset_refine(args, begin_l, matcher, add_decoder_num):
    return HOITransformerTSOffsetRefine(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        begin_l=begin_l,
        matcher=matcher,
        add_decoder_num=add_decoder_num
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
