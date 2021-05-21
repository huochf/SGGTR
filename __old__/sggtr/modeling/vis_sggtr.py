# Modified from DETR (https://github.com/facebookresearch/detr)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ---------------------------------------------------------------------------------------------
from typing import List, Optional
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

from .backbone import build_backbone
from .detector import Detector
from .detection_header import (
    build_obj_detection_header, 
    build_rel_detection_header,
    build_graph_detection_header,
    build_graph_with_exknowledge_detection_header
)
from . import registry
from .query_generator import (
    ObjectQueryGenerator, 
    DenseObjectQueryGenerator,
    GraphQueryGenerator,
)
from .losses import build_loss
from .postprocessor import build_postprocessor

from sggtr.structures.image_list import ImageList
from sggtr.structures.bounding_box import BoxList


class XXXTR(nn.Module):
    """
    This is the XXXTR module that performs object detection or relationship detection 
    or scene graph generation using Transformer.
    """
    def __init__(self, backbone, detector, loss, postprocess):
        """
        Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            detector: torch module of the transformer architecture, which take tensor list
                      as input and output object features or graph features. See ./detectors/*.py
            loss:
            postprocess: 
        """
        super().__init__()
        self.backbone = backbone
        self.detector = detector
        self.loss = loss
        self.postprocess = postprocess


    def forward(self, samples: ImageList, targets: List[BoxList] = None, visualize=True):
        image_sizes = [size[::-1] for size in samples.image_sizes] # from (h, w) to (w, h)
        features_list = self.backbone(samples)
        outputs = self.detector(features_list[-1])

        if self.training:
            loss_dict = self.loss(outputs, targets)
            results = None
            if visualize:
                results = self.postprocess(outputs, image_sizes)
            return loss_dict, results, outputs
        else:
            return self.postprocess(outputs, image_sizes), outputs


def build_model(cfg):
    backbone = build_backbone(cfg)
    detector = registry.DETECTOR[cfg.MODEL.DETECTOR.TYPE](cfg)
    loss = build_loss(cfg)
    postprocessor = build_postprocessor(cfg)
    model = XXXTR(backbone, detector, loss, postprocessor)

    if cfg.MODEL.PRETRAINED:
        print("Loading pretrained params from " + cfg.MODEL.PRETRAINED)
        state_dict = torch.load(cfg.MODEL.PRETRAINED)
        model.load_state_dict(state_dict, strict=False)
    return model


def build_scene_graph_sgg_only_module(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    enc_layers = cfg.MODEL.DETECTOR.ENC_LAYERS
    dec_layers = cfg.MODEL.DETECTOR.DEC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE
    num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY 
    num_hidden_rel_query = cfg.MODEL.DETECTOR.NUM_HIDDEN_REL_QUERY

    return SceneGraphSGGOnlyModule(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_obj_query=num_obj_query, 
        num_hidden_rel=num_hidden_rel_query,
        num_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate=True,
    )


@registry.DETECTOR.register("SGGTRSGGOnly")
class SGGTRSGGOnly(Detector):

    def __init__(self, cfg):
        super(SGGTRSGGOnly, self).__init__()
        self.query_generator = GraphQueryGenerator(cfg)
        self.query_pos_generator = GraphQueryGenerator(cfg)
        self.transformer = build_scene_graph_sgg_only_module(cfg)
        self.graph_detector_header = build_graph_detection_header(cfg)


    def forward(self, tensor_list):
        graph_query = self.query_generator()
        graph_query_embed = self.query_pos_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        graph_features, attn_weights_list = self.transformer(features, feature_mask, feature_pos, graph_query, graph_query_embed)

        graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits = self.graph_detector_header(graph_features)

        return {"graph_obj_cls_logits": graph_obj_cls_logits, 
                "graph_obj_box_coord": graph_obj_box_coord,
                "graph_rel_cls_logits": graph_rel_cls_logits,
                "attn_weights_list": attn_weights_list}


class SceneGraphSGGOnlyModule(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate=False):
        super().__init__()

        scene_graph_layer = SceneGraphModuleLayer(d_model, nhead, dim_feedforward, num_obj_query, num_hidden_rel,
                                                  dropout, activation, normalize_before)
        self.scene_graph_layers = _get_clones(scene_graph_layer, num_layers)
        self.num_layers = num_layers
        # self.feature_norm = nn.LayerNorm(d_model)
        self.graph_norm = nn.LayerNorm(d_model)

        # self.feature_proj = nn.Linear(d_model * 2, d_model)

        self.return_intermediate = return_intermediate


    def forward(self, features, feature_mask, feature_pos, graph_query, graph_query_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)
        feature_pos = feature_pos.flatten(2).permute(2, 0, 1)
        feature_mask = feature_mask.flatten(1)
        graph_query_embed = graph_query_embed.unsqueeze(1).repeat(1, bs, 1)
        # graph_queries = graph_query.unsqueeze(1).repeat(1, bs, 1)
        graph_queries = torch.zeros_like(graph_query_embed)

        graph_features = graph_queries
        graph_feature_all_layer = []
        reference_point_features = features
        all_attn_weight = []
        for layer in self.scene_graph_layers:
            # Is this line below necessary ?
            # reference_point_features = self.feature_proj(torch.cat((reference_point_features, features), -1))
            reference_point_features, graph_features, attn_weight_dict = layer(graph_features, reference_point_features, None, feature_mask, feature_pos, graph_query_embed)
            if self.return_intermediate:
                graph_feature_all_layer.append(self.graph_norm(graph_features))
                all_attn_weight.append(attn_weight_dict)

        if self.return_intermediate:
            return torch.stack(graph_feature_all_layer).transpose(1, 2), all_attn_weight

        return self.graph_norm(graph_features).transpose(0, 1), [attn_weight_dict]


class SceneGraphModuleLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.graph_project_layer = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.graph_reasoning_layer = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.graph_reproject_layer = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)

        self.ffn1 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)
        self.ffn2 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)
        self.ffn3 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)
        self.ffn4 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)

        self.feature_proj = nn.Linear(d_model * 2, d_model)

        self.relation_project = RelationProjectLayer(d_model, nhead, num_obj_query, num_hidden_rel, dropout, activation, normalize_before)
        self.relation_project2 = RelationProjectLayer(d_model, nhead, num_obj_query, num_hidden_rel, dropout, activation, normalize_before)


    def forward(self, graph_queries, features,
                feature_mask: Optional[Tensor] = None,
                feature_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        original_queries, original_query_pos = graph_queries, query_pos

        queries, query_pos, hidden_to_query_attn = self.relation_project(True, graph_queries, original_query_pos)
        graph_node_features, feature_to_query_attn = self.graph_project_layer(queries, features, feature_mask, feature_key_padding_mask, pos, query_pos)
        graph_node_features = self.ffn1(graph_node_features)

        graph_node_features, query_to_query_attn = self.graph_reasoning_layer(graph_node_features, pos=query_pos)
        # graph_node_features = self.relation_project(False, graph_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn2(graph_node_features)

        # _graph_node_features, query_pos = self.relation_project2(True, graph_node_features, original_query_pos)
        refined_features, query_to_feature_attn = self.graph_reproject_layer(features, graph_node_features, None, None, query_pos, pos)
        refined_features = self.ffn3(refined_features)

        graph_node_features, query_to_hidden_attn = self.relation_project(False, graph_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn4(graph_node_features)

        features = refined_features
        # TODO: If this line below is necessary? 
        # cat better than add??
        # features = self.feature_proj(torch.cat((features, refined_features), -1))
        # features = features + refined_features
        # features = refined_features

        return features, graph_node_features, \
                { 'hidden_to_query_attn': hidden_to_query_attn,
                  'feature_to_query_attn': feature_to_query_attn,
                  'query_to_query_attn': query_to_query_attn,
                  'query_to_feature_attn': query_to_feature_attn,
                  'query_to_hidden_attn': query_to_hidden_attn,
                }


class RelationProjectLayer(nn.Module):

    def __init__(self, d_model, nhead=8, num_obj_query=100, num_hidden_rel=100, 
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.num_obj_query = num_obj_query
        self.num_hidden_rel = num_hidden_rel
        # TODO: Should we initial query with zeros vectors or learned vectors with prior?
        self.rel_hidden_queries = nn.Embedding(num_hidden_rel, d_model)
        self.rel_hidden_pos = nn.Embedding(num_hidden_rel, d_model)
        self.rel_input_to_hidden = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.rel_hidden_to_input = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)


    def split_into_obj_rel(self, tensor):
        return tensor[:self.num_obj_query, :], tensor[self.num_obj_query:, :]


    def forward_project(self, graph_queries, query_pos):
        obj_queries, rel_queries = self.split_into_obj_rel(graph_queries)
        obj_query_pos, rel_query_pos = self.split_into_obj_rel(query_pos)
        b = graph_queries.shape[1]
        rel_hidden_queries = self.rel_hidden_queries.weight.unsqueeze(1).repeat(1, b, 1)
        rel_hidden_pos = self.rel_hidden_pos.weight.unsqueeze(1).repeat(1, b, 1)

        rel_hidden_queries, attn_weight = self.rel_input_to_hidden(tgt=rel_hidden_queries,
                                                      memory=rel_queries,
                                                      pos=rel_query_pos,
                                                      query_pos=rel_hidden_pos)
        queries = torch.cat((obj_queries, rel_hidden_queries), dim=0)
        query_pos = torch.cat((obj_query_pos, rel_hidden_pos), dim=0)
        
        return queries, query_pos, attn_weight


    def forward_reproject(self, original_queries, original_query_pos, graph_node_features, ):
        obj_queries, rel_queries = self.split_into_obj_rel(original_queries)
        obj_query_pos, rel_query_pos = self.split_into_obj_rel(original_query_pos)
        obj_node_features, hidden_rel_node_features = self.split_into_obj_rel(graph_node_features)

        b = original_queries.shape[1]
        rel_hidden_pos = self.rel_hidden_pos.weight.unsqueeze(1).repeat(1, b, 1)

        rel_node_features, attn_weight = self.rel_hidden_to_input(tgt=rel_queries,
                                                     memory=hidden_rel_node_features,
                                                     pos=rel_hidden_pos,
                                                     query_pos=rel_query_pos)
        # skip connection is necessary here?
        rel_node_features = rel_queries + rel_node_features
        graph_node_features = torch.cat((obj_node_features, rel_node_features), 0)

        return graph_node_features, attn_weight


    def forward(self, is_project, *args):
        if is_project:
            return self.forward_project(*args)
        else:
            return self.forward_reproject(*args)

        raise NotImplementedError("please use 'forward_project' or 'forward_reproject'")


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_pre(self, src, 
                    src_mask: Optional[Tensor] = None, 
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        residual = src
        src = self.norm(src)
        q = k = self.with_pos_embed(src, pos)
        src, attn_weight = self.self_attention(q, k, value=src, attn_mask=src_mask, 
                                  key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout(src)

        return src, attn_weight


    def forward_post(self, src, 
                     src_mask: Optional[Tensor] = None, 
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        residual = src
        q = k = self.with_pos_embed(src, pos)
        src, attn_weight = self.self_attention(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout(src)
        src = self.norm(src)

        return src, attn_weight


    def forward(self, src, 
                src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask,src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        residual = tgt
        tgt = self.norm(tgt)
        q, k = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        tgt, attn_weight = self.cross_attention(q, k, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = residual + self.dropout(tgt)
        return tgt, attn_weight


    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        residual = tgt
        q, k = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        tgt, attn_weight = self.cross_attention(q, k, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = residual + self.dropout(tgt)
        tgt = self.norm(tgt)
        return tgt, attn_weight


    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of FeedForward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def forward_pre(self, src):
        residual = src
        src = self.norm(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout1(src)
        return src


    def forward_post(self, src):
        residual = src
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout1(src)
        src = self.norm(src)
        return src


    def forward(self, src):
        if self.normalize_before:
            return self.forward_pre(src)
        return self.forward_post(src)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


