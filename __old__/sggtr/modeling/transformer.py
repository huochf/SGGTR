# Modified from DETR (https://github.com/facebookresearch/detr)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ---------------------------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import pickle

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = None
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
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)[0]
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class EncoderOnlyTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, 
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, 
                                          return_intermediate=return_intermediate)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        hs = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return hs.transpose(1, 2)


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, mask, pos, queries, query_embed):
        # flatten NxCxHxW to HWxBxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        queries = queries.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        hs = self.decoder(queries, src, memory_key_padding_mask=mask,
                          pos=pos, query_pos=query_embed)
        return hs.transpose(1, 2)


class DETRGraphModule(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100, dropout=0.1, 
                 activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        scene_graph_decoder_layer = SceneGraphDecoderLayer(d_model, nhead, dim_feedforward, num_obj_query, num_hidden_rel, 
                                                           dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = SceneGraphDecoder(scene_graph_decoder_layer, num_decoder_layers, decoder_norm,
                                         return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, mask, pos_embed, query_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)[0]
        hs = self.decoder(tgt, memory, mask, pos_embed, query_embed)

        return hs.transpose(1, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
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

        return output.unsqueeze(0)


class SceneGraphDecoder(nn.Module):

    def __init__(self, scene_graph_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.scene_graph_layers = _get_clones(scene_graph_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate


    def forward(self, graph_queries, features, feature_mask, feature_pos, query_pos):
        graph_node_features = graph_queries

        graph_node_features_all_layers = []

        for scene_graph_layer in self.scene_graph_layers:
            graph_node_features = scene_graph_layer(graph_node_features, features, None, feature_mask, feature_pos, query_pos)

            if self.return_intermediate:
                graph_node_features_all_layers.append(self.norm(graph_node_features))

        if self.return_intermediate:
            return torch.stack(graph_node_features_all_layers)

        if self.norm is not None:
            graph_node_features = self.norm(graph_node_features)

        return graph_node_features.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.ffn = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)


    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src = self.self_attn(src, src_mask, src_key_padding_mask, pos)
        src = self.ffn(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.ffn = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt = self.self_attn(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        tgt = self.cross_attn(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        tgt = self.ffn(tgt)

        return tgt


class SceneGraphDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100, 
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.ffn = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)

        self.relation_project = RelationProjectLayer(d_model, nhead, num_obj_query, num_hidden_rel, dropout, activation, normalize_before)


    def forward(self, graph_queries, features,
                feature_mask: Optional[Tensor] = None,
                feature_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        original_queries, original_query_pos = graph_queries, query_pos

        queries, query_pos = self.relation_project.forward_project(graph_queries, query_pos)

        queries = self.self_attn(queries, pos=query_pos)
        graph_node_features = self.cross_attn(queries, features, feature_mask, feature_key_padding_mask, pos, query_pos)

        graph_node_features = self.relation_project.forward_reproject(original_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn(graph_node_features)

        return graph_node_features


class SceneGraphModule(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate=False):
        super().__init__()

        scene_graph_layer = SceneGraphModuleLayer(d_model, nhead, dim_feedforward, num_obj_query, num_hidden_rel,
                                                  dropout, activation, normalize_before)
        self.scene_graph_layers = _get_clones(scene_graph_layer, num_layers)
        self.num_layers = num_layers
        self.feature_norm = nn.LayerNorm(d_model)
        self.graph_norm = nn.LayerNorm(d_model)

        self.feature_proj = nn.Linear(d_model * 2, d_model)

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
        feature_all_layer = []
        graph_feature_all_layer = []
        reference_point_features = features
        for layer in self.scene_graph_layers:
            # Is this line below necessary ?
            # reference_point_features = self.feature_proj(torch.cat((reference_point_features, features), -1))
            reference_point_features, graph_features = layer(graph_features, reference_point_features, None, feature_mask, feature_pos, graph_query_embed)
            if self.return_intermediate:
                feature_all_layer.append(self.feature_norm(reference_point_features))
                graph_feature_all_layer.append(self.graph_norm(graph_features))

        if self.return_intermediate:
            return torch.stack(feature_all_layer).transpose(1, 2), torch.stack(graph_feature_all_layer).transpose(1, 2)

        return self.feature_norm(features).transpose(0, 1), self.graph_norm(graph_features).transpose(0, 1)


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
        for layer in self.scene_graph_layers:
            # Is this line below necessary ?
            # reference_point_features = self.feature_proj(torch.cat((reference_point_features, features), -1))
            reference_point_features, graph_features = layer(graph_features, reference_point_features, None, feature_mask, feature_pos, graph_query_embed)
            if self.return_intermediate:
                graph_feature_all_layer.append(self.graph_norm(graph_features))

        if self.return_intermediate:
            return torch.stack(graph_feature_all_layer).transpose(1, 2)

        return self.graph_norm(graph_features).transpose(0, 1)


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

        queries, query_pos = self.relation_project(True, graph_queries, original_query_pos)
        graph_node_features = self.graph_project_layer(queries, features, feature_mask, feature_key_padding_mask, pos, query_pos)
        graph_node_features = self.ffn1(graph_node_features)

        graph_node_features = self.graph_reasoning_layer(graph_node_features, pos=query_pos)
        # graph_node_features = self.relation_project(False, graph_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn2(graph_node_features)

        # _graph_node_features, query_pos = self.relation_project2(True, graph_node_features, original_query_pos)
        refined_features = self.graph_reproject_layer(features, graph_node_features, None, None, query_pos, pos)
        refined_features = self.ffn3(refined_features)

        graph_node_features = self.relation_project(False, graph_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn4(graph_node_features)

        features = refined_features
        # TODO: If this line below is necessary? 
        # cat better than add??
        # features = self.feature_proj(torch.cat((features, refined_features), -1))
        # features = features + refined_features
        # features = refined_features

        return features, graph_node_features


class SGGTRWithEXKnowledge(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100,
                 word_embedding=None, correlation=None,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate=False):
        super().__init__()

        scene_graph_layer = SceneGraphWithEXKnowledge(d_model, nhead, dim_feedforward, num_obj_query, num_hidden_rel,
                                                      word_embedding, correlation,
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
        for layer in self.scene_graph_layers:
            # Is this line below necessary ?
            # reference_point_features = self.feature_proj(torch.cat((reference_point_features, features), -1))
            reference_point_features, graph_features = layer(graph_features, reference_point_features, None, feature_mask, feature_pos, graph_query_embed)
            if self.return_intermediate:
                graph_feature_all_layer.append(self.graph_norm(graph_features))

        if self.return_intermediate:
            return torch.stack(graph_feature_all_layer).transpose(1, 2)

        return self.graph_norm(graph_features).transpose(0, 1)



class SceneGraphWithEXKnowledge(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100,
                 word_embedding=None, correlation=None,
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

        self.incorporate_exknowledge_layer = EXKnowledgeLayer(d_model, nhead, num_obj_query, num_hidden_rel, word_embedding, correlation, dropout, activation, normalize_before)


    def forward(self, graph_queries, features,
                feature_mask: Optional[Tensor] = None,
                feature_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        original_queries, original_query_pos = graph_queries, query_pos

        queries, query_pos = self.relation_project(True, graph_queries, original_query_pos)
        graph_node_features = self.graph_project_layer(queries, features, feature_mask, feature_key_padding_mask, pos, query_pos)
        graph_node_features = self.ffn1(graph_node_features)

        graph_node_features = self.incorporate_exknowledge_layer(graph_node_features, query_pos)

        graph_node_features = self.graph_reasoning_layer(graph_node_features, pos=query_pos)
        # graph_node_features = self.relation_project(False, graph_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn2(graph_node_features)

        # _graph_node_features, query_pos = self.relation_project2(True, graph_node_features, original_query_pos)
        refined_features = self.graph_reproject_layer(features, graph_node_features, None, None, query_pos, pos)
        refined_features = self.ffn3(refined_features)

        graph_node_features = self.relation_project(False, graph_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn4(graph_node_features)

        features = refined_features

        return features, graph_node_features


class EXKnowledgeLayer(nn.Module):

    def __init__(self, d_model, nhead, num_obj_query, num_hidden_rel, word_embedding, correlation, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()

        self.num_obj_query = num_obj_query
        self.num_hidden_rel = num_hidden_rel

        self.obj_semantic_features = nn.Parameter(torch.from_numpy(word_embedding['obj_features']), requires_grad=False) # [151, 300]
        self.rel_semantic_features = nn.Parameter(torch.from_numpy(word_embedding['rel_features']), requires_grad=False) # [51,  300]
        self.semantic_proj = nn.Linear(300, d_model)
        self.semantic_pos_proj = nn.Linear(300, d_model)
        self.obj2obj = nn.Parameter(torch.tensor(correlation['obj2obj'][:8], dtype=torch.bool), requires_grad=False) # [9, 151, 151]
        self.rel2rel = nn.Parameter(torch.tensor(correlation['rel2rel'], dtype=torch.bool), requires_grad=False) # [4,  51,  51]
        self.obj2rel = nn.Parameter(torch.tensor(correlation['obj2rel'][:2], dtype=torch.bool), requires_grad=False) # [3, 151,  51]
        self.rel2obj = nn.Parameter(torch.tensor(correlation['rel2obj'][:2], dtype=torch.bool), requires_grad=False) # [3,  51, 151]

        self.visual2semantic = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.semantic2visual = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)

        self.obj2obj_attn = CrossAttentionLayer(d_model, 8, dropout, activation, normalize_before)
        self.rel2obj_attn = CrossAttentionLayer(d_model, 2, dropout, activation, normalize_before)
        self.obj2rel_attn = CrossAttentionLayer(d_model, 2, dropout, activation, normalize_before)
        self.rel2rel_attn = CrossAttentionLayer(d_model, 4, dropout, activation, normalize_before)


    def forward(self, graph_node_features, graph_pos):
        bs = graph_node_features.shape[1]
        obj_node_features, rel_node_features = graph_node_features[:self.num_obj_query, :], graph_node_features[self.num_obj_query:, :]
        obj_node_pos, rel_node_pos = graph_pos[:self.num_obj_query, :], graph_pos[self.num_obj_query:, :]

        obj_semantic_features = self.semantic_proj(self.obj_semantic_features).unsqueeze(1).repeat(1, bs, 1)
        rel_semantic_features = self.semantic_proj(self.rel_semantic_features).unsqueeze(1).repeat(1, bs, 1)
        obj_semantic_pos = self.semantic_pos_proj(self.obj_semantic_features).unsqueeze(1).repeat(1, bs, 1)
        rel_semantic_pos = self.semantic_pos_proj(self.rel_semantic_features).unsqueeze(1).repeat(1, bs, 1)

        obj_vis_sem_features = self.visual2semantic(obj_semantic_features, obj_node_features, None, None, obj_node_pos, obj_semantic_pos)
        rel_vis_sem_features = self.visual2semantic(rel_semantic_features, rel_node_features, None, None, rel_node_pos, rel_semantic_pos)

        obj2obj_mask = (~self.obj2obj).unsqueeze(0).repeat(bs, 1, 1, 1).view(bs * 8, 151, 151)
        rel2rel_mask = (~self.rel2rel).unsqueeze(0).repeat(bs, 1, 1, 1).view(bs * 4, 51, 51)
        obj2rel_mask = (~self.obj2rel).unsqueeze(0).repeat(bs, 1, 1, 1).view(bs * 2, 151, 51)
        rel2obj_mask = (~self.rel2obj).unsqueeze(0).repeat(bs, 1, 1, 1).view(bs * 2, 51, 151)

        obj2obj_mask[:, 0, :] = 0
        rel2rel_mask[:, 0, :] = 0
        obj2rel_mask[:, 0, :] = 0
        rel2obj_mask[:, 0, :] = 0

        obj2obj_mask[:, :, 0] = 0
        rel2rel_mask[:, :, 0] = 0
        obj2rel_mask[:, :, 0] = 0
        rel2obj_mask[:, :, 0] = 0

        obj2obj_features = self.obj2obj_attn(obj_vis_sem_features, obj_vis_sem_features, obj2obj_mask, None, obj_semantic_pos, obj_semantic_pos)
        obj2rel_features = self.obj2rel_attn(obj_vis_sem_features, rel_vis_sem_features, obj2rel_mask, None, rel_semantic_pos, obj_semantic_pos)
        rel2obj_features = self.rel2obj_attn(rel_vis_sem_features, obj_vis_sem_features, rel2obj_mask, None, obj_semantic_pos, rel_semantic_pos)
        rel2rel_features = self.rel2rel_attn(rel_vis_sem_features, rel_vis_sem_features, rel2rel_mask, None, rel_semantic_pos, rel_semantic_pos)

        obj_refined_features = obj2obj_features + obj2rel_features
        rel_refined_features = rel2obj_features + rel2rel_features

        # print(obj_refined_features, force=True)

        # obj_refined_features = obj_vis_sem_features
        # rel_refined_features = rel_vis_sem_features

        obj_node_refined_features = self.semantic2visual(obj_node_features, obj_refined_features, None, None, obj_semantic_pos, obj_node_pos)
        rel_node_refined_features = self.semantic2visual(rel_node_features, rel_refined_features, None, None, rel_semantic_pos, rel_node_pos)

        return torch.cat((obj_node_refined_features, rel_node_refined_features))


class SGGDEModule(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate=False):
        super().__init__()

        scene_graph_layer = SGGDEModuleLayer(d_model, nhead, dim_feedforward, num_obj_query, num_hidden_rel,
                                                  dropout, activation, normalize_before)
        self.scene_graph_layers = _get_clones(scene_graph_layer, num_layers)
        self.num_layers = num_layers
        self.feature_norm = nn.LayerNorm(d_model)
        self.graph_norm = nn.LayerNorm(d_model)

        self.return_intermediate = return_intermediate


    def forward(self, features, feature_mask, feature_pos, graph_query_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)
        feature_pos = feature_pos.flatten(2).permute(2, 0, 1)
        feature_mask = feature_mask.flatten(1)
        graph_query_embed = graph_query_embed.unsqueeze(1).repeat(1, bs, 1)
        graph_queries = torch.zeros_like(graph_query_embed)

        graph_features = graph_queries
        feature_all_layer = []
        graph_feature_all_layer = []
        for layer in self.scene_graph_layers:
            features, graph_features = layer(graph_features, features, None, feature_mask, feature_pos, graph_query_embed)
            if self.return_intermediate:
                feature_all_layer.append(self.feature_norm(features))
                graph_feature_all_layer.append(self.graph_norm(graph_features))

        if self.return_intermediate:
            return torch.stack(feature_all_layer).transpose(1, 2), torch.stack(graph_feature_all_layer).transpose(1, 2)

        return self.feature_norm(features).transpose(0, 1), self.graph_norm(graph_features).transpose(0, 1)


class SGGDEModuleLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, num_obj_query=100, num_hidden_rel=100,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.feature_refine_layer = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.graph_project_layer = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.graph_reasoning_layer = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.graph_reproject_layer = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)

        self.ffn = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)
        self.ffn1 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)
        self.ffn2 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)
        self.ffn3 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation, normalize_before)

        self.relation_project = RelationProjectLayer(d_model, nhead, num_obj_query, num_hidden_rel, dropout, activation, normalize_before)
        self.relation_project2 = RelationProjectLayer(d_model, nhead, num_obj_query, num_hidden_rel, dropout, activation, normalize_before)


    def forward(self, graph_queries, features,
                feature_mask: Optional[Tensor] = None,
                feature_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        original_queries, original_query_pos = graph_queries, query_pos

        queries, query_pos = self.relation_project.forward_project(graph_queries, query_pos)

        graph_node_features = self.graph_project_layer(queries, features, feature_mask, feature_key_padding_mask, pos, query_pos)
        graph_node_features = self.ffn1(graph_node_features)

        graph_node_features = self.graph_reasoning_layer(graph_node_features, None, None, query_pos)
        graph_node_features = self.relation_project.forward_reproject(original_queries, original_query_pos, graph_node_features)
        graph_node_features = self.ffn2(graph_node_features)

        _graph_node_features, query_pos = self.relation_project2.forward_project(graph_node_features, original_query_pos)
        refined_features = self.graph_reproject_layer(features, _graph_node_features, None, None, query_pos, pos)
        refined_features = self.ffn3(refined_features)

        features = self.feature_refine_layer(features, feature_mask, feature_key_padding_mask, pos)
        features = self.ffn(features)
        features = features + refined_features

        return features, graph_node_features


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

        rel_hidden_queries = self.rel_input_to_hidden(tgt=rel_hidden_queries,
                                                      memory=rel_queries,
                                                      pos=rel_query_pos,
                                                      query_pos=rel_hidden_pos)
        queries = torch.cat((obj_queries, rel_hidden_queries), dim=0)
        query_pos = torch.cat((obj_query_pos, rel_hidden_pos), dim=0)
        
        return queries, query_pos


    def forward_reproject(self, original_queries, original_query_pos, graph_node_features, ):
        obj_queries, rel_queries = self.split_into_obj_rel(original_queries)
        obj_query_pos, rel_query_pos = self.split_into_obj_rel(original_query_pos)
        obj_node_features, hidden_rel_node_features = self.split_into_obj_rel(graph_node_features)

        b = original_queries.shape[1]
        rel_hidden_pos = self.rel_hidden_pos.weight.unsqueeze(1).repeat(1, b, 1)

        rel_node_features = self.rel_hidden_to_input(tgt=rel_queries,
                                                     memory=hidden_rel_node_features,
                                                     pos=rel_hidden_pos,
                                                     query_pos=rel_query_pos)
        # skip connection is necessary here?
        rel_node_features = rel_queries + rel_node_features
        graph_node_features = torch.cat((obj_node_features, rel_node_features), 0)

        return graph_node_features


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
        src = self.self_attention(q, k, value=src, attn_mask=src_mask, 
                                  key_padding_mask=src_key_padding_mask)[0]
        src = residual + self.dropout(src)

        return src


    def forward_post(self, src, 
                     src_mask: Optional[Tensor] = None, 
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        residual = src
        q = k = self.with_pos_embed(src, pos)
        src = self.self_attention(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = residual + self.dropout(src)
        src = self.norm(src)

        return src


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
        tgt = self.cross_attention(q, k, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = residual + self.dropout(tgt)
        return tgt


    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        residual = tgt
        q, k = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        tgt = self.cross_attention(q, k, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = residual + self.dropout(tgt)
        tgt = self.norm(tgt)
        return tgt


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


def build_scene_graph_decoder(cfg):
    d_model = cfg.MODEL.SGGTR.D_MODEL
    nhead = cfg.MODEL.SGGTR.NHEAD
    dim_feedforward = cfg.MODEL.SGGTR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.SGGTR.DROPOUT
    num_layers = cfg.MODEL.SGGTR.NUM_LAYERS

    scene_graph_layer = SceneGraphDecoderLayer(d_model, nhead, dim_feedforward, dropout)
    self_refine_layer = SelfAttentionLayer(d_model, nhead, dim_feedforward, dropout)
    reproject_layer = CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
    decoder = SceneGraphDecoder(scene_graph_layer, self_refine_layer, reproject_layer, num_layers)

    return decoder


def build_transformer(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    enc_layers = cfg.MODEL.DETECTOR.ENC_LAYERS
    dec_layers = cfg.MODEL.DETECTOR.DEC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE

    return Transformer(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )


def build_encoder_only_transformer(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    enc_layers = cfg.MODEL.DETECTOR.ENC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE

    return EncoderOnlyTransformer(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        normalize_before=pre_norm,
        return_intermediate=True,
    )


def build_decoder_only_transformer(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    dec_layers = cfg.MODEL.DETECTOR.DEC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE

    return DecoderOnlyTransformer(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate=True,
    )


def build_detr_based_graph_module(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    enc_layers = cfg.MODEL.DETECTOR.ENC_LAYERS
    dec_layers = cfg.MODEL.DETECTOR.DEC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE
    num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY 
    num_hidden_rel_query = cfg.MODEL.DETECTOR.NUM_HIDDEN_REL_QUERY

    return DETRGraphModule(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_obj_query=num_obj_query, 
        num_hidden_rel=num_hidden_rel_query,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )


def build_scene_graph_module(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    enc_layers = cfg.MODEL.DETECTOR.ENC_LAYERS
    dec_layers = cfg.MODEL.DETECTOR.DEC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE
    num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY 
    num_hidden_rel_query = cfg.MODEL.DETECTOR.NUM_HIDDEN_REL_QUERY

    return SceneGraphModule(
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


def build_scene_graph_with_exknowledge_module(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    enc_layers = cfg.MODEL.DETECTOR.ENC_LAYERS
    dec_layers = cfg.MODEL.DETECTOR.DEC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE
    num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY 
    num_hidden_rel_query = cfg.MODEL.DETECTOR.NUM_HIDDEN_REL_QUERY

    with open(cfg.MODEL.DETECTOR.WORD_EMBEDDING_FILE, 'rb') as f:
        emb_mtx = pickle.load(f)
    word_embedding = {'obj_features': emb_mtx[0], 'rel_features': emb_mtx[1]}

    with open(cfg.MODEL.DETECTOR.CORRELATION_FILE, 'rb') as f:
        all_edges = pickle.load(f)
    correlation = {'obj2obj': all_edges['edges_ent2ent'],
                   'rel2rel': all_edges['edges_pred2pred'],
                   'obj2rel': all_edges['edges_ent2pred'],
                   'rel2obj': all_edges['edges_pred2ent']}

    return SGGTRWithEXKnowledge(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_obj_query=num_obj_query, 
        num_hidden_rel=num_hidden_rel_query,
        num_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate=True,
        word_embedding=word_embedding, 
        correlation=correlation,
    )


def build_sggdetr_module(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    nhead = cfg.MODEL.DETECTOR.NHEAD
    dim_feedforward = cfg.MODEL.DETECTOR.DIM_FEEDFORWARD
    dropout = cfg.MODEL.DETECTOR.DROPOUT
    enc_layers = cfg.MODEL.DETECTOR.ENC_LAYERS
    dec_layers = cfg.MODEL.DETECTOR.DEC_LAYERS
    pre_norm = cfg.MODEL.DETECTOR.NORMALIZE_BEFORE
    num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY 
    num_hidden_rel_query = cfg.MODEL.DETECTOR.NUM_HIDDEN_REL_QUERY

    return SGGDEModule(
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

