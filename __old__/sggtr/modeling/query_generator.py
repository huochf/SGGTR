import torch
import torch.nn as nn 

from . import registry
from .position_encoding import build_query_position_encoding

@registry.QUERY_GENERATOR.register("ObjectQueryGenerator")
class ObjectQueryGenerator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY
        d_model = cfg.MODEL.DETECTOR.D_MODEL
        self.query_embed = nn.Embedding(num_obj_query, d_model)


    def forward(self, ):
        return self.query_embed.weight


@registry.QUERY_GENERATOR.register("DenseObjectQueryGenerator")
class DenseObjectQueryGenerator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        out_channels = 2048 # we fix it here
        d_model = cfg.MODEL.DETECTOR.D_MODEL
        self.feature_proj = nn.Conv2d(out_channels, d_model, kernel_size=1)
        self.pos_embed = build_query_position_encoding(cfg)


    def forward(self, tensor_list):
        # queries = self.feature_proj(tensor_list.tensors)
        queries = tensor_list.tensors
        query_embed = self.pos_embed(tensor_list)

        return queries, query_embed


@registry.QUERY_GENERATOR.register("GraphQueryGenerator")
class GraphQueryGenerator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY
        d_model = cfg.MODEL.DETECTOR.D_MODEL
        self.num_obj_query = num_obj_query
        self.d_model = d_model
        self.query_embed = nn.Embedding(num_obj_query, d_model)
        self.rel_proj = nn.Linear(2 * d_model, d_model)


    def forward(self, ):
        obj_query_embed = self.query_embed.weight
        rel_query_embed = torch.cat((
            obj_query_embed.unsqueeze(0).repeat(self.num_obj_query, 1, 1),
            obj_query_embed.unsqueeze(1).repeat(1, self.num_obj_query, 1)
        ), -1).reshape(-1, self.d_model * 2)
        rel_query_embed = self.rel_proj(rel_query_embed)
        graph_query = torch.cat((obj_query_embed, rel_query_embed), 0)
        return graph_query






