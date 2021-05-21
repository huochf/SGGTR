# Modified from DETR (https://github.com/facebookresearch/detr)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectHeader(nn.Module):

    def __init__(self, d_model, num_classes, need_sigmoid=True):
        super().__init__()
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.need_sigmoid = need_sigmoid


    def forward(self, object_features, ):
        """
        Arguments:
            object_features: [..., d_model]
        Returns:
            obj_class_logits: [..., num_classes + 1]
            obj_box_coords: [..., 4]
        """
        obj_class_logits = self.class_embed(object_features)
        obj_box_coords = self.bbox_embed(object_features)
        if self.need_sigmoid:
            obj_box_coords = obj_box_coords.sigmoid()

        return obj_class_logits, obj_box_coords


class RelationHeader(nn.Module):

    def __init__(self, d_model, num_obj_classes, num_rel_classes,):
        super().__init__()
        self.obj_class_embed = nn.Linear(d_model, num_obj_classes + 1)
        self.sub_class_embed = nn.Linear(d_model, num_obj_classes + 1)
        self.rel_class_embed = nn.Linear(d_model, num_rel_classes + 1)
        self.obj_bbox_embed = MLP(d_model, d_model, 4, 3)
        self.sub_bbox_embed = MLP(d_model, d_model, 4, 3)


    def forward(self, relation_features):
        """
        Arguments:
        """
        obj_class_logits = self.obj_class_embed(relation_features)
        sub_class_logits = self.sub_class_embed(relation_features)
        rel_class_logits = self.rel_class_embed(relation_features)
        obj_box_coords = self.obj_bbox_embed(relation_features).sigmoid()
        sub_box_coords = self.sub_bbox_embed(relation_features).sigmoid()

        outs = {"relation_obj_cls_logits": obj_class_logits,
                "relation_sub_cls_logits": sub_class_logits,
                "relation_rel_cls_logits": rel_class_logits,
                "relation_obj_box_coords": obj_box_coords,
                "relation_sub_box_coords": sub_box_coords}

        return outs


class GraphHeader(nn.Module):

    def __init__(self, d_model, num_obj_classes, num_rel_classes, num_obj_query):
        super().__init__()
        self.obj_class_embed = nn.Linear(d_model, num_obj_classes + 1)
        self.rel_class_embed = nn.Linear(d_model, num_rel_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.num_obj_query = num_obj_query


    def forward(self, graph_features):
        """
        Arguments:
            graph_features: [num_layer, b, num_nodes, d_model]
        Returns:
            obj_class_logits: [..., num_classes + 1]
            obj_box_coords: [..., 4]
        """
        obj_features, rel_features = graph_features[:, :, :self.num_obj_query], graph_features[:, :, self.num_obj_query:]
        obj_class_logits = self.obj_class_embed(obj_features)
        rel_class_logits = self.rel_class_embed(rel_features)
        obj_box_coords = self.bbox_embed(obj_features).sigmoid()
        return obj_class_logits, obj_box_coords, rel_class_logits


class GraphHeaderWithExknowledge(nn.Module):

    def __init__(self, d_model, num_obj_classes, num_rel_classes, num_obj_query):
        super().__init__()
        self.obj_class_embed = nn.Linear(d_model, num_obj_classes + 1)
        self.rel_class_embed = nn.Linear(d_model, num_rel_classes + 1)

        self.obj2obj_embed = nn.Linear(d_model, (num_obj_classes + 1) * 8)
        self.obj2rel_embed = nn.Linear(d_model, (num_rel_classes + 1) * 2)
        self.rel2obj_embed = nn.Linear(d_model, (num_obj_classes + 1) * 2)
        self.rel2rel_embed = nn.Linear(d_model, (num_rel_classes + 1) * 4)

        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.num_obj_query = num_obj_query


    def forward(self, graph_features):
        """
        Arguments:
            graph_features: [num_layer, b, num_nodes, d_model]
        Returns:
            obj_class_logits: [..., num_classes + 1]
            obj_box_coords: [..., 4]
        """
        obj_features, rel_features = graph_features[:, :, :self.num_obj_query], graph_features[:, :, self.num_obj_query:]
        obj_class_logits = self.obj_class_embed(obj_features)
        rel_class_logits = self.rel_class_embed(rel_features)

        obj2obj_class_logits = self.obj2obj_embed(obj_features).sigmoid()
        obj2rel_class_logits = self.obj2rel_embed(obj_features).sigmoid()
        rel2obj_class_logits = self.rel2obj_embed(rel_features).sigmoid()
        rel2rel_class_logits = self.rel2rel_embed(rel_features).sigmoid()

        obj_box_coords = self.bbox_embed(obj_features).sigmoid()
        return obj_class_logits, obj_box_coords, rel_class_logits, obj2obj_class_logits, obj2rel_class_logits, rel2obj_class_logits, rel2rel_class_logits


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_obj_detection_header(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    num_obj_class = cfg.MODEL.DETECTION_HEADER.NUM_OBJ_CLASS
    return ObjectHeader(d_model, num_obj_class, cfg.MODEL.DETECTION_HEADER.BOX_FORMAT != "bcxbcywh")


def build_rel_detection_header(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    num_obj_class = cfg.MODEL.DETECTION_HEADER.NUM_OBJ_CLASS
    num_rel_class = cfg.MODEL.DETECTION_HEADER.NUM_REL_CLASS

    return RelationHeader(d_model, num_obj_class, num_rel_class)


def build_graph_detection_header(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    num_obj_class = cfg.MODEL.DETECTION_HEADER.NUM_OBJ_CLASS
    num_rel_class = cfg.MODEL.DETECTION_HEADER.NUM_REL_CLASS
    num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY
    return GraphHeader(d_model, num_obj_class, num_rel_class, num_obj_query)


def build_graph_with_exknowledge_detection_header(cfg):
    d_model = cfg.MODEL.DETECTOR.D_MODEL
    num_obj_class = cfg.MODEL.DETECTION_HEADER.NUM_OBJ_CLASS
    num_rel_class = cfg.MODEL.DETECTION_HEADER.NUM_REL_CLASS
    num_obj_query = cfg.MODEL.DETECTOR.NUM_OBJ_QUERY
    return GraphHeaderWithExknowledge(d_model, num_obj_class, num_rel_class, num_obj_query)
