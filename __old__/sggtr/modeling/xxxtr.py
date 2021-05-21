# Modified from DETR (https://github.com/facebookresearch/detr)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ---------------------------------------------------------------------------------------------
from typing import List, Optional

import torch
import torch.nn as nn

from .backbone import build_backbone
from .detector import Detector
from .transformer import (
    build_transformer, 
    build_encoder_only_transformer, 
    build_decoder_only_transformer,
    build_detr_based_graph_module, 
    build_scene_graph_module, 
    build_sggdetr_module,
    build_scene_graph_sgg_only_module,
    build_scene_graph_with_exknowledge_module
)
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
            return loss_dict, results
        else:
            return self.postprocess(outputs, image_sizes)


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


@registry.DETECTOR.register("DETR")
class DETR(Detector):

    def __init__(self, cfg):
        super(DETR, self).__init__()

        self.query_generator = ObjectQueryGenerator(cfg)
        self.transformer = build_transformer(cfg)
        self.detector_header = build_obj_detection_header(cfg)


    def forward(self, tensor_list):
        obj_query_embed = self.query_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        obj_features, _ = self.transformer(features, feature_mask, obj_query_embed, feature_pos)
        obj_cls_logits, obj_box_coord = self.detector_header(obj_features)

        return {"obj_cls_logits": obj_cls_logits, "obj_box_coord": obj_box_coord}


@registry.DETECTOR.register("EncoderOnlyDETR")
class EncoderOnlyDETR(Detector):

    def __init__(self, cfg):
        super(EncoderOnlyDETR, self).__init__()

        self.transformer = build_encoder_only_transformer(cfg)
        self.detector_header = build_obj_detection_header(cfg)
        self.reference_point_generator = ReferencePointsGenerator()


    def forward(self, tensor_list):
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        obj_features = self.transformer(features, feature_mask, feature_pos)
        obj_cls_logits, obj_box_coord = self.detector_header(obj_features)
        reference_points = self.reference_point_generator(feature_mask)

        return {"ref_obj_cls_logits": obj_cls_logits, 
                "ref_obj_box_coord": obj_box_coord, 
                "reference_points": reference_points}


@registry.DETECTOR.register("DecoderOnlyDETR")
class DecoderOnlyDETR(Detector):

    def __init__(self, cfg):
        super(DecoderOnlyDETR, self).__init__()

        self.query_generator = DenseObjectQueryGenerator(cfg)
        self.transformer = build_decoder_only_transformer(cfg)
        self.detector_header = build_obj_detection_header(cfg)
        self.reference_point_generator = ReferencePointsGenerator()


    def forward(self, tensor_list):
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        obj_queries, obj_query_embed = self.query_generator(tensor_list)
        obj_features = self.transformer(features, feature_mask, feature_pos, obj_queries, obj_query_embed)
        obj_cls_logits, obj_box_coord = self.detector_header(obj_features)
        reference_points = self.reference_point_generator(feature_mask)

        return {"ref_obj_cls_logits": obj_cls_logits, 
                "ref_obj_box_coord": obj_box_coord,
                "reference_points": reference_points}


@registry.DETECTOR.register("RELTR")
class RELTR(Detector):

    def __init__(self, cfg):
        super(RELTR, self).__init__()

        self.query_generator = ObjectQueryGenerator(cfg)
        self.transformer = build_transformer(cfg)
        self.detector_header = build_rel_detection_header(cfg)


    def forward(self, tensor_list):
        obj_query_embed = self.query_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        obj_features, _ = self.transformer(features, feature_mask, obj_query_embed, feature_pos)
        outputs = self.detector_header(obj_features)

        return outputs


@registry.DETECTOR.register("DETRBasedSGGTR")
class DETRBasedSGGTR(Detector):

    def __init__(self, cfg):
        super(DETRBasedSGGTR, self).__init__()

        self.query_generator = GraphQueryGenerator(cfg)
        self.transformer = build_detr_based_graph_module(cfg)
        self.detector_header = build_graph_detection_header(cfg)


    def forward(self, tensor_list):
        graph_query_embed = self.query_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        graph_features = self.transformer(features, feature_mask, feature_pos, graph_query_embed)
        graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits = self.detector_header(graph_features)

        return {"graph_obj_cls_logits": graph_obj_cls_logits, 
                "graph_obj_box_coord": graph_obj_box_coord,
                "graph_rel_cls_logits": graph_rel_cls_logits,}


@registry.DETECTOR.register("SGGTR")
class SGGTR(Detector):

    def __init__(self, cfg):
        super(SGGTR, self).__init__()

        self.obj_on = cfg.MODEL.DETECTION_HEADER.OBJ_HEADER_ON
        self.graph_on = cfg.MODEL.DETECTION_HEADER.GRAPH_HEADER_ON
        assert self.obj_on or self.graph_on

        self.query_generator = GraphQueryGenerator(cfg)
        self.query_pos_generator = GraphQueryGenerator(cfg)
        self.transformer = build_scene_graph_module(cfg)
        if self.obj_on:
            self.obj_detector_header = build_obj_detection_header(cfg)
            self.reference_point_generator = ReferencePointsGenerator()
        if self.graph_on:
            self.graph_detector_header = build_graph_detection_header(cfg)


    def forward(self, tensor_list):
        graph_query = self.query_generator()
        graph_query_embed = self.query_pos_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        features, graph_features = self.transformer(features, feature_mask, feature_pos, graph_query, graph_query_embed)

        obj_cls_logits = obj_box_coord = graph_obj_cls_logits = None
        reference_points = graph_obj_box_coord = graph_rel_cls_logits = None
        if self.obj_on:
            obj_cls_logits, obj_box_coord = self.obj_detector_header(features)
            reference_points = self.reference_point_generator(feature_mask)
        if self.graph_on:
            graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits = self.graph_detector_header(graph_features)

        return {"ref_obj_cls_logits": obj_cls_logits,
                "ref_obj_box_coord": obj_box_coord,
                "graph_obj_cls_logits": graph_obj_cls_logits, 
                "graph_obj_box_coord": graph_obj_box_coord,
                "graph_rel_cls_logits": graph_rel_cls_logits,
                "reference_points": reference_points}


@registry.DETECTOR.register("SGGTRSGGOnly")
class SGGTRSGGOnly(Detector):

    def __init__(self, cfg):
        super(SGGTRSGGOnly, self).__init__()
        self.query_generator = GraphQueryGenerator(cfg)
        self.query_pos_generator = GraphQueryGenerator(cfg)
        self.transformer = build_scene_graph_sgg_only_module(cfg)
        self.with_exknowledge_loss = cfg.LOSS.EXKNOWLEDGE_LOSS
        if cfg.LOSS.EXKNOWLEDGE_LOSS:
            self.graph_detector_header = build_graph_with_exknowledge_detection_header(cfg)
        else:
            self.graph_detector_header = build_graph_detection_header(cfg)


    def forward(self, tensor_list):
        graph_query = self.query_generator()
        graph_query_embed = self.query_pos_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        graph_features = self.transformer(features, feature_mask, feature_pos, graph_query, graph_query_embed)

        if self.with_exknowledge_loss:
            graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits, \
                obj2obj_class_logits, obj2rel_class_logits, rel2obj_class_logits, rel2rel_class_logits = self.graph_detector_header(graph_features)

            return {"graph_obj_cls_logits": graph_obj_cls_logits, 
                    "graph_obj_box_coord": graph_obj_box_coord,
                    "graph_rel_cls_logits": graph_rel_cls_logits,
                    "obj2obj_class_logits": obj2obj_class_logits,
                    "obj2rel_class_logits": obj2rel_class_logits,
                    "rel2obj_class_logits": rel2obj_class_logits,
                    "rel2rel_class_logits": rel2rel_class_logits}
        else:
            graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits = self.graph_detector_header(graph_features)

            return {"graph_obj_cls_logits": graph_obj_cls_logits, 
                    "graph_obj_box_coord": graph_obj_box_coord,
                    "graph_rel_cls_logits": graph_rel_cls_logits}


@registry.DETECTOR.register("SGGTRWithEXKnowledge")
class SGGTRWithEXKnowledge(Detector):

    def __init__(self, cfg):
        super(SGGTRWithEXKnowledge, self).__init__()
        self.query_generator = GraphQueryGenerator(cfg)
        self.query_pos_generator = GraphQueryGenerator(cfg)
        self.transformer = build_scene_graph_with_exknowledge_module(cfg)
        self.with_exknowledge_loss = cfg.LOSS.EXKNOWLEDGE_LOSS
        if cfg.LOSS.EXKNOWLEDGE_LOSS:
            self.graph_detector_header = build_graph_with_exknowledge_detection_header(cfg)
        else:
            self.graph_detector_header = build_graph_detection_header(cfg)


    def forward(self, tensor_list):
        graph_query = self.query_generator()
        graph_query_embed = self.query_pos_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        graph_features = self.transformer(features, feature_mask, feature_pos, graph_query, graph_query_embed)

        if self.with_exknowledge_loss:
            graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits, \
                obj2obj_class_logits, obj2rel_class_logits, rel2obj_class_logits, rel2rel_class_logits = self.graph_detector_header(graph_features)

            return {"graph_obj_cls_logits": graph_obj_cls_logits, 
                    "graph_obj_box_coord": graph_obj_box_coord,
                    "graph_rel_cls_logits": graph_rel_cls_logits,
                    "obj2obj_class_logits": obj2obj_class_logits,
                    "obj2rel_class_logits": obj2rel_class_logits,
                    "rel2obj_class_logits": rel2obj_class_logits,
                    "rel2rel_class_logits": rel2rel_class_logits}
        else:
            graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits = self.graph_detector_header(graph_features)

            return {"graph_obj_cls_logits": graph_obj_cls_logits, 
                    "graph_obj_box_coord": graph_obj_box_coord,
                    "graph_rel_cls_logits": graph_rel_cls_logits}


@registry.DETECTOR.register("SGGTR-DETR")
class SGGDETR(Detector):

    def __init__(self, cfg):
        super(SGGDETR, self).__init__()

        self.obj_on = cfg.MODEL.DETECTION_HEADER.OBJ_HEADER_ON
        self.graph_on = cfg.MODEL.DETECTION_HEADER.GRAPH_HEADER_ON
        assert self.obj_on or self.graph_on

        self.graph_query_generator = GraphQueryGenerator(cfg)
        self.transformer = build_sggdetr_module(cfg)
        if self.obj_on:
            self.obj_detector_header = build_obj_detection_header(cfg)
            self.reference_point_generator = ReferencePointsGenerator()
        if self.graph_on:
            self.graph_detector_header = build_graph_detection_header(cfg)


    def forward(self, tensor_list):
        graph_query_embed = self.graph_query_generator()
        features, feature_mask, _, feature_pos = tensor_list.decompose()
        features, graph_features = self.transformer(features, feature_mask, feature_pos, graph_query_embed)

        obj_cls_logits = obj_box_coord = graph_obj_cls_logits = None
        graph_obj_box_coord = graph_rel_cls_logits = reference_points = None
        if self.obj_on:
            obj_cls_logits, obj_box_coord = self.obj_detector_header(features)
            reference_points = self.reference_point_generator(feature_mask)
        if self.graph_on:
            graph_obj_cls_logits, graph_obj_box_coord, graph_rel_cls_logits = self.graph_detector_header(graph_features)

        return {"ref_obj_cls_logits": obj_cls_logits,
                "ref_obj_box_coord": obj_box_coord,
                "graph_obj_cls_logits": graph_obj_cls_logits, 
                "graph_obj_box_coord": graph_obj_box_coord,
                "graph_rel_cls_logits": graph_rel_cls_logits,
                "reference_points": reference_points}


class ReferencePointsGenerator(object):

    def __init__(self, ):
        pass


    def __call__(self, tensor_masks):
        # tensor_masks: [b, h, w]
        b = tensor_masks.shape[0]
        not_mask = ~tensor_masks
        x_pos = not_mask.cumsum(2, dtype=torch.float32)
        y_pos = not_mask.cumsum(1, dtype=torch.float32)

        eps = 1e-6
        x_pos = x_pos / (x_pos[:, :, -1:] + eps) + eps
        y_pos = y_pos / (y_pos[:, -1:, :] + eps) + eps

        # inverse sigmoid
        x_pos = (x_pos / (1 - x_pos + eps)).log()
        y_pos = (y_pos / (1 - y_pos + eps)).log()

        reference_points = torch.stack((x_pos, y_pos), -1)
        reference_points = reference_points.reshape(b, -1, 2)
        return reference_points
