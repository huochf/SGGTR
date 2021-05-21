# Modified from DETR (https://github.com/facebookresearch/detr)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pickle

from .matcher import build_matcher

from sggtr.structures.boxlist_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


# =======================================
# TODO: to make these codes more compact.
# =======================================

class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR, RELTR or SGGTR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, num_obj_class, num_rel_class, no_obj_cls_weight, no_rel_cls_weight, rel_multi_label=False,
                 box_format="bcxbcywh", compute_aux_loss=True, with_exknowledge_loss=False, all_edges=None,
                 with_additional_rel_loss=False, additional_no_rel_cls_weight=0.1):
        super().__init__()

        self.matcher = matcher
        self.num_obj_class = num_obj_class
        self.num_rel_class = num_rel_class

        obj_empty_weight = torch.ones(self.num_obj_class + 1)
        obj_empty_weight[0] = no_obj_cls_weight
        rel_empty_weight = torch.ones(self.num_rel_class + 1)
        rel_empty_weight[0] = no_rel_cls_weight
        self.register_buffer("obj_empty_weight", obj_empty_weight)
        self.register_buffer("rel_empty_weight", rel_empty_weight)
        self.no_obj_cls_weight = no_obj_cls_weight
        self.no_rel_cls_weight = no_rel_cls_weight

        self.with_additional_rel_loss = with_additional_rel_loss
        self.additional_no_rel_cls_weight = additional_no_rel_cls_weight
        additional_rel_empty_weight = torch.ones(self.num_rel_class + 1)
        rel_empty_weight[0] = additional_no_rel_cls_weight
        self.register_buffer("additional_rel_empty_weight", additional_rel_empty_weight)

        assert box_format in ("bcxbcywh", "bx1by1bx2by2", "cxcywh")
        self.box_format = box_format

        self.compute_aux_loss = compute_aux_loss
        self.rel_multi_label = rel_multi_label

        self.with_exknowledge_loss = with_exknowledge_loss
        self.all_edges = all_edges

        if with_exknowledge_loss:
            correlation = {'obj2obj': all_edges['edges_ent2ent'],
                   'rel2rel': all_edges['edges_pred2pred'],
                   'obj2rel': all_edges['edges_ent2pred'],
                   'rel2obj': all_edges['edges_pred2ent']}
            self.obj2obj = nn.Parameter(torch.tensor(correlation['obj2obj'][:8], dtype=torch.float32), requires_grad=False) # [9, 151, 151]
            self.rel2rel = nn.Parameter(torch.tensor(correlation['rel2rel'], dtype=torch.float32), requires_grad=False) # [4,  51,  51]
            self.obj2rel = nn.Parameter(torch.tensor(correlation['obj2rel'][:2], dtype=torch.float32), requires_grad=False) # [3, 151,  51]
            self.rel2obj = nn.Parameter(torch.tensor(correlation['rel2obj'][:2], dtype=torch.float32), requires_grad=False) # [3,  51, 151]


    def loss_labels(self, pred_logits, target_labels, matched_indices, empty_weight):
        """
        Classification loss (NLL)
        Arguments:
            pred_logits: [b, num_queries, num_cls + 1]
            target_labels: list of Tensor with shape [num_box, ]
        """
        idx = self._get_src_permutation_idx(matched_indices)

        target_labels = torch.cat([label[J] for label, (_, J) in zip(target_labels, matched_indices)])
        target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_labels

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, empty_weight)

        return loss_ce


    def obj_loss_knowledge_multi_label(self, obj_pred_logits, rel_pred_logits, target_labels, matched_indices):
        """
        Arguments:
            obj_pred_logits: [b, num_queries, 8 * 151]
            rel_pred_logits: [b, num_queries, 2 * 51]
            target_labels: list of Tensor with shape [num_box, ]
        """
        idx = self._get_src_permutation_idx(matched_indices)

        target_labels = torch.cat([label[J] for label, (_, J) in zip(target_labels, matched_indices)])
        target_obj_exknowledge_label = self.obj2obj.transpose(0, 1)[target_labels] # [all_queries, 8, 151]
        target_obj_exknowledge_label = target_obj_exknowledge_label.view((target_obj_exknowledge_label.shape[0], 8 * 151))
        target_rel_exknowledge_label = self.obj2rel.transpose(0, 1)[target_labels] # [all_queries, 2, 51]
        target_rel_exknowledge_label = target_rel_exknowledge_label.view((target_rel_exknowledge_label.shape[0], 2 * 51))

        b, num_queries = obj_pred_logits.shape[:2]
        target_obj_exknowledge_logits = torch.zeros((b, num_queries, 8 * 151), dtype=torch.float32, device=obj_pred_logits.device)
        target_obj_exknowledge_logits[idx] = target_obj_exknowledge_label
        target_rel_exknowledge_logits = torch.zeros((b, num_queries, 2 * 51), dtype=torch.float32, device=obj_pred_logits.device)
        target_rel_exknowledge_logits[idx] = target_rel_exknowledge_label

        loss_knowledge_obj = self._neg_loss(obj_pred_logits, target_obj_exknowledge_logits, weight=self.no_obj_cls_weight)
        loss_knowledge_rel = self._neg_loss(rel_pred_logits, target_rel_exknowledge_logits, weight=self.no_obj_cls_weight)

        return loss_knowledge_obj, loss_knowledge_rel


    def rel_loss_knowledge_multi_label(self, obj_pred_logits, rel_pred_logits, target_labels):
        """
        Arguments:
            obj_pred_logits: [b, num_queries, 2 * 151]
            rel_pred_logits: [b, num_queries, 4 * 51]
            target_labels: list of Tensor with shape [b, num_queries, ]
        """

        target_labels = target_labels.reshape(-1) # [all_queries, ]
        target_obj_exknowledge_label = self.rel2obj.transpose(0, 1)[target_labels] # [all_queries, 8, 151]
        target_obj_exknowledge_label = target_obj_exknowledge_label.view((target_obj_exknowledge_label.shape[0], 2 * 151))
        target_rel_exknowledge_label = self.rel2rel.transpose(0, 1)[target_labels] # [all_queries, 2, 51]
        target_rel_exknowledge_label = target_rel_exknowledge_label.view((target_rel_exknowledge_label.shape[0], 4 * 51))

        b, num_queries = obj_pred_logits.shape[:2]

        loss_knowledge_obj = self._neg_loss(obj_pred_logits.view(b * num_queries, -1), target_obj_exknowledge_label, weight=self.no_rel_cls_weight)
        loss_knowledge_rel = self._neg_loss(rel_pred_logits.view(b * num_queries, -1), target_rel_exknowledge_label, weight=self.no_rel_cls_weight)

        return loss_knowledge_obj, loss_knowledge_rel


    def loss_boxes(self, pred_boxes, target_boxes, matched_indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(matched_indices)
        pred_boxes = pred_boxes[idx]
        target_boxes = torch.cat([boxes[i] for boxes, (_, i) in zip(target_boxes, matched_indices)])

        loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction="none", )
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou


    def _get_src_permutation_idx(self, indices):
        # permute perdictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def bcxbcywh_to_cxcywh(self, pred_boxes, reference_points):
        """
        Arguments:
            pred_boxes: [num_layer, b, num_query, 4]
            reference_points: [num_query, 2]
        """
        cx, cy = reference_points.unbind(-1)
        bcx, bcy, w, h = pred_boxes.unbind(-1)
        w = w.sigmoid()
        h = h.sigmoid()
        cx = (bcx + cx).sigmoid()
        cy = (bcy + cy).sigmoid()

        return torch.stack((cx, cy, w, h), -1)


    def bx1by1bx2by2_to_cxcywh(self, pred_boxes, reference_points):
        reference_points = reference_points.sigmoid()
        cx, cy = reference_points.unbind(-1)
        bx1, by1, bx2, by2 = pred_boxes.unbind(-1)
        x1, y1, x2, y2 = cx - bx1, cy - by1, cx + bx2, cy + by2
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)

        return torch.stack((cx, cy, w, h), -1)


    def forward(self, outputs, targets):
        """
        Arguments:
            outputs:
                - outputs["obj_cls_logits"], Tensor with shape [num_layer, b, num_obj_query, num_obj_class + 1]
                - outputs["obj_box_coord"], Tensor with shape [num_layer, b, num_obj_query, 4]
                - outputs["relation_obj_cls_logits"], Tensor with shape [num_layer, b, num_rel_query, num_obj_class + 1]
                - outputs["relation_obj_box_coord"], Tensor with shape [num_layer, b, num_rel_query, 4]
                - outputs["relation_rel_cls_logits"], Tensor with shape [num_layer, b, num_rel_query, num_rel_class + 1]
                - outputs["graph_obj_cls_logits"], Tensor with shape [num_layer, b, num_obj_query, num_obj_class + 1]
                - outputs["graph_obj_box_coord"], Tensor with shape [num_layer, b, num_obj_query, 4]
                - outputs["graph_rel_cls_logits"], Tensor with shape [num_layer, b, num_obj_query * num_obj_query, num_rel_class + 1]
            targets:
                list of BoxList
        Returns:
            loss_dict:
                - loss_dict[""]
        """
        loss_dict = {}
        targets = [target.convert("cxcywh").normalize_boxes() for target in targets] # normalized cxcywh

        # For DETR model
        if "obj_cls_logits" in outputs and outputs["obj_cls_logits"] is not None:
            pred_logits = outputs["obj_cls_logits"]
            pred_boxes = outputs["obj_box_coord"]
            target_labels = [target.get_field("labels") for target in targets]
            target_boxes = [target.bbox for target in targets]

            matched_indices = self.matcher(pred_logits, pred_boxes, target_labels, target_boxes)

            num_boxes = sum(len(target) for target in targets)
            loss_obj_class = self.loss_labels(pred_logits[-1], target_labels, matched_indices, self.obj_empty_weight)
            loss_bbox, loss_giou = self.loss_boxes(pred_boxes[-1], target_boxes, matched_indices, num_boxes)
            loss_dict["loss_obj_class"] = loss_obj_class
            loss_dict["loss_bbox"] = loss_bbox
            loss_dict["loss_giou"] = loss_giou

            if self.compute_aux_loss:
                num_layers = pred_logits.shape[0]
                for i in range(num_layers - 1):
                    aux_loss_obj_class = self.loss_labels(pred_logits[i], target_labels, matched_indices, self.obj_empty_weight)
                    aux_loss_bbox, aux_loss_giou = self.loss_boxes(pred_boxes[i], target_boxes, matched_indices, num_boxes)
                    loss_dict["aux_obj_class_%d" % i] = aux_loss_obj_class
                    loss_dict["aux_loss_bbox_%d" % i] = aux_loss_bbox
                    loss_dict["aux_loss_giou_%d" % i] = aux_loss_giou 

        # For Encoder-Only DETR ...
        if "ref_obj_cls_logits" in outputs and outputs["ref_obj_cls_logits"] is not None:
            pred_logits = outputs["ref_obj_cls_logits"]
            pred_boxes = outputs["ref_obj_box_coord"] # bcxbcywh or bx1by1bx2by2 format
            reference_points = outputs["reference_points"]

            if self.box_format == "bcxbcywh":
                pred_boxes = self.bcxbcywh_to_cxcywh(pred_boxes, reference_points)
            elif self.box_format == "bx1by1bx2by2":
                pred_boxes = self.bx1by1bx2by2_to_cxcywh(pred_boxes, reference_points)

            target_labels = [target.get_field("labels") for target in targets]
            target_boxes = [target.bbox for target in targets]

            matched_indices = self.matcher(pred_logits, pred_boxes, target_labels, target_boxes)

            num_boxes = sum(len(target) for target in targets)
            loss_obj_class = self.loss_labels(pred_logits[-1], target_labels, matched_indices, self.obj_empty_weight)
            loss_bbox, loss_giou = self.loss_boxes(pred_boxes[-1], target_boxes, matched_indices, num_boxes)
            loss_dict["loss_obj_class"] = loss_obj_class
            loss_dict["loss_bbox"] = loss_bbox
            loss_dict["loss_giou"] = loss_giou

            if self.compute_aux_loss:
                num_layers = pred_logits.shape[0]
                for i in range(num_layers - 1):
                    aux_loss_obj_class = self.loss_labels(pred_logits[i], target_labels, matched_indices, self.obj_empty_weight)
                    aux_loss_bbox, aux_loss_giou = self.loss_boxes(pred_boxes[i], target_boxes, matched_indices, num_boxes)
                    loss_dict["aux_obj_class_%d" % i] = aux_loss_obj_class
                    loss_dict["aux_loss_bbox_%d" % i] = aux_loss_bbox
                    loss_dict["aux_loss_giou_%d" % i] = aux_loss_giou 

        # For RELTR ...
        if "relation_rel_cls_logits" in outputs and outputs["relation_rel_cls_logits"] is not None:
            pred_obj_logits = outputs["relation_obj_cls_logits"]
            pred_sub_logits = outputs["relation_sub_cls_logits"]
            pred_rel_logits = outputs["relation_rel_cls_logits"]
            pred_obj_boxes = outputs["relation_obj_box_coords"]
            pred_sub_boxes = outputs["relation_sub_box_coords"]

            target_obj_labels = []
            target_sub_labels = []
            target_obj_boxes = []
            target_sub_boxes = []
            target_rel_labels = []
            for target in targets:
                all_boxes = target.bbox
                all_labels = target.get_field("labels")
                triplet = target.get_field("relation_tuple")
                sub_indices, obj_indices, relation_labels = triplet.unbind(-1)

                target_obj_labels.append(all_labels[obj_indices])
                target_sub_labels.append(all_labels[sub_indices])
                target_obj_boxes.append(all_boxes[obj_indices])
                target_sub_boxes.append(all_boxes[sub_indices])
                target_rel_labels.append(relation_labels)

            matched_indices = self.matcher(pred_obj_logits, pred_obj_boxes, target_obj_labels, target_obj_boxes,
                                           pred_sub_logits, pred_sub_boxes, target_sub_labels, target_sub_boxes,
                                           pred_rel_logits, target_rel_labels)

            num_boxes = sum([boxes.shape[0] for boxes in target_obj_boxes])

            obj_class_loss = self.loss_labels(pred_obj_logits[-1], target_obj_labels, matched_indices, self.obj_empty_weight)
            sub_class_loss = self.loss_labels(pred_sub_logits[-1], target_sub_labels, matched_indices, self.obj_empty_weight)
            rel_class_loss = self.loss_labels(pred_rel_logits[-1], target_rel_labels, matched_indices, self.rel_empty_weight)
            obj_loss_bbox, obj_loss_giou = self.loss_boxes(pred_obj_boxes[-1], target_obj_boxes, matched_indices, num_boxes)
            sub_loss_bbox, sub_loss_giou = self.loss_boxes(pred_sub_boxes[-1], target_sub_boxes, matched_indices, num_boxes)
            loss_dict["obj_class_loss"] = obj_class_loss
            loss_dict["sub_class_loss"] = sub_class_loss
            loss_dict["rel_class_loss"] = rel_class_loss
            loss_dict["obj_loss_bbox"] = obj_loss_bbox
            loss_dict["obj_loss_giou"] = obj_loss_giou
            loss_dict["sub_loss_bbox"] = sub_loss_bbox
            loss_dict["sub_loss_giou"] = sub_loss_giou

            if self.compute_aux_loss:
                num_layers = pred_obj_logits.shape[0]
                for i in range(num_layers - 1):
                    obj_class_loss = self.loss_labels(pred_obj_logits[i], target_obj_labels, matched_indices, self.obj_empty_weight)
                    sub_class_loss = self.loss_labels(pred_sub_logits[i], target_sub_labels, matched_indices, self.obj_empty_weight)
                    rel_class_loss = self.loss_labels(pred_rel_logits[i], target_rel_labels, matched_indices, self.rel_empty_weight)
                    obj_loss_bbox, obj_loss_giou = self.loss_boxes(pred_obj_boxes[i], target_obj_boxes, matched_indices, num_boxes)
                    sub_loss_bbox, sub_loss_giou = self.loss_boxes(pred_sub_boxes[i], target_sub_boxes, matched_indices, num_boxes)
                    loss_dict["aux_obj_class_loss_%d" % i] = obj_class_loss
                    loss_dict["aux_sub_class_loss_%d" % i] = sub_class_loss
                    loss_dict["aux_rel_class_loss_%d" % i] = rel_class_loss
                    loss_dict["aux_obj_loss_bbox_%d" % i] = obj_loss_bbox
                    loss_dict["aux_obj_loss_giou_%d" % i] = obj_loss_giou
                    loss_dict["aux_sub_loss_bbox_%d" % i] = sub_loss_bbox
                    loss_dict["aux_sub_loss_giou_%d" % i] = sub_loss_giou

        # For SGGTR models
        if "graph_obj_cls_logits" in outputs and outputs["graph_obj_cls_logits"] is not None:
            obj_cls_logits = outputs["graph_obj_cls_logits"]
            rel_cls_logits = outputs["graph_rel_cls_logits"]
            obj_box_coord = outputs["graph_obj_box_coord"]

            target_labels = [target.get_field("labels") for target in targets]
            target_boxes = [target.bbox for target in targets]

            matched_indices = self.matcher(obj_cls_logits, obj_box_coord, target_labels, target_boxes)

            num_boxes = sum(len(target) for target in targets)
            loss_obj_class = self.loss_labels(obj_cls_logits[-1], target_labels, matched_indices, self.obj_empty_weight)
            loss_bbox, loss_giou = self.loss_boxes(obj_box_coord[-1], target_boxes, matched_indices, num_boxes)
            loss_dict["loss_sgg_obj_class"] = loss_obj_class
            loss_dict["loss_sgg_bbox"] = loss_bbox
            loss_dict["loss_sgg_giou"] = loss_giou

            relationship_maps = [target.get_field("relation") for target in targets]
            num_layer, b, num_obj_query, _ = obj_cls_logits.shape
            target_relationship_map = torch.zeros((b, num_obj_query, num_obj_query), dtype=torch.int64, device=relationship_maps[0].device)

            for i, (src_idx, tgt_idx) in enumerate(matched_indices):
                target_relationship_map[i, src_idx[:, None], src_idx[None, :]] = relationship_maps[i][tgt_idx[:, None], tgt_idx[None, :]]
            target_relationship_map = target_relationship_map.reshape(b, -1)

            if self.rel_multi_label:
                relationship_maps = [target.get_field("relation_cls_binary_map") for target in targets]
                target_relationship_map = torch.zeros((b, num_obj_query, num_obj_query, 118), dtype=torch.int64, device=relationship_maps[0].device)

                for i, (src_idx, tgt_idx) in enumerate(matched_indices):
                    target_relationship_map[i, src_idx[:, None], src_idx[None, :]] = relationship_maps[i][tgt_idx[:, None], tgt_idx[None, :]]
                target_relationship_map = target_relationship_map.reshape(b, -1, 118)
                loss_rel_class = self._neg_loss(rel_cls_logits[-1].sigmoid(), target_relationship_map, weight=self.no_rel_cls_weight)
            else:
                loss_rel_class = F.cross_entropy(rel_cls_logits[-1].transpose(1, 2), target_relationship_map, self.rel_empty_weight)
            loss_dict["loss_sgg_rel_class"] = loss_rel_class

            if self.with_additional_rel_loss:
                if self.rel_multi_label:
                    additional_rel_loss = 0
                    count = 0
                    for b, (src_idx, tgt_idx) in enumerate(matched_indices):
                        match_box_num = len(src_idx)
                        src_rel_pred_logits = rel_cls_logits[-1][b].reshape(num_obj_query, num_obj_query, self.num_rel_class + 1)
                        src_rel_pred_logits = src_rel_pred_logits[src_idx[:, None], src_idx[None, :]].reshape(match_box_num, match_box_num, self.num_rel_class + 1)
                        tgt_rel_pred_labels = relationship_maps[b][tgt_idx[:, None], tgt_idx[None, :]].reshape(match_box_num, match_box_num, self.num_rel_class + 1)

                        row_idx = tgt_rel_pred_labels.any(2).any(1).nonzero().reshape(-1)
                        col_idx = tgt_rel_pred_labels.any(2).any(0).nonzero().reshape(-1)
                        if len(row_idx) != 0 and len(col_idx) != 0:
                            src_rel_pred_logits = src_rel_pred_logits[row_idx[:, None], col_idx[None, :]].reshape(-1, self.num_rel_class + 1)
                            tgt_rel_pred_labels = tgt_rel_pred_labels[row_idx[:, None], col_idx[None, :]].reshape(-1, self.num_rel_class + 1)
                            additional_rel_loss += self._neg_loss(rel_cls_logits[-1].sigmoid(), target_relationship_map, weight=self.additional_no_rel_cls_weight)
                            count += 1
                    if count != 0:
                        additional_rel_loss /= count

                else:
                    additional_rel_loss = 0
                    count = 0
                    for b, (src_idx, tgt_idx) in enumerate(matched_indices):
                        match_box_num = len(src_idx)
                        src_rel_pred_logits = rel_cls_logits[-1][b].reshape(num_obj_query, num_obj_query, self.num_rel_class + 1)
                        src_rel_pred_logits = src_rel_pred_logits[src_idx[:, None], src_idx[None, :]].reshape(match_box_num, match_box_num, self.num_rel_class + 1)
                        tgt_rel_pred_labels = relationship_maps[b][tgt_idx[:, None], tgt_idx[None, :]].reshape(match_box_num, match_box_num)

                        row_idx = tgt_rel_pred_labels.any(1).nonzero().reshape(-1)
                        col_idx = tgt_rel_pred_labels.any(0).nonzero().reshape(-1)
                        if len(row_idx) != 0 and len(col_idx) != 0:
                            src_rel_pred_logits = src_rel_pred_logits[row_idx[:, None], col_idx[None, :]].reshape(-1, self.num_rel_class + 1)
                            tgt_rel_pred_labels = tgt_rel_pred_labels[row_idx[:, None], col_idx[None, :]].reshape(-1)
                            additional_rel_loss +=  F.cross_entropy(src_rel_pred_logits, tgt_rel_pred_labels, self.additional_rel_empty_weight)
                            count += 1
                    if count != 0:
                        additional_rel_loss /= count
                loss_dict['addi_rel_loss'] = additional_rel_loss


            if self.with_exknowledge_loss:
                pred_obj2obj_class_logits = outputs["obj2obj_class_logits"]
                pred_obj2rel_class_logits = outputs["obj2rel_class_logits"]
                pred_rel2obj_class_logits = outputs["rel2obj_class_logits"]
                pred_rel2rel_class_logits = outputs["rel2rel_class_logits"]

                obj2obj_knowledge_loss, obj2rel_knowledge_loss = self.obj_loss_knowledge_multi_label(pred_obj2obj_class_logits[-1], pred_obj2rel_class_logits[-1], target_labels, matched_indices)
                rel2obj_knowledge_loss, rel2rel_knowledge_loss = self.rel_loss_knowledge_multi_label(pred_rel2obj_class_logits[-1], pred_rel2rel_class_logits[-1], target_relationship_map)

                loss_dict["obj2obj_knowledge_loss"] = obj2obj_knowledge_loss
                loss_dict["obj2rel_knowledge_loss"] = obj2rel_knowledge_loss
                loss_dict["rel2obj_knowledge_loss"] = rel2obj_knowledge_loss
                loss_dict["rel2rel_knowledge_loss"] = rel2rel_knowledge_loss

            if self.compute_aux_loss:
                for i in range(num_layer - 1):
                    loss_obj_class = self.loss_labels(obj_cls_logits[i], target_labels, matched_indices, self.obj_empty_weight)
                    loss_bbox, loss_giou = self.loss_boxes(obj_box_coord[i], target_boxes, matched_indices, num_boxes)
                    loss_dict["aux_sgg_loss_obj_class_%d" % i] = loss_obj_class
                    loss_dict["aux_sgg_loss_bbox_%d" % i] = loss_bbox
                    loss_dict["aux_sgg_loss_giou_%d" % i] = loss_giou
                    if self.rel_multi_label:
                        loss_rel_class = self._neg_loss(rel_cls_logits[i].sigmoid(), target_relationship_map, weight=self.no_rel_cls_weight)
                    else:
                        loss_rel_class = F.cross_entropy(rel_cls_logits[i].transpose(1, 2), target_relationship_map, self.rel_empty_weight)
                    loss_dict["aux_sgg_loss_rel_class_%d" % i] = loss_rel_class

                
                    if self.with_additional_rel_loss:
                        if self.rel_multi_label:
                            additional_rel_loss = 0
                            count = 0
                            for b, (src_idx, tgt_idx) in enumerate(matched_indices):
                                match_box_num = len(src_idx)
                                src_rel_pred_logits = rel_cls_logits[i][b].reshape(num_obj_query, num_obj_query, self.num_rel_class + 1)
                                src_rel_pred_logits = src_rel_pred_logits[src_idx[:, None], src_idx[None, :]].reshape(match_box_num, match_box_num, self.num_rel_class + 1)
                                tgt_rel_pred_labels = relationship_maps[b][tgt_idx[:, None], tgt_idx[None, :]].reshape(match_box_num, match_box_num, self.num_rel_class + 1)

                                row_idx = tgt_rel_pred_labels.any(2).any(1).nonzero().reshape(-1)
                                col_idx = tgt_rel_pred_labels.any(2).any(0).nonzero().reshape(-1)
                                if len(row_idx) != 0 and len(col_idx) != 0:
                                    src_rel_pred_logits = src_rel_pred_logits[row_idx[:, None], col_idx[None, :]].reshape(-1, self.num_rel_class + 1)
                                    tgt_rel_pred_labels = tgt_rel_pred_labels[row_idx[:, None], col_idx[None, :]].reshape(-1, self.num_rel_class + 1)
                                    additional_rel_loss += self._neg_loss(rel_cls_logits[-1].sigmoid(), target_relationship_map, weight=self.additional_no_rel_cls_weight)
                                    count += 1
                            if count != 0:
                                additional_rel_loss /= count

                        else:
                            additional_rel_loss = 0
                            count = 0
                            for b, (src_idx, tgt_idx) in enumerate(matched_indices):
                                match_box_num = len(src_idx)
                                src_rel_pred_logits = rel_cls_logits[i][b].reshape(num_obj_query, num_obj_query, self.num_rel_class + 1)
                                src_rel_pred_logits = src_rel_pred_logits[src_idx[:, None], src_idx[None, :]].reshape(match_box_num, match_box_num, self.num_rel_class + 1)
                                tgt_rel_pred_labels = relationship_maps[b][tgt_idx[:, None], tgt_idx[None, :]].reshape(match_box_num, match_box_num)

                                row_idx = tgt_rel_pred_labels.any(1).nonzero().reshape(-1)
                                col_idx = tgt_rel_pred_labels.any(0).nonzero().reshape(-1)
                                if len(row_idx) != 0 and len(col_idx) != 0:
                                    src_rel_pred_logits = src_rel_pred_logits[row_idx[:, None], col_idx[None, :]].reshape(-1, self.num_rel_class + 1)
                                    tgt_rel_pred_labels = tgt_rel_pred_labels[row_idx[:, None], col_idx[None, :]].reshape(-1)
                                    additional_rel_loss +=  F.cross_entropy(src_rel_pred_logits, tgt_rel_pred_labels, self.additional_rel_empty_weight)
                                    count += 1
                            if count != 0:
                                additional_rel_loss /= count
                        
                        loss_dict['aux_addi_rel_loss_%d' % i] = additional_rel_loss

                    # if i < 1:
                    #     continue

                    if self.with_exknowledge_loss:
                        obj2obj_knowledge_loss, obj2rel_knowledge_loss = self.obj_loss_knowledge_multi_label(pred_obj2obj_class_logits[i], pred_obj2rel_class_logits[-1], target_labels, matched_indices)
                        rel2obj_knowledge_loss, rel2rel_knowledge_loss = self.rel_loss_knowledge_multi_label(pred_rel2obj_class_logits[i], pred_rel2rel_class_logits[-1], target_relationship_map)

                        loss_dict["aux_obj2obj_knowledge_loss_%d" % i] = obj2obj_knowledge_loss
                        loss_dict["aux_obj2rel_knowledge_loss_%d" % i] = obj2rel_knowledge_loss
                        loss_dict["aux_rel2obj_knowledge_loss_%d" % i] = rel2obj_knowledge_loss
                        loss_dict["aux_rel2rel_knowledge_loss_%d" % i] = rel2rel_knowledge_loss
        return loss_dict


    def _neg_loss(self, pred, gt, weight=1):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * weight

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss


def build_loss(cfg):
    matcher = build_matcher(cfg)
    if cfg.LOSS.EXKNOWLEDGE_LOSS:
        with open(cfg.MODEL.DETECTOR.CORRELATION_FILE, 'rb') as f:
            all_edges = pickle.load(f)
    else:
        all_edges = None

    loss = SetCriterion(matcher=matcher,
                        num_obj_class=cfg.MODEL.DETECTION_HEADER.NUM_OBJ_CLASS,
                        num_rel_class=cfg.MODEL.DETECTION_HEADER.NUM_REL_CLASS,
                        no_obj_cls_weight=cfg.LOSS.NO_OBJ_CLS_WEIGHT, 
                        no_rel_cls_weight=cfg.LOSS.NO_REL_CLS_WEIGHT, 
                        with_additional_rel_loss=cfg.LOSS.WITH_ADDITIONAL_REL_LOSS, 
                        additional_no_rel_cls_weight=cfg.LOSS.ADDITION_NO_REL_CLS_WEIGHT,
                        rel_multi_label=cfg.LOSS.REL_MULTI_LABEL,
                        box_format=cfg.MODEL.DETECTION_HEADER.BOX_FORMAT, 
                        compute_aux_loss=cfg.LOSS.WITH_AUX_LOSS,
                        with_exknowledge_loss=cfg.LOSS.EXKNOWLEDGE_LOSS,
                        all_edges=all_edges)
    return loss
