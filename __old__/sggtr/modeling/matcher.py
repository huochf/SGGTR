# Modified from DETR (https://github.com/facebookresearch/detr)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ---------------------------------------------------------------------------------------------
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from sggtr.structures.boxlist_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_obj_class: float = 1, cost_rel_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Creates the matcher

        Params:
            cost_obj_class: This is the relative weight of the object classification
            cost_rel_class: This is the relative weight of the relation classification
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_rel_class = cost_rel_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_obj_class != 0 or cost_rel_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"


    @torch.no_grad()
    def forward(self, pred_obj_logits, pred_obj_boxes, target_obj_labels, target_obj_boxes,
                pred_sub_logits=None, pred_sub_boxes=None, target_sub_labels=None, target_sub_boxes=None,
                pred_rel_logits=None, target_rel_labels=None):
        """
        Performs the matching

        Params:
            pred_logits: Tensor with shape [num_layers, b, num_queries, num_obj_cls + 1]
            pred_boxes: Tensor with shape [num_layers, b, num_queries, 4] cxcywh format
            target_labels: list of Tensor with shape [num_box_per_image,]
            target_boxes: list of Tensor with shape [num_box_per_image, 4] xyxy format
            pred_rel_logits: Tensor with shape [num_layers, b, num_queries, num_rel_cls + 1]
            target_rel_labels: list of Tensor with shape [num_box_per_image,]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        pred_logits = pred_obj_logits[-1] # [b, num_queries, num_obj_cls + 1]
        pred_boxes = pred_obj_boxes[-1] # [b, num_queries, 4]

        bs, num_queries = pred_logits.shape[:2]
        sizes = [box.shape[0] for box in target_obj_labels]

        # We flatten to compute the cost matrices in a batch
        pred_logits = pred_logits.flatten(0, 1).softmax(-1) # [b * num_queries, num_obj_cls + 1]
        pred_boxes = pred_boxes.flatten(0, 1) # [b * num_queries, 4]

        # Also concat the target labels and boxes
        target_labels = torch.cat(target_obj_labels)
        target_boxes = torch.cat(target_obj_boxes)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximat it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_obj_class = -pred_logits[:, target_labels]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))

        # Final cost matrix
        C = self.cost_obj_class * cost_obj_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

        # do the same for sub and rel
        if pred_rel_logits is not None:
            pred_sub_logits = pred_sub_logits[-1]
            pred_sub_boxes = pred_sub_boxes[-1]
            pred_rel_logits = pred_rel_logits[-1]
            pred_sub_logits = pred_sub_logits.flatten(0, 1).softmax(-1)
            pred_sub_boxes = pred_sub_boxes.flatten(0, 1)
            pred_rel_logits = pred_rel_logits.flatten(0, 1).softmax(-1)
            target_sub_labels = torch.cat(target_sub_labels)
            target_sub_boxes = torch.cat(target_sub_boxes)
            target_rel_labels = torch.cat(target_rel_labels)

            cost_sub_class = -pred_sub_logits[:, target_sub_labels]
            cost_rel_class = -pred_rel_logits[:, target_rel_labels]

            cost_sub_bbox = torch.cdist(pred_sub_boxes, target_sub_boxes)
            cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_sub_boxes), box_cxcywh_to_xyxy(target_sub_boxes))

            C += self.cost_obj_class * cost_sub_class + self.cost_bbox * cost_sub_bbox + self.cost_giou * cost_sub_giou
            C += self.cost_rel_class * cost_rel_class

        C = C.view(bs, num_queries, -1).cpu()
        try:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        except:
            print(cost_giou, force=True)
            print(cost_sub_class, force=True)
            print(cost_sub_bbox, force=True)
            print(C, force=True)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):

    matcher = HungarianMatcher(cost_obj_class=cfg.LOSS.COST_OBJ_CLASS,
                               cost_rel_class=cfg.LOSS.COST_REL_CLASS, 
                               cost_bbox=cfg.LOSS.COST_BBOX,
                               cost_giou=cfg.LOSS.COST_GIOU)

    return matcher
