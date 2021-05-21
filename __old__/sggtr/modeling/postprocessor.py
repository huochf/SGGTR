import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms as NMS

from sggtr.structures.bounding_box import BoxList


class PostProcessor(nn.Module):

    def __init__(self, box_format="bcxbcywh", rel_multi_label=False):
        super().__init__()
        assert box_format in ("bcxbcywh", "bx1by1bx2by2", "cxcywh")
        self.box_format = box_format
        self.rel_multi_label = rel_multi_label


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


    def forward(self, outputs, image_sizes):
        """
        Arguments:

        """
        results = {}
        # For DETR model
        if "obj_cls_logits" in outputs and outputs["obj_cls_logits"] is not None:
            pred_logits = outputs["obj_cls_logits"][-1]
            pred_boxes = outputs["obj_box_coord"][-1]
            pred_scores = F.softmax(pred_logits, -1)
            pred_obj_scores, pred_obj_labels = pred_scores.max(dim=2)
            b = pred_obj_labels.shape[0]
            box_lists = []
            for i in range(b):
                keep = pred_obj_labels[i] != 0
                box_list = BoxList(bbox=pred_boxes[i][keep], image_size=image_sizes[i], mode="cxcywh", normalized=True)
                box_list.add_field("pred_labels", pred_obj_labels[i][keep])
                box_list.add_field("pred_scores", pred_obj_scores[i][keep])
                box_list = box_list.convert("xyxy").denormalize_boxes()
                box_lists.append(box_list)

            results["box_lists"] = box_lists

        # For Encoder-Only DETR ...
        if "ref_obj_cls_logits" in outputs and outputs["ref_obj_cls_logits"] is not None:
            pred_logits = outputs["ref_obj_cls_logits"][-1]
            pred_boxes = outputs["ref_obj_box_coord"][-1] # bcxbcywh or bx1by1bx2by2 format
            reference_points = outputs["reference_points"]

            if self.box_format == "bcxbcywh":
                pred_boxes = self.bcxbcywh_to_cxcywh(pred_boxes, reference_points)
            elif self.box_format == "bx1by1bx2by2":
                pred_boxes = self.bx1by1bx2by2_to_cxcywh(pred_boxes, reference_points)

            pred_scores = F.softmax(pred_logits, -1)
            pred_obj_scores, pred_obj_labels = pred_scores[:,:,1:].max(dim=2)
            pred_obj_labels = pred_obj_labels + 1
            _, vis_pred_labels = pred_scores.max(dim=2)
            b = pred_obj_labels.shape[0]
            box_lists = []
            for i in range(b):
                keep = vis_pred_labels[i] != 0
                box_list = BoxList(bbox=pred_boxes[i], image_size=image_sizes[i], mode="cxcywh", normalized=True)
                box_list.add_field("pred_labels", pred_obj_labels[i])
                box_list.add_field("pred_scores", pred_obj_scores[i])
                box_list.add_field("keep", keep)
                box_list = box_list.convert("xyxy").denormalize_boxes()
                box_lists.append(box_list)

            results["box_lists"] = box_lists

        # For RELTR ...
        if "relation_rel_cls_logits" in outputs and outputs["relation_rel_cls_logits"] is not None:
            pred_obj_logits = outputs["relation_obj_cls_logits"][-1]
            pred_sub_logits = outputs["relation_sub_cls_logits"][-1]
            pred_rel_logits = outputs["relation_rel_cls_logits"][-1]
            pred_obj_boxes = outputs["relation_obj_box_coords"][-1]
            pred_sub_boxes = outputs["relation_sub_box_coords"][-1]

            pred_obj_scores = F.softmax(pred_obj_logits, -1)
            pred_sub_scores = F.softmax(pred_sub_logits, -1)
            pred_rel_scores = F.softmax(pred_rel_logits, -1)
            pred_obj_scores, pred_obj_labels = pred_obj_scores.max(dim=2)
            pred_sub_scores, pred_sub_labels = pred_sub_scores.max(dim=2)
            pred_rel_scores, pred_rel_labels = pred_rel_scores.max(dim=2)
            box_lists = []
            b = pred_obj_labels.shape[0]
            for i in range(b):
                keep = (pred_obj_labels[i] != 0) & (pred_sub_labels[i] != 0) & (pred_rel_labels[i] != 0)
                bbox = torch.cat((pred_sub_boxes[i][keep], pred_obj_boxes[i][keep]))
                scores = torch.cat((pred_sub_scores[i][keep], pred_obj_scores[i][keep]))
                labels = torch.cat((pred_sub_labels[i][keep], pred_obj_labels[i][keep]))
                box_list = BoxList(bbox=bbox, image_size=image_sizes[i], mode="cxcywh", normalized=True)
                box_list.add_field("pred_labels", labels)
                box_list.add_field("pred_scores", scores)

                sub_idx = torch.arange(0, keep.shape[0], device=scores.device)[keep]
                obj_idx = sub_idx + keep.shape[0]

                box_list.add_field("rel_pair_idxs", torch.stack((sub_idx, obj_idx), -1))
                box_list.add_field("pred_rel_scores", pred_rel_scores[i][keep])
                box_list.add_field("rel_cls_scores", pred_rel_labels[i][keep])

                box_list = box_list.convert("xyxy").denormalize_boxes()
                box_lists.append(box_list)

            results["rel_box_lists"] = box_lists

        # For SGGTR models
        if "graph_obj_cls_logits" in outputs and outputs["graph_obj_cls_logits"] is not None:
            obj_cls_logits = outputs["graph_obj_cls_logits"][-1]
            rel_cls_logits = outputs["graph_rel_cls_logits"][-1]
            obj_box_coord = outputs["graph_obj_box_coord"][-1]

            if self.rel_multi_label:
                pred_rel_all_logits = rel_cls_logits.sigmoid()
            else:
                pred_rel_all_logits = F.softmax(rel_cls_logits, -1)

            pred_obj_all_scores = F.softmax(obj_cls_logits, -1)
            pred_obj_scores, pred_obj_labels = pred_obj_all_scores[:,:,1:].max(dim=2)
            pred_rel_scores, pred_rel_labels = pred_rel_all_logits[:,:,1:].max(dim=2)
            _, vis_pred_rel_labels = pred_rel_all_logits.max(dim=2)
            pred_obj_labels = pred_obj_labels + 1
            pred_rel_labels = pred_rel_labels + 1
            _, vis_obj_labels = pred_obj_all_scores.max(dim=2)
            _, vis_rel_labels = pred_rel_all_logits.max(dim=2)
            obj_box_lists = []
            b, num_obj_query = pred_obj_labels.shape
            num_rel_cls = pred_rel_all_logits.shape[-1]
            for i in range(b):
                keep = vis_obj_labels[i] != 0
                obj_box_list = BoxList(bbox=obj_box_coord[i], image_size=image_sizes[0], mode="cxcywh", normalized=True)

                box_list_xyxy = obj_box_list.convert('xyxy').denormalize_boxes()
                keep_indices = NMS(box_list_xyxy.bbox, pred_obj_scores[i], 0.5)
                keep_nms = torch.zeros_like(pred_obj_scores[i], dtype=torch.bool)
                keep_nms[keep_indices] = True
                keep = keep & keep_nms

                obj_box_list.add_field("pred_labels", pred_obj_labels[i])
                obj_box_list.add_field("pred_scores", pred_obj_scores[i])
                obj_box_list.add_field("keep", keep)

                label_map = pred_rel_labels[i].reshape((num_obj_query, num_obj_query))
                rel_scores_map = pred_rel_scores[i].reshape((num_obj_query, num_obj_query))
                rel_cls_score_map = pred_rel_all_logits[i].reshape((num_obj_query, num_obj_query, num_rel_cls))
                vis_labels_map = vis_pred_rel_labels[i].reshape((num_obj_query, num_obj_query))

                triplet_score_map = pred_obj_scores[i] * pred_obj_scores[i][None,:] * rel_scores_map
                triplet_score_map[~keep, :] = 0
                triplet_score_map[:, ~keep] = 0
                # triplet_score_map[vis_labels_map == 0] = 0
                vis_triplet_index = (triplet_score_map > 0).nonzero()
                _, idx = torch.sort(triplet_score_map[vis_triplet_index[:, 0], vis_triplet_index[:, 1]], descending=True)
                vis_triplet_index = vis_triplet_index[idx[:20]]

                # For HOI 
                rel_scores_map[~keep, :] = 0
                rel_scores_map[:, ~keep] = 0
                # rel_scores_map[vis_labels_map == 0] = 0

                triplet_index = (rel_scores_map > 0).nonzero() # TODO: how to select threshold ?
                rel_triplet_scores = rel_scores_map[triplet_index[:, 0], triplet_index[:, 1]]
                _, idx = torch.sort(rel_triplet_scores, descending=True)
                idx = idx[:20]
                triplet_index = triplet_index[idx]
                rel_triplet_scores = rel_triplet_scores[idx]
                obj_box_list.add_field("rel_pair_idxs", torch.cat((triplet_index, label_map[triplet_index[:, 0], triplet_index[:, 1]].unsqueeze(-1)), -1))
                obj_box_list.add_field("pred_rel_scores", rel_triplet_scores)
                obj_box_list.add_field("rel_cls_scores", rel_cls_score_map[triplet_index[:, 0], triplet_index[:, 1]])

                obj_box_list.add_field("vis_rel_pair_idxs", torch.cat((vis_triplet_index, label_map[vis_triplet_index[:, 0], vis_triplet_index[:, 1]].unsqueeze(-1)), -1))
                obj_box_list.add_field("vis_pred_rel_scores", triplet_score_map[vis_triplet_index[:, 0], vis_triplet_index[:, 1]])
                box_list = obj_box_list.convert("xyxy").denormalize_boxes()

                obj_box_lists.append(box_list)

            results["graph_obj_box_lists"] = obj_box_lists

        return results


def build_postprocessor(cfg):

    return PostProcessor(box_format=cfg.MODEL.DETECTION_HEADER.BOX_FORMAT, rel_multi_label=cfg.LOSS.REL_MULTI_LABEL)
