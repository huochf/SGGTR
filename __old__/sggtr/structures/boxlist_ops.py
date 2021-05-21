# Modified from Mask R-CNN benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
import torch
import scipy.linalg
from torchvision.ops.boxes import box_area

from .bounding_box import BoxList


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N, M, 2]

    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0]  * wh[:, :, 1] # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # if (boxes1.isnan()).any():
    #     print("detected nan in generalized_box_iou", force=True)

    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), boxes1
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), boxes2
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    eps = 1e-6
    area = wh[:, :, 0] * wh[:, :, 1] + eps

    return iou - (area - union) / area


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both size >= min_size

    Arguments:
        boxlist (BoxList)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlists.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """
    Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
        box1: (BoxList) bounding boxes, sized [N, 4].
        box2: (BoxList) bounding boxes, sized [M, 4].

    Returns:
        (tensor) iou, sized [N, M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    assert boxlist1.normalized == boxlist2.normalized

    if boxlist1.size != boxlist2.size:
        raise RuntimeError("boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area(need_denormalized=False)
    area2 = boxlist2.area(need_denormalized=False)

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2]) # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:]) # [N, M, 2]

    TO_REMOVE = 0

    wh = (rb - lt + TO_REMOVE).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist_union(boxlist1, boxlist2):
    """
    Compute the union region of two set of boxes

    Arguments:
        box1: (BoxList) bounding boxes, sized [N, 4].
        box2: (BoxList) bounding boxes, sized [N, 4].

    Returns:
        (tensor) union, sized [N, 4].
    """
    assert boxlist1.normalized == boxlist2.normalized
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    union_box = torch.cat((
        torch.min(boxlist1.bbox[:, :2], boxlist2.bbox[:, :2]),
        torch.max(boxlist1.bbox[:, 2:], boxlist2.bbox[:, 2:])
    ), dim=1)
    return BoxList(union_box, boxlist1.size, "xyxy", boxlist1.normalized)


def boxlist_intersection(boxlist1, boxlist2):
    """
    Compute the intersection region of two set of boxes

    Arguments:
        box1: (BoxList) bounding boxes, sized [N, 4].
        box2: (BoxList) bounding boxes, sized [N, 4].

    Returns:
        (tensor) intersection, sized [N, 4].
    """
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size and boxlist1.normalized == boxlist2.normalized
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    inter_box = torch.cat((
        torch.max(boxlist1.bbox[:, :2], boxlist2.bbox[:, :2]),
        torch.min(boxlist1.bbox[:, 2:], boxlist2.bbox[:, 2:])
    ), dim=1)
    invalid_bbox = torch.max((inter_box[:, 0] >= inter_box[:, 2]).long(), (inter_box[:, 1] >= inter_box[:, 3]).long())
    inter_box[invalid_bbox > 0] = 0
    return BoxList(inter_box, boxlist1.size, "xyxy", boxlist1.normalized)


# TODO redudant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    normalized = bboxes[0].normalized
    assert all(bbox.normalized == normalized for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) -- fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode, normalized)

    for field in fields:
        if field in bboxes[0].triplet_extra_fields:
            triplet_list = [bbox.get_field(field).numpy() for bbox in bboxes]
            data = torch.from_numpy(scipy.linalg.block_diag(*triplet_list))
            cat_boxes.add_field(field, data, is_tripler=True)
        else:
            data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
            cat_boxes.add_field(field, data)

    return cat_boxes
