
# Modified from QPIC (https://github.com/hitachi-rd-cv/qpic)
# ---------------------------------------------------------------------------------------------
# Modified from HoiTransformer (https://github.com/bbepoch/HoiTransformer)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
import os
import sys
import json
import random
from PIL import Image
import numpy as np

import torch

from sggtr.structures.bounding_box import BoxList
from sggtr.structures.boxlist_ops import boxlist_iou, box_iou, generalized_box_iou


class HICODataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, annotation_dir, transforms=None, flip_aug=False, filter_empty_rels=True, num_im=-1, num_val_im=5000, ):
        """
        Torch dataset for VisualRelationshipDetection dataset
        Parameters:

        """
        # for debug
        # num_im = 2000
        # num_val_im = 12
        assert split in {'train', 'val', 'test'}

        self.split = split
        self.img_dir = os.path.join(img_dir, 'train2015') if split == 'train' else os.path.join(img_dir, 'test2015')
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.flip_aug = flip_aug
        self.filter_empty_rels = filter_empty_rels

        self._valid_obj_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90, )
        self._valid_verb_ids = list(range(0, 118))

        annotation_file = os.path.join(annotation_dir, 'trainval_hico.json') if split == 'train' else os.path.join(annotation_dir, 'test_hico.json')
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        if split == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']):
                        break
                else:
                    self.ids.append(idx)
        else:
            self.ids = list(range(len(self.annotations)))

        if num_im != -1:
            self.ids = self.ids[:num_im]

        with open(os.path.join(annotation_dir, 'coco_and_verb_names.json'), 'r') as f:
            self.ind_to_predicates = json.load(f)['verb_names']

        self.ind_to_classes = [coco_instance_ID_to_name[inds] for inds in self._valid_obj_ids]
        self.categories = coco_instance_ID_to_name

        img_info_file = os.path.join(annotation_dir, 'img_info.json')
        with open(img_info_file, 'r') as f:
            img_info = json.load(f)[0]
            self.img_info = img_info['train'] if split == 'train' else img_info['test']



    def get_img_info(self, index):
        return self.img_info[self.ids[index]]


    def __getitem__(self, index):

        img_anno = self.annotations[self.ids[index]]
        img_file = os.path.join(self.img_dir, img_anno['file_name'])
        img = Image.open(img_file).convert("RGB")

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == "train")
        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


    def get_groundtruth(self, index, flip_img=False, evaluation=False):
        im_info = self.img_info[self.ids[index]]
        im_size = (im_info['width'], im_info['height'])

        img_anno = self.annotations[self.ids[index]]
        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if flip_img:
            new_xmin = im_size[0] - box[:, 2]
            new_xmax = im_size[0] - box[:, 0]
            boxes[:, 0] = new_xmin
            boxes[:, 2] = new_xmax
        classes = torch.tensor(classes, dtype=torch.int64)

        # filter out duplicated boxes
        keep, ind2ind = self._filter_duplicated(boxes, classes)
        boxes = boxes[keep]
        classes = classes[keep]

        num_box = boxes.shape[0]
        rel_triplets = []
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        relation_cls_binary_map = torch.zeros((num_box, num_box, 118), dtype=torch.int64)
        for hoi in img_anno['hoi_annotation']:
            sub, obj = ind2ind[hoi['subject_id']], ind2ind[hoi['object_id']]
            rel_triplets.append([sub, obj, self._valid_verb_ids.index(hoi['category_id'])])
            relation_cls_binary_map[int(sub), int(obj), int(hoi['category_id'])] = 1
            try:
                if relation_map[int(sub), int(obj)] > 0:
                    if random.random() > 0.5:
                        relation_map[int(sub), int(obj)] = int(hoi['category_id'])
                else:
                    relation_map[int(sub), int(obj)] = int(hoi['category_id'])
            except:
                print(img_anno, force=True)
                print(keep, force=True)
                print(ind2ind, force=True)
        no_relation = relation_cls_binary_map.sum(-1)
        if len(relation_cls_binary_map[no_relation == 0]) > 0:
            relation_cls_binary_map[no_relation == 0][0] = 1

        rel_triplets = torch.as_tensor(rel_triplets, dtype=torch.int64)

        target = BoxList(boxes, im_size, 'xyxy', False)
        target.add_field('labels', classes)
        target.add_field("relation_tuple", rel_triplets, is_triplet=True)
        target.add_field("relation", relation_map, is_triplet=True)
        target.add_field("relation_cls_binary_map", relation_cls_binary_map, is_triplet=True)

        return target


    def _filter_duplicated(self, boxes, classes):
        ind2ind = []

        # iou = box_iou(boxes, boxes)[0]
        iou = generalized_box_iou(boxes, boxes)
        for i in range(iou.shape[0]):
            iou[i, i:] = 0

        max_iou, father_idx = iou.max(1)
        duplicated = (max_iou > 0.5) & (classes == classes[father_idx])

        keep = ~duplicated
        keep_num = keep.sum()

        identity = torch.arange(0, boxes.shape[0])
        identity[keep] = torch.arange(0, keep_num)

        for i, is_duplicated in enumerate(duplicated):
            if is_duplicated:
                identity[i] = identity[father_idx[i]]

        return keep, identity


    def __len__(self):
        return len(self.ids)


coco_classes_originID = {
    '__background__': 0,
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}


coco_instance_ID_to_name = {
    0: "__background__",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}
