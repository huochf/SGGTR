import os
import sys
import json
import random
from PIL import Image
import numpy as np

import torch

from sggtr.structures.bounding_box import BoxList
from sggtr.structures.boxlist_ops import boxlist_iou


class VRDDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, annotation_dir, transforms=None, flip_aug=False, filter_empty_rels=True, num_im=-1, num_val_im=5000, ):
        """
        Torch dataset for VisualRelationshipDetection dataset
        Parameters:

        """
        # for debug
        # num_im = 500
        # num_val_im = 4
        assert split in {'train', 'val', 'test'}

        self.split = split
        self.img_dir = os.path.join(img_dir, 'sg_train_images') if split == 'train' else os.path.join(img_dir, 'sg_test_images')
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.flip_aug = flip_aug
        self.filter_empty_rels = filter_empty_rels

        self.ind_to_classes, self.ind_to_predicates = load_info(annotation_dir)
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.img_file_names, self.obj_boxes, self.obj_cls, self.rel_triplets = load_graphs(split, annotation_dir, self.img_dir, num_im, num_val_im, filter_empty_rels)

        self.img_info = self.load_img_info()


    def load_img_info(self, ):
        img_info = []

        for img_file in self.img_file_names:
            im_size = Image.open(os.path.join(self.img_dir, img_file)).size
            img_info.append({'width': im_size[0], 'height': im_size[1]})

        return img_info


    def get_img_info(self, index):
        return self.img_info[index]


    def __getitem__(self, index):

        img = Image.open(os.path.join(self.img_dir, self.img_file_names[index])).convert("RGB")
        w, h = img.size[0], img.size[1]

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == "train")
        target = self.get_groundtruth(index, (w, h), flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


    def get_groundtruth(self, index, im_size=-1, flip_img=False, evaluation=False):
        if im_size == -1:
            img_info = self.img_info[index]
            im_size = (img_info['width'], img_info['height'])
        boxes = np.array(self.obj_boxes[index]) # [yyxx]
        boxes = np.stack([boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]], -1)
        box_cls = self.obj_cls[index]
        triplets = self.rel_triplets[index]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        box_cls = torch.tensor(box_cls, dtype=torch.int64)
        triplets = torch.tensor(triplets, dtype=torch.int64)
        target = BoxList(boxes, im_size, 'xyxy', False) # xyxy
        target.add_field("labels", box_cls)
        target.add_field("relation_tuple", triplets, is_triplet=True)

        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(triplets.shape[0]):
            if triplets[i, 2] > 0:
                relation_map[int(triplets[i, 0]), int(triplets[i, 1])] = int(triplets[i, 2])

        target.add_field("relation", relation_map, is_triplet=True)

        return target      


    def __len__(self):
        return len(self.img_file_names)



def load_info(annotation_dir):
    with open(os.path.join(annotation_dir, 'objects.json'), 'r') as f:
        object_list = json.load(f)
    with open(os.path.join(annotation_dir, 'predicates.json'), 'r') as f:
        predicate_list = json.load(f)

    object_list.insert(0, '__background__')
    predicate_list.insert(0, '__background__')

    return object_list, predicate_list


def load_graphs(split, annotation_dir, img_dir, num_im, num_val_im, filter_empty_rels):

    annotation_file = "annotations_train.json" if split == "train" else "annotations_test.json"
    with open(os.path.join(annotation_dir, annotation_file), 'r') as f:
        annotation_dict = json.load(f)
    img_file_names, obj_boxes, obj_cls, rel_triplets = [], [], [], []
    for file_name, rel_list in annotation_dict.items():
        if not os.path.exists(os.path.join(img_dir, file_name)):
            print("file " + os.path.join(img_dir, file_name) + " not exists.", force=True)
            continue

        if len(rel_list) == 0:
            continue

        obj_boxes_per_image, obj_cls_per_image, rel_triplet_per_image = [], [], []

        for rel_triplet in rel_list:
            subject_dict = rel_triplet['subject']
            object_dict = rel_triplet['object']
            pred_labels = rel_triplet['predicate'] + 1 # 0 for '__background__'

            sub_index = insert_into_box_list(obj_boxes_per_image, obj_cls_per_image, subject_dict)
            obj_index = insert_into_box_list(obj_boxes_per_image, obj_cls_per_image, object_dict)

            rel_triplet_per_image.append([sub_index, obj_index, pred_labels])

        img_file_names.append(file_name)
        obj_boxes.append(obj_boxes_per_image)
        obj_cls.append(obj_cls_per_image)
        rel_triplets.append(rel_triplet_per_image)

    return img_file_names[:num_im], obj_boxes[:num_im], obj_cls[:num_im], rel_triplets[:num_im]


def insert_into_box_list(obj_boxes_per_image, obj_cls_per_image, obj_dict):
    for i, (obj_box, obj_cls) in enumerate(zip(obj_boxes_per_image, obj_cls_per_image)):
        if obj_box == obj_dict['bbox'] and obj_cls == obj_dict['category'] + 1:
            return i
    obj_boxes_per_image.append(obj_dict['bbox'])
    obj_cls_per_image.append(obj_dict['category'] + 1)
    return len(obj_boxes_per_image) - 1

