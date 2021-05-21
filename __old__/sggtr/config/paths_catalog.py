# Modified from Mask R-CNN benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
"""Centralized catalog of paths."""

import os
import copy


class DatasetCatalog(object):
    DATA_DIR = "/public/sist/home/FWS20210094/data/"
    DATASETS = {
        "VG_stanford_filtered_with_attribute": {
            "img_dir": "VG/images/VG_100K",
            "roidb_file": "VG/annotations/scene_graph/vg/VG-SGG-with-attri.h5",
            "dict_file": "VG/annotations/scene_graph/vg/VG-SGG-dicts-with-attri.json",
            "image_file": "VG/annotations/scene_graph/vg/image_data.json",
        },
        "VRD": {
            "img_dir": "VRD/sg_dataset/sg_dataset/",
            "annotation_dir": "VRD/json_dataset/",
        },
        "HICO": {
            "img_dir": "HICO-DET/hico_20160224_det/images", 
            "annotation_dir": "HICO-DET/annotations/", 
        }
    }


    @staticmethod
    def get(name, cfg):
        if "VG" in name:
            # name should be something like VG_stanford_filtered_train
            p = name.rfind("_")
            name, split = name[:p], name[p+1:]
            assert name in DatasetCatalog.DATASETS and split in {"train", "val", "test"}
            data_dir = DatasetCatalog.DATA_DIR
            args = copy.deepcopy(DatasetCatalog.DATASETS[name])
            for k, v in args.items():
                args[k] = os.path.join(data_dir, v)
            args["split"] = split
            args["filter_non_overlap"] = False
            args["filter_empty_rels"] = True
            args["flip_aug"] = cfg.MODEL.FLIP_AUG
            args["custom_eval"] = "" # cfg.TEST.CUSTUM_EVAL
            args["custom_path"] = "" # cfg.TEST.CUSTUM_PATH
            return dict(
                factory="VGDataset",
                args=args,
            )
        if "VRD" in name:
            p = name.rfind("_")
            name, split = name[:p], name[p + 1: ]
            assert name in DatasetCatalog.DATASETS and split in {"train", "val", "test"}
            data_dir = DatasetCatalog.DATA_DIR
            args = copy.deepcopy(DatasetCatalog.DATASETS[name])
            for k, v in args.items():
                args[k] = os.path.join(data_dir, v)
            args["split"] = split
            args["filter_empty_rels"] = True
            args["flip_aug"] = cfg.MODEL.FLIP_AUG
            return dict(
                factory="VRDDataset",
                args=args,
            )

        if "HICO" in name:
            p = name.rfind("_")
            name, split = name[:p], name[p + 1: ]
            assert name in DatasetCatalog.DATASETS and split in {"train", "val", "test"}
            data_dir = DatasetCatalog.DATA_DIR
            args = copy.deepcopy(DatasetCatalog.DATASETS[name])
            for k, v in args.items():
                args[k] = os.path.join(data_dir, v)
            args["split"] = split
            args["filter_empty_rels"] = True
            args["flip_aug"] = cfg.MODEL.FLIP_AUG
            return dict(
                factory="HICODataset",
                args=args,
            )

        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):

    @staticmethod
    def get(name):
        raise NotImplementedError()
