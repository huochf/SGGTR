# Modified from Mask R-CNN benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
import os

from yacs.config import CfgNode as CN

# ----------------------------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# ----------------------------------------------------------------------------------------------
# Whether an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# ----------------------------------------------------------------------------------------------
# Config definition
# ----------------------------------------------------------------------------------------------

_C = CN()


# ----------------------------------------------------------------------------------------------
# MODEL options
# ----------------------------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.FLIP_AUG = False

_C.MODEL.PRETRAINED = '' # '/p300/projects/scene_graph/sggtr/pretrained_models/detr-r50-coco.pth'

_C.MODEL.WEIGHT = ""

_C.MODEL.BACKBONE = CN()
# Name of the convolutional backbone to use
_C.MODEL.BACKBONE.NAME = "resnet50"
# If true, we replace stride with dilation in the last convolutional block (DC5)
_C.MODEL.BACKBONE.DILATION = False


_C.MODEL.POS_ENCODER = CN()
_C.MODEL.POS_ENCODER.TYPE = 'sine' # 'sine' or 'learned'


_C.MODEL.DETECTOR = CN()
_C.MODEL.DETECTOR.TYPE = "DETR" # "EncoderOnlyDETR", "DecoderOnlyDETR", "RELTR", "SGGTR", "SGGTR-DETR"

# Size of the embeddings (dimension of the transformer)
_C.MODEL.DETECTOR.D_MODEL = 256

_C.MODEL.DETECTOR.NHEAD = 8
_C.MODEL.DETECTOR.DIM_FEEDFORWARD = 2048
_C.MODEL.DETECTOR.DROPOUT = 0.1
_C.MODEL.DETECTOR.ENC_LAYERS = 6
_C.MODEL.DETECTOR.DEC_LAYERS = 6
_C.MODEL.DETECTOR.NORMALIZE_BEFORE = True

_C.MODEL.DETECTOR.NUM_OBJ_QUERY = 100

# For Decoder-only-DETR
_C.MODEL.DETECTOR.QUERY_POS_TYPE = 'sine'

# For SGGTR only
_C.MODEL.DETECTOR.NUM_HIDDEN_REL_QUERY = 100


_C.MODEL.DETECTION_HEADER = CN()

_C.MODEL.DETECTION_HEADER.NUM_OBJ_CLASS = 150
_C.MODEL.DETECTION_HEADER.NUM_REL_CLASS = 50

# For SGGTR only
_C.MODEL.DETECTION_HEADER.OBJ_HEADER_ON = True
_C.MODEL.DETECTION_HEADER.GRAPH_HEADER_ON = True

_C.MODEL.DETECTION_HEADER.BOX_FORMAT = "cxcywh" # "bcxbcywh", "bx1by1bx2by2"

_C.MODEL.DETECTOR.WORD_EMBEDDING_FILE = '/public/sist/home/FWS20210094/projects/scene_graph/SGGTR/data/vg_exknowledge/emb_mtx.pkl'
_C.MODEL.DETECTOR.CORRELATION_FILE = '/public/sist/home/FWS20210094/projects/scene_graph/SGGTR/data/vg_exknowledge/all_edges.pkl'


# ----------------------------------------------------------------------------------------------
# LOSS options
# ----------------------------------------------------------------------------------------------
_C.LOSS = CN()

_C.LOSS.COST_OBJ_CLASS = 1
_C.LOSS.COST_REL_CLASS = 1
_C.LOSS.COST_BBOX = 5
_C.LOSS.COST_GIOU = 2

_C.LOSS.CLS_COEF = 1
_C.LOSS.REL_CLS_COEF = 100
_C.LOSS.BBOX_LOSS_COEF = 5
_C.LOSS.GIOU_LOSS_COEF = 2
_C.LOSS.AUX_LOSS_COEF = 0.1
_C.LOSS.KNOWLEDGE_LOSS_COEF = 0.1

_C.LOSS.SGG_ONLY = False
_C.LOSS.OD_ONLY = False

_C.LOSS.NO_OBJ_CLS_WEIGHT = 0.1
_C.LOSS.NO_REL_CLS_WEIGHT = 0.1

_C.LOSS.WITH_AUX_LOSS = True
_C.LOSS.REL_MULTI_LABEL = False
_C.LOSS.EXKNOWLEDGE_LOSS = False

_C.LOSS.WITH_ADDITIONAL_REL_LOSS = False
_C.LOSS.ADDITION_NO_REL_CLS_WEIGHT = 0.1


# ----------------------------------------------------------------------------------------------
# INPUT options
# ----------------------------------------------------------------------------------------------
_C.INPUT = CN()

_C.INPUT.NO_AUG = True
_C.INPUT.USE_DETR_AUG = False
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800, ) # (800, )
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406] # [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225] # [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False # True

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0


# ----------------------------------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ("VG_stanford_filtered_with_attribute_train", )
# List of the dataset names for val, as present in paths_catalog.py
# Note that except dataset names, all remaining val configs reuse those of test
_C.DATASETS.VAL = ("VG_stanford_filtered_with_attribute_val", )
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ("VG_stanford_filtered_with_attribute_test", )


# ----------------------------------------------------------------------------------------------
# DataLoader options
# ----------------------------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, add landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ----------------------------------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = None # 40000

_C.SOLVER.LR_BACKBONE = 1e-5
_C.SOLVER.LR = 1e-4
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.MAX_EPOCHS = 300
_C.SOLVER.LR_DROP = 200
_C.SOLVER.CLIP_MAX_NORM = 0.1

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 8

_C.SOLVER.UPDATE_SCHEDULER_DURING_LOAD = True

# How long we print logs per iteration
_C.SOLVER.LOG_PERIOD = 10
# How long we visualize train images per iteration
_C.SOLVER.VISUALIZE_PERIOD = 1000
# How long we evaluate model during training per epoch
_C.SOLVER.VAL_PERIOD = 10
# How long we save checkpoint per epoch
_C.SOLVER.CHECKPOINT_PERIOD = 10

_C.SOLVER.CLIP_MAX_NORM = 0.1

# ----------------------------------------------------------------------------------------------
# Test options
# ----------------------------------------------------------------------------------------------
_C.TEST = CN()

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8


# ----------------------------------------------------------------------------------------------
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ----------------------------------------------------------------------------------------------
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

_C.TEST.ALLOW_LOAD_FROM_CACHE = False

_C.TEST.CUSTOM_EVAL = False

_C.TEST.SYNC_GATHER = True


_C.TEST.RELATION = CN()

_C.TEST.RELATION.MULTIPLE_PREDS = False
_C.TEST.RELATION.IOU_THRESHOLD = 0.5

# ----------------------------------------------------------------------------------------------
# Misc options
# ----------------------------------------------------------------------------------------------
_C.OUTPUT_DIR = "/public/sist/home/FWS20210094/projects/scene_graph/SGGTR/outputs"

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")




