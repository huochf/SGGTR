# Modified from Mask R-CNN benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ---------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
import sys
sys.path.append('/p300/projects/scene_graph/sggtr')
import os
import argparse
import random
import time
import datetime
import matplotlib
matplotlib.use('Agg') 

import numpy as np 
import torch

from sggtr.config import cfg
from sggtr.utils.miscellaneous import mkdir, save_config
from sggtr.utils.comm import init_distributed_mode, get_rank, synchronize, is_main_process
from sggtr.utils.logger import setup_logger
from sggtr.utils.metric_logger import MetricLogger, reduce_loss_dict
from sggtr.utils.checkpoint import Checkpointer
from sggtr.utils.visualize import plot_prediction_results
from sggtr.modeling.xxxtr import build_model
from sggtr.data import make_data_loader
from sggtr.engine.inference import inference


def evaluate(cfg, local_rank, distributed, logger, device):
    model = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: ", n_parameters)

    output_dir = cfg.OUTPUT_DIR

    # val_data_loaders = [make_data_loader(cfg, mode="train", is_distributed=distributed)]
    val_data_loaders = make_data_loader(cfg, mode="val", is_distributed=distributed)

    for checkpoint_name in sorted(os.listdir(output_dir)):
        if checkpoint_name != 'model_068.pth':
            continue
        if os.path.splitext(checkpoint_name)[-1] == ".pth":
            checkpoint_path = os.path.join(output_dir, checkpoint_name)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model"])
            epoch = checkpoint["last_epoch"]
            logger.info("Begin to evaluate model: " + checkpoint_path)
            logger.info(epoch)
            # if epoch < 44:
            #     continue
            run_val(cfg, epoch, model, val_data_loaders, distributed, logger, device)


def run_val(cfg, epoch, model, val_data_loaders, distributed, logger, device):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()

    iou_types = ("relations", "bbox") if cfg.MODEL.DETECTOR.TYPE in ["RELTR", "SGGTR", "SGGTRSGGOnly", "SGGTR-DETR", "DETRBasedSGGTR", "SGGTRWithEXKnowledge"] else ("bbox", )

    # iou_types = ("relations", )
    dataset_names = cfg.DATASETS.VAL
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        inference(
            cfg, epoch,
            model, val_data_loader,
            dataset_name=dataset_name,
            device=device,
            iou_types=iou_types,
            output_folder=None,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch SGGTR Evaluation")
    parser.add_argument("--config-file", default="/p300/projects/scene_graph/sggtr/configs/detr.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--skip-test", dest="skip_test", help="Do not test the final model", action="store_true")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0)
    args = parser.parse_args()

    init_distributed_mode(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("SGGTR", output_dir, get_rank(), filename="eval_log.txt")
    logger.info("Using {} GPUs".format(args.world_size))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config: \n{}".format(cfg))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    evaluate(cfg, args.gpu, args.distributed, logger, device)


if __name__ == "__main__":
    main()
