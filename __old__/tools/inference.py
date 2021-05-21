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


def inference(cfg, local_rank, distributed, logger, device):
    model = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: ", n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.SOLVER.LR_BACKBONE,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.SOLVER.LR_DROP)
    max_norm = cfg.SOLVER.CLIP_MAX_NORM

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = Checkpointer(cfg, model, optimizer, lr_scheduler, output_dir, save_to_disk)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, update_schedule=cfg.SOLVER.UPDATE_SCHEDULER_DURING_LOAD)
    arguments = {}
    arguments["last_epoch"] = 0
    arguments.update(extra_checkpoint_data)

    train_data_loader = make_data_loader(cfg, mode='train', is_distributed=distributed, )
    val_data_loaders = make_data_loader(cfg, mode="val", is_distributed=distributed)[0]

    model.eval()
    for iteration, (images, targets_cpu, _) in enumerate(val_data_loaders):
        iteration += 1 # 1, 2, 3, ...
        with torch.no_grad():
            images = images.to(device)
            targets = [target.to(device) for target in targets_cpu]

            results = model(images, targets, visualize=True)

            if is_main_process():
                plot_prediction_results(cfg, images, targets_cpu, results, arguments["last_epoch"], iteration, val_data_loaders.dataset.ind_to_classes, 
                    val_data_loaders.dataset.ind_to_predicates, os.path.join(output_dir, 'train_inference'))


def main():
    parser = argparse.ArgumentParser(description="PyTorch SGGTR Training")
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

    logger = setup_logger("SGGTR", output_dir, get_rank(), filename="inference_log.txt")
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

    inference(cfg, args.gpu, args.distributed, logger, device)


if __name__ == "__main__":
    main()
