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


def train(cfg, local_rank, distributed, logger, device):
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

    # for params in optimizer.param_groups:
    #     params['lr'] *= 0.1

    train_data_loader = make_data_loader(cfg, mode='train', is_distributed=distributed, )
    val_data_loaders = make_data_loader(cfg, mode="train", is_distributed=distributed)

    start_epoch = arguments["last_epoch"]
    logger.info("Start training")
    max_epoch = cfg.SOLVER.MAX_EPOCHS

    meters = MetricLogger(delimiter=" ")
    start_training_time = time.time()
    end = time.time()
    iter_per_epoch = len(train_data_loader)
    for epoch_idx in range(start_epoch, max_epoch):
        epoch_idx += 1 # 1, 2, 3, ...
        model.train()
        arguments["last_epoch"] = epoch_idx
        for iteration, (images, targets_cpu, _) in enumerate(train_data_loader):
            iteration += 1 # 1, 2, 3, ...
            visualize = iteration % cfg.SOLVER.VISUALIZE_PERIOD == 0
            data_time = time.time() - end

            images = images.to(device)
            targets = [target.to(device) for target in targets_cpu]

            loss_dict, results = model(images, targets, visualize=visualize)

            losses = sum(loss * get_loss_weight(cfg, name) for name, loss in loss_dict.items())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss * get_loss_weight(cfg, name) for name, loss in loss_dict_reduced.items())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * ( (max_epoch - epoch_idx) * iter_per_epoch + iter_per_epoch - iteration )
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % cfg.SOLVER.LOG_PERIOD == 0 or iteration == iter_per_epoch:
                if is_main_process():
                    logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "epoch: {epoch} / {max_epoch}",
                                "iter: {iter} / {max_iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            epoch=epoch_idx,
                            max_epoch=max_epoch,
                            iter=iteration,
                            max_iter=iter_per_epoch,
                            meters=str(meters),
                            lr=optimizer.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )

            if visualize:
                if is_main_process():
                    plot_prediction_results(cfg, images, targets_cpu, results, epoch_idx, iteration, train_data_loader.dataset.ind_to_classes, 
                        train_data_loader.dataset.ind_to_predicates, output_dir)

        lr_scheduler.step()

        if epoch_idx % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            run_val(cfg, model, val_data_loaders, distributed, logger, device=device)
        if epoch_idx % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpointer.save("model_{:03d}".format(epoch_idx), **arguments)

        if epoch_idx == max_epoch:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it".format(
            total_time_str, total_training_time / ((max_epoch - start_epoch) * iter_per_epoch)
        )
    )

    return model


def get_loss_weight(cfg, name):
    factor = 1.0
    if 'rel' in name:
        factor = factor * cfg.LOSS.REL_CLS_COEF
    if 'box' in name:
        factor = factor * cfg.LOSS.BBOX_LOSS_COEF
    if 'giou' in name:
        factor = factor * cfg.LOSS.GIOU_LOSS_COEF

    if 'aux' in name:
        factor = factor * cfg.LOSS.AUX_LOSS_COEF

    if cfg.LOSS.SGG_ONLY:
        if 'sgg' not in name:
            factor = 0.0

    if cfg.LOSS.OD_ONLY:
        if 'sgg' in name:
            factor = 0.0

    return factor


def run_val(cfg, model, val_data_loaders, distributed, logger, device):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()

    iou_types = ("relations", ) if cfg.MODEL.DETECTOR.TYPE in ["RELTR", "SGGTR", "SGGTR-DETR", "DETRBasedSGGTR"] else ("bbox", )

    dataset_names = cfg.DATASETS.VAL
    # for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
    #     inference(
    #         cfg,
    #         model, val_data_loader,
    #         dataset_name=dataset_name,
    #         device=device,
    #         iou_types=iou_types,
    #         output_folder=None,
    #     )
    #     synchronize()

    inference(
        cfg,
        model, val_data_loaders,
        dataset_name="test",
        device=device,
        iou_types=iou_types,
        output_folder=None,
    )
    synchronize()


def run_test(cfg, model, distributed, logger, device):
    pass


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

    logger = setup_logger("SGGTR", output_dir, get_rank())
    logger.info("Using {} GPUs".format(args.world_size))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config: \n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yaml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = train(cfg, args.gpu, args.distributed, logger, device)

    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger, device)


if __name__ == "__main__":
    main()
