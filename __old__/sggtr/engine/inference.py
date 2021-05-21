# Modified from Mask R-CNN benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
import os
import logging
import pickle
from tqdm import tqdm

import torch

from sggtr.utils.comm import get_world_size, synchronize, all_gather, is_main_process
from sggtr.utils.timer import Timer, get_time_str
from sggtr.data.datasets.evaluation import evaluate


def compute_on_dataset(epoch, model, data_loader, device, synchronize_gather=True, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()

            outputs = model(images.to(device), targets)
            if timer:
                if not device != cpu_device:
                    torch.cuda.synchronize()
                timer.toc()
            # for o_k, o_v in outputs.items():
            #     o_v = [o.to(cpu_device) for o in o_v]
            #     outputs[o_k] = o_v
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather(
                {
                    field: {img_id: result.to(cpu_device) for img_id, result in zip(image_ids, output)}
                    for field, output in outputs.items()
                }
            )
            if is_main_process():
                for p in multi_gpu_predictions:
                    for field, output in p.items():
                        if field not in results_dict:
                            results_dict[field] = {}
                        results_dict[field].update(output)
        else:
            for field, output in p.items():
                if field not in results_dict:
                    results_dict[field] = {}
                results_dict[field].update(output)
    torch.cuda.empty_cache()
    return results_dict


def inference(cfg,
              epoch,
              model,
              data_loader,
              dataset_name,
              device="cuda",
              iou_types=("bbox",),
              box_only=False,
              expected_results=(),
              expected_results_sigma_tol=4,
              output_folder=None,
              logger=None,
):
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and  \
                                 output_folder is not None and \
                                 os.path.exists(os.path.join(output_folder, str(epoch) +  "_eval_results.pytorch"))
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("SGGTR.inference")
    dataset = data_loader.dataset
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if load_prediction_from_cache:
        predictions = torch.load(os.path.join(output_folder, str(epoch) +  "_eval_results.pytorch"), map_location=torch.device("cpu"))['predictions']
    else:
        predictions = compute_on_dataset(epoch, model, data_loader, device, synchronize_gather=cfg.TEST.SYNC_GATHER, timer=inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.SYNC_GATHER)

    if not is_main_process():
        return -1.0

    # if output_folder is not None and not load_prediction_from_cache:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol)

    return evaluate(cfg=cfg, epoch=epoch,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    **extra_args)


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True):
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        assert False
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
    
    # convert a dict where the key is the index in a list
    for field, prediction in predictions.items():
        print(field)
        print(len(prediction))
        image_ids = list(sorted(prediction.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("SGGTR.inference")
            logger.warning(
                "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions[field] = [predictions[field][i] for i in image_ids]
    print("accumulate done...")
    return predictions
