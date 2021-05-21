# Modified from Scene-Graph-Benchmark (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
# ----------------------------------------------------------------------------------------------------
from .vg_eval import do_vg_evaluation


def vg_evaluation(cfg, epoch, dataset, predictions, output_folder, logger, iou_types, **_):
    return do_vg_evaluation(
        cfg=cfg,
        epoch=epoch,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
    )
