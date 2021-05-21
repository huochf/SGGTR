# Modified from Scene-Graph-Benchmark (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
# ----------------------------------------------------------------------------------------------------
from .hico_eval import do_hico_evaluation


def hico_evaluation(cfg, epoch, dataset, predictions, output_folder, logger, iou_types, **_):
    return do_hico_evaluation(
        cfg=cfg,
        epoch=epoch,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
    )
