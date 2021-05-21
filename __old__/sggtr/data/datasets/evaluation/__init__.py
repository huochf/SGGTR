# Modified from Scene-Graph-Benchmark (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
# ----------------------------------------------------------------------------------------------------
from sggtr.data import datasets

from .vg import vg_evaluation
from .vrd import vrd_evaluation
from .hico import hico_evaluation


def evaluate(cfg, epoch, dataset, predictions, output_folder, logger, **kwargs):

    args = dict(
        cfg=cfg, epoch=epoch, dataset=dataset, predictions=predictions, output_folder=output_folder, logger=logger, **kwargs)
    if isinstance(dataset, datasets.VGDataset):
        return vg_evaluation(**args)
    elif isinstance(dataset, datasets.VRDDataset):
        return vrd_evaluation(**args)
    elif isinstance(dataset, datasets.HICODataset):
        return hico_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
