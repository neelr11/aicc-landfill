import torch

from .classification import ClassificationTask
from .detection import DetectionTask
from .util import get_ckpt_callback, get_early_stop_callback
from .util import get_logger


def get_task(args):
    if args.get("task", "classification") == "classification":
        return ClassificationTask(args)
    else:
        return DetectionTask(args)

def load_task(ckpt_path, **kwargs):
    args = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']
    if args.get("task", "classification") == "classification":
        task = ClassificationTask
    else:
        task = DetectionTask
    return task.load_from_checkpoint(ckpt_path, **kwargs)
