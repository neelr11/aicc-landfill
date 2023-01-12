"""Define Logger class for logging information to stdout and disk."""
import json
import os
from os.path import join
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.base import rank_zero_experiment

class RunningWandbLogger(WandbLogger):
    def __init__(self, metric_summary=None, **kwargs):
        super().__init__(**kwargs)
        self._metric_summary = metric_summary
    
    @property
    @rank_zero_experiment
    def experiment(self):
        _experiment = super().experiment
        if getattr(_experiment, "define_metric", None) \
                and self._metric_summary is not None:
            for _metric, _summary in self._metric_summary:
                _experiment.define_metric(_metric, summary=_summary)
        self._experiment = _experiment
        return _experiment

def get_ckpt_dir(save_path, exp_name):
    return os.path.join(save_path, exp_name, "ckpts")


def get_ckpt_callback(save_path, exp_name, monitor):
    ckpt_dir = os.path.join(save_path, exp_name, "ckpts")
    return ModelCheckpoint(dirpath=ckpt_dir,
                           save_top_k=1,
                           verbose=True,
                           monitor=monitor,
                           mode='max')


def get_early_stop_callback(patience, monitor):
    return EarlyStopping(monitor=monitor,
                         patience=patience,
                         verbose=True,
                         mode='max')


def get_logger(logger_type, save_path, exp_name, project_name=None, metric_summary=None):
    if logger_type == 'wandb':
        return RunningWandbLogger(name=exp_name,
                           project=project_name,
                           metric_summary=metric_summary)
    elif logger_type == 'test_tube':
        exp_dir = os.path.join(save_path, exp_name)
        return TestTubeLogger(save_dir=exp_dir,
                              name='lightning_logs',
                              version="0")
    else:
        raise ValueError(f'{logger_type} is not a supported logger.')
