import os
import re
import warnings

import torch
import numpy as np
from tensorboardX import SummaryWriter
from torchfitter.callbacks.base import Callback
from torchfitter.conventions import ParamsDict


class ModelCheckpoint(Callback):
    """Callback to save the best pytorch model and optimizer's state.
    Args:
        filepath: string or `PathLike`, path to save the model file.
            `filepath` can contain named formatting options,
            which will be filled the value of `epoch` and keys in `logs`
            (passed in `on_epoch_end`).
            The `filepath` name needs to end with `".pth"`
            For example:
            `filepath` is `"{epoch:02d}-{val_loss:.2f}.pth"`.
        monitor: The metric name to monitor. type(TorchMetric)__name__.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
            displays messages when the callback takes an action.
        save_best_only: if `save_best_only=True`, it only saves when the model
            is considered the "best" and the latest best model according to the
            quantity monitored will not be overwritten.
        mode: one of {`"auto"`, `"min"`, `"max"`}. If `save_best_only=True`, the
            decision to overwrite the current save file is made based on either
            the maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `"max"`, for `val_loss` this should be
            `"min"`, etc. In `"auto"` mode, the mode is set to `"max"`.
        save_optimizer: if True, then only the optimizre's state will be saved
        save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. 
        initial_value_threshold: Floating point initial "best" value of the
            metric to be monitored. 
    """

    def __init__(
        self,
        ckpt_path,
        verbose=0,
        monitor=None,
        save_best=True,
        save_optimizer=True,
        mode="auto",
        save_freq=1,
        initial_value_threshold=None,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.ckpt_path = ckpt_path
        self.save_best = save_best
        self.save_optimizer = save_optimizer
        self.save_freq = save_freq
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self.best = initial_value_threshold

        if self.ckpt_path and not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"ModelCheckpoint mode '{mode}' is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        else: #mode=="max"
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf

        if not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected save_freq are 'epoch' or integer values"
            )

    def on_epoch_end(self, params_dict: dict):
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        if self.save_best:
            current = params_dict[ParamsDict.EPOCH_HISTORY][self.monitor]["validation"][-1]
            if current is None:
                warnings.warn(
                    f"Can save best model only with {self.monitor} available, skipping.",
                    stacklevel=2,
                )
            else:
                if self.monitor_op(current, self.best):
                    filepath = os.path.join(self.ckpt_path, "best_val.pth")
                    if self.verbose > 0:
                        print(
                            f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model to {filepath}"
                        )
                    self.best = current
                    checkpoint = {}
                    checkpoint['epoch'] = epoch
                    checkpoint['state_dict'] = params_dict[ParamsDict.MODEL].state_dict().copy()
                    if self.save_optimizer:
                        checkpoint['optimizer'] = params_dict[ParamsDict.OPTIMIZER].state_dict().copy()
                    torch.save(checkpoint, filepath)

        if ((epoch%self.save_freq) == 0):
            filepath = os.path.join(self.ckpt_path, "epoch{:03d}.pth".format(epoch))
            print(
                f"\nEpoch {epoch}: saving model to {filepath}"
            )
            checkpoint = {}
            checkpoint['epoch'] = epoch
            checkpoint['state_dict'] = params_dict[ParamsDict.MODEL].state_dict().copy()
            if self.save_optimizer:
                checkpoint['optimizer'] = params_dict[ParamsDict.OPTIMIZER].state_dict().copy()
            torch.save(checkpoint, filepath)

    def on_fit_end(self, params_dict: dict) -> None:
        checkpoint = {}
        checkpoint['epoch'] =  params_dict[ParamsDict.EPOCH_NUMBER]
        checkpoint['state_dict'] = params_dict[ParamsDict.MODEL].state_dict().copy()
        if self.save_optimizer:
            checkpoint['optimizer'] = params_dict[ParamsDict.OPTIMIZER].state_dict().copy()
        torch.save(checkpoint, os.path.join(self.ckpt_path, "last.pth"))


class TensorBoardLogger(Callback):
    """
    This callback displays a progress bar to report the state of the training
    process: on each epoch, a new bar will be created and stacked below the
    previous bars.

    Metrics are logged using the library logger.

    Parameters
    ----------
    log_lr : str, directory to log the results.
    """

    def __init__(self, log_dir:str, metric:str):
        super(TensorBoardLogger, self).__init__()
        self.tf_writer = SummaryWriter(log_dir=log_dir)
        self.metric = metric

    def on_epoch_end(self, params_dict: dict) -> None:
        epoch = params_dict[ParamsDict.EPOCH_NUMBER]
        self.tf_writer.add_scalar('loss/train', params_dict[ParamsDict.EPOCH_HISTORY][ParamsDict.LOSS]["train"][-1], epoch)
        self.tf_writer.add_scalar('acc/train', params_dict[ParamsDict.EPOCH_HISTORY][self.metric]["train"][-1], epoch)
        self.tf_writer.add_scalar('loss/val', params_dict[ParamsDict.EPOCH_HISTORY][ParamsDict.LOSS]["validation"][-1], epoch)
        self.tf_writer.add_scalar('acc/val', params_dict[ParamsDict.EPOCH_HISTORY][self.metric]["validation"][-1], epoch)
        self.tf_writer.add_scalar('lr', params_dict[ParamsDict.EPOCH_HISTORY][ParamsDict.HISTORY_LR][-1], epoch)