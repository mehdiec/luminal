import numpy as np
import pytorch_lightning as pl
from typing import List, Optional, Dict, Callable, Sequence, Tuple, Union
from torch import Tensor, nn
from torch.optim import Optimizer, AdamW
import torch
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
from torchmetrics import Metric, MetricCollection
from pathaia.util.basic import ifnone

from src.losses import get_loss_name


def get_scheduler_func(
    name: str, total_steps: int, lr: float
) -> Callable[[Optimizer], Optional[Dict[str, Union[str, _LRScheduler]]]]:
    r"""
    Get a function that given an optimizer, returns the corresponding scheduler dict
    formatted for `PytorchLightning <https://www.pytorchlightning.ai/>`_.

    Args:
        name: name of the scheduler. Can either be "one-cycle", "cosine-anneal",
            "reduce-on-plateau" or "none".
        total_steps: total number of training iterations, only useful for "one-cycle" or
            "cosine-anneal".
        lr: baseline learning rate.

    Returns:
        Function that takes an optimizer as input and returns a scheduler dict formatted
        for PytorchLightning.
    """

    def scheduler_func(opt: Optimizer):
        if name == "one-cycle":
            sched = {
                "scheduler": OneCycleLR(opt, lr, total_steps=total_steps),
                "interval": "step",
            }
        elif name == "cosine-anneal":
            sched = {
                "scheduler": CosineAnnealingLR(opt, total_steps),
                "interval": "step",
            }
        elif name == "reduce-on-plateau":
            sched = {
                "scheduler": ReduceLROnPlateau(opt, patience=3),
                "interval": "epoch",
                "monitor": "val_loss",
            }
        else:
            return None
        return sched

    return scheduler_func


class BasicClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        lr: float,
        wd: float,
        scheduler_func: Optional[Callable] = None,
        metrics: Optional[Sequence[Metric]] = None,
    ):
        """_summary_

        Args:
            model (nn.Module):  underlying PyTorch model.
            loss (nn.Module): loss function.
            lr (float):  learning rate.
            wd (float): weight decay for AdamW optimizer.
            scheduler_func (Optional[Callable], optional): Function that takes an optimizer as input and returns a
            scheduler dict formatted for PytorchLightning.. Defaults to None.
            metrics (Optional[Sequence[Metric]], optional): class:`torchmetrics.Metric` metrics to compute on validation.. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.wd = wd
        self.scheduler_func = scheduler_func
        self.metrics = MetricCollection(ifnone(metrics, []))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(1)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y.float())

        self.log(f"train_loss_{get_loss_name(self.loss)}", loss)
        if self.sched is not None:
            self.log("learning_rate", self.sched["scheduler"].get_last_lr()[0])
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y.float())
        self.log(f"val_loss_{get_loss_name(self.loss)}", loss, sync_dist=True)

        y_hat = torch.sigmoid(y_hat)
        # if batch_idx % 100 == 0 and self.trainer.training_type_plugin.global_rank == 0:
        #     self.log_images(x, y, y_hat, batch_idx)

        self.metrics(y_hat, y.int())

    def validation_epoch_end(self, outputs: Dict[str, Tensor]):
        self.log_dict(self.metrics.compute(), sync_dist=True)

    def configure_optimizers(
        self,
    ) -> Union[
        Optimizer, Dict[str, Union[Optimizer, Dict[str, Union[str, _LRScheduler]]]]
    ]:
        self.opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler_func is None:
            self.sched = None
            return self.opt
        else:
            self.sched = self.scheduler_func(self.opt)
            return {"optimizer": self.opt, "lr_scheduler": self.sched}

    def log_images(self, x: Tensor, y: Tensor, y_hat: Tensor, batch_idx: int):
        y = y[:, None].repeat(1, 3, 1, 1)
        y_hat = y_hat[:, None].repeat(1, 3, 1, 1)
        sample_imgs = torch.cat((x, y, y_hat))
        grid = make_grid(sample_imgs, y.shape[0])
        self.logger.experiment.log_image(
            to_pil_image(grid),
            f"val_image_sample_{self.current_epoch}_{batch_idx}",
            step=self.current_epoch,
        )

    # def freeze_encoder(self):
    #     for m in named_leaf_modules(self.model):
    #         if "encoder" in m.name and not isinstance(m, nn.BatchNorm2d):
    #             for param in m.parameters():
    #                 param.requires_grad = False
