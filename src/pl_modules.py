import sys
from numpy import squeeze
import pytorch_lightning as pl
from typing import Optional, Dict, Callable, Sequence, Tuple, Union
from torch import Tensor, nn
from torch.optim import Optimizer, AdamW
import torch
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from PIL import Image
from torchmetrics import ROC, ConfusionMatrix
from torchmetrics.functional import auc
from torchvision.transforms.functional import to_pil_image
from torchmetrics import Metric, MetricCollection
from pathaia.util.basic import ifnone

from src.losses import get_loss_name
import json
import cv2
import numpy as np


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
                "scheduler": ReduceLROnPlateau(opt, patience=2),
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
        scheduler_name=None,
        logdir=None,
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
        self.cm = ConfusionMatrix(num_classes=2, compute_on_step=False)
        self.roc = ROC(compute_on_step=False)
        self.temp_dict = {}
        self.scheduler_name = scheduler_name
        self.slide_info = {
            "idx": [],
            "y_hat": None,
            "true": None,
            "pos_x": None,
            "pos_y": None,
        }
        self.logdir = logdir

        self.main_device = "cuda:0"
        self.count_lumA_0 = {i: 0 for i in range(10)}
        self.count_lumA_1 = {i: 0 for i in range(10)}
        self.count_lumB_0 = {i: 0 for i in range(10)}
        self.count_lumB_1 = {i: 0 for i in range(10)}

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x.squeeze(1).float())  # .squeeze(1)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        loss, _, _, _, _, _, _ = self.common_step(batch)

        self.log(f"train_loss_{get_loss_name(self.loss)}", loss)

        # if self.scheduler_func is not None:

        self.log("learning_rate", self.opt.param_groups[0]["lr"])
        # self.log("learning_rate", self.sched["scheduler"].get_last_lr()[0])

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):

        loss, y_hat, y, slide_idx, images, p_x, p_y = self.common_step(batch)

        self.log(f"val_loss", loss, sync_dist=True)
        # print(slide_idx)

        self.concat_info(slide_idx, y_hat, y, p_x, p_y)

        # at first slide info is a dict with empty values

        preds = torch.sigmoid(y_hat)
        count = 0
        for i in range(10):
            count += self.count_lumB_0[i]
            count += self.count_lumA_1[i]
            count += self.count_lumB_1[i]
            count += self.count_lumA_0[i]
        if count < 400:
            self.log_image_check(preds, y, images, slide_idx)

        # if batch_idx % 100 == 0 and self.trainer.training_type_plugin.global_rank == 0:
        #     self.log_images(x, y, y_hat, batch_idx)

        self.update_metrics(preds.squeeze(), y)

    def validation_epoch_end(self, outputs: Dict[str, Tensor]):

        self.log_metrics(self.metrics, self.cm, self.roc, suffix="patch")
        self.compute_slide_metrics()
        self.count_lumA_0 = {i: 0 for i in range(10)}
        self.count_lumA_1 = {i: 0 for i in range(10)}
        self.count_lumB_0 = {i: 0 for i in range(10)}
        self.count_lumB_1 = {i: 0 for i in range(10)}

        # self.temp_dict = {}

    def common_step(self, batch):

        image = batch["image"]
        slide_idx = batch["idx"]
        target = batch["target"]
        p_x = batch["pos_x"]
        p_y = batch["pos_y"]
        y_hat = self(image)
        loss = self.loss(y_hat.squeeze(), target.float())

        return loss, y_hat, target.int(), slide_idx, image, p_x, p_y

    def concat_info(self, slide_idx, y_hat, y, p_x, p_y):
        if len(self.slide_info["idx"]) == 0:
            self.slide_info["idx"] = slide_idx
            self.slide_info["y_hat"] = y_hat
            self.slide_info["true"] = y.int()
            self.slide_info["pos_x"] = p_x
            self.slide_info["pos_y"] = p_y
        # after first checking it is possible concatenate the tensors
        else:
            self.slide_info["idx"] = torch.cat((self.slide_info["idx"], slide_idx), 0)
            self.slide_info["y_hat"] = torch.cat((self.slide_info["y_hat"], y_hat), 0)
            self.slide_info["true"] = torch.cat((self.slide_info["true"], y.int()), 0)
            self.slide_info["pos_x"] = torch.cat((self.slide_info["pos_x"], p_x), 0)
            self.slide_info["pos_y"] = torch.cat((self.slide_info["pos_y"], p_y), 0)

    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor):
        self.roc(y_hat, y)
        self.cm(y_hat, y)
        self.metrics(y_hat, y.int())

    def compute_slide_metrics(self):

        # all the slide ids are put in a dictionary with empty values
        y_hat_slide = {int(i.cpu().numpy()): [] for i in self.slide_info["idx"]}
        target_slide = {int(i.cpu().numpy()): -1 for i in self.slide_info["idx"]}
        pos_x = {int(i.cpu().numpy()): -1 for i in self.slide_info["idx"]}
        pos_y = {int(i.cpu().numpy()): -1 for i in self.slide_info["idx"]}

        # the two dictionaries are populated by the info
        for i, idx in enumerate(self.slide_info["idx"]):
            cpu_idx = int(idx.cpu().numpy())
            if isinstance(y_hat_slide[cpu_idx], list):
                y_hat_slide[cpu_idx] = self.slide_info["y_hat"][i : i + 1]

                pos_x[cpu_idx] = self.slide_info["pos_x"][i : i + 1]
                pos_y[cpu_idx] = self.slide_info["pos_y"][i : i + 1]

            else:
                y_hat_slide[cpu_idx] = torch.cat(
                    (y_hat_slide[cpu_idx], self.slide_info["y_hat"][i : i + 1]), dim=0
                )
                pos_x[cpu_idx] = torch.cat(
                    (pos_x[cpu_idx], self.slide_info["pos_x"][i : i + 1]), dim=0
                )
                pos_y[cpu_idx] = torch.cat(
                    (pos_y[cpu_idx], self.slide_info["pos_y"][i : i + 1]), dim=0
                )
            target_slide[cpu_idx] = self.slide_info["true"][i]

        pred_slide_mean = []
        pred_slide_vote = []
        t = self.current_epoch

        patches_predictions = {cpu_idx: y_hat for cpu_idx, y_hat in y_hat_slide.items()}

        for y_hat in y_hat_slide.values():
            pred_slide_mean.append(y_hat.mean(0))

        pred_slide_mean = torch.Tensor(pred_slide_mean)

        # slide_prediction = pred_slide_mean.cpu.numpy.to_list()
        patches_predictions_hashable = {
            cpu_idx: y_hat.cpu().numpy().tolist()
            for cpu_idx, y_hat in y_hat_slide.items()
        }
        pos_x_hash = {
            cpu_idx: y_hat.cpu().numpy().tolist() for cpu_idx, y_hat in pos_x.items()
        }
        pos_y_hash = {
            cpu_idx: y_hat.cpu().numpy().tolist() for cpu_idx, y_hat in pos_y.items()
        }

        sample = {
            "prediction_slide": pred_slide_mean.cpu().numpy().tolist(),
            "prediction_patch": patches_predictions_hashable,
            "pos_x": pos_x_hash,
            "pos_y": pos_y_hash,
        }
        # the results are saved in a json
        with open(self.logdir + f"/result__{t}.json", "w") as fp:
            json.dump(sample, fp)

        targets = torch.LongTensor(list(target_slide.values()))

        print(torch.sigmoid(pred_slide_mean), targets)
        pred = torch.sigmoid(pred_slide_mean)

        # the metrics are reseted then compute them
        self.metrics.reset()
        self.cm.reset()
        self.roc.reset()

        self.update_metrics(pred.to(self.main_device), targets.to(self.main_device))

        # self.log_dict(self.metrics.compute(), sync_dist=True)
        self.log_metrics(self.metrics, cm=self.cm, roc=self.roc, suffix="slide_mean")

        # self.metrics(pred_slide_vote, targets)
        # self.roc(pred_slide_vote, targets)
        # self.cm(pred_slide_vote, targets)
        # # self.log_dict(self.metrics.compute(), sync_dist=True)
        # self.log_metrics(self.metrics, self.cm, self.roc, suffix="slide_vote")

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

    def log_metrics(self, metrics, cm=None, roc=None, suffix: str = None):
        log = {}
        app = f"_{suffix}" if suffix is not None else ""
        metric_dict = metrics.compute()
        print(metrics.compute())
        for metric in metric_dict:
            val = metric_dict[metric]
            log[metric + app] = val
        if cm:

            if not self.trainer.sanity_checking:
                mat = cm.compute().cpu().numpy()
                self.logger.experiment.log_confusion_matrix(
                    # labels=self.hparams.classes,
                    matrix=mat,
                    step=self.global_step,
                    epoch=self.current_epoch,
                    file_name=f"confusion_matrix{app}_{self.current_epoch}.json",
                )
                fprs, tprs, _ = roc.compute()

                self.logger.experiment.log_curve(
                    f"ROC{app}_{self.current_epoch}",
                    x=fprs.tolist(),
                    y=tprs.tolist(),
                    step=self.current_epoch,
                    overwrite=False,
                )
                log[f"AUC{app}"] = auc(fprs, tprs)
                cm.reset()
                roc.reset()

        metrics.reset()
        self.log_dict(log, on_step=False, on_epoch=True)

    def log_image_check(self, preds, y, images, slide_idx):
        for pred, taget, image, slide_id in zip(preds, y, images, slide_idx):
            slide_id = int(slide_id.cpu().numpy())
            if taget == 1 and pred < 0.251:
                if self.count_lumB_0[slide_id] < 10:
                    self.log_images(
                        image * 255,
                        slide_id,
                        title=f"/BA luminal B classe comme A| prediction:{pred[0]}",
                    )
                    self.count_lumB_0[slide_id] += 1

            if taget == 1 and pred > 0.729:
                if self.count_lumB_1[slide_id] < 10:
                    self.log_images(
                        image * 255,
                        slide_id,
                        title=f"/BB luminal B classe comme B| prediction:{pred[0]}",
                    )
                    self.count_lumB_1[slide_id] += 1
            if taget == 0 and pred < 0.251:
                if self.count_lumA_0[slide_id] < 10:
                    self.log_images(
                        image * 255,
                        slide_id,
                        title=f"/AA luminal A classe comme A| prediction:{pred[0]}",
                    )
                    self.count_lumA_0[slide_id] += 1
            if taget == 0 and pred > 0.729:
                if self.count_lumA_1[slide_id] < 10:
                    self.log_images(
                        image * 255,
                        slide_id,
                        title=f"/AB luminal A classe comme B| prediction:{pred[0]}",
                    )
                    self.count_lumA_1[slide_id] += 1

    def log_images(self, x: Tensor, slide_id: int, title: str):
        # sample_imgs = torch.zeros([100, 100, 3])  #
        # print(sample_imgs.transpose(0, 1).transpose(1, 2).shape)
        # b, g, r = image.split()
        # image = Image.merge("RGB", (r, g, b))
        a = self.logdir + f"/{self.current_epoch}/{slide_id}" + title + ".png"
        if not cv2.imwrite(a, np.transpose(x.cpu().numpy(), (1, 2, 0))):
            raise Exception("Could not write image")

        # self.logger.experiment.log_image(
        #     sample_imgs,
        #     name=title,
        #     step=self.current_epoch,
        # )
