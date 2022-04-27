from ordered_set import T
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
from torchmetrics import ROC, ConfusionMatrix
from torchmetrics.functional import auc
from torchvision.utils import make_grid
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
        scheduler_name=None,
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
        self.slide_info = {"idx": [], "y_hat": None, "true": None}

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(1)

    def common_step(self, batch, batch_idx):

        image = batch["image"]
        slide_idx = batch["idx"]
        target = batch["target"]
        y_hat = self(image)
        loss = self.loss(y_hat, target.float())

        return loss, y_hat, target.int(), slide_idx

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        loss, _, _, _ = self.common_step(batch, batch_idx)

        self.log(f"train_loss_{get_loss_name(self.loss)}", loss)

        print(self.opt.param_groups[0]["lr"])

        if self.scheduler_func is not None:

            if self.scheduler_name != "reduce-on-plateau":

                self.log("learning_rate", self.sched["scheduler"].get_last_lr()[0])

            else:

                self.log("learning_rate", self.opt.param_groups[0]["lr"])

        return loss

    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor):
        # y = y.to(self.main_device)
        # y_hat = y_hat.to(self.main_device)
        self.roc(y_hat, y)
        self.cm(y_hat, y)
        self.metrics(y_hat, y.int())

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ):
        loss, y_hat, y, slide_idx = self.common_step(batch, batch_idx)

        self.log(f"val_loss", loss, sync_dist=True)
        # print(slide_idx)

        if len(self.slide_info["idx"]) == 0:
            self.slide_info["idx"] = slide_idx

            self.slide_info["y_hat"] = y_hat
            self.slide_info["true"] = y.int()
        else:
            self.slide_info["idx"] = torch.cat((self.slide_info["idx"], slide_idx), 0)
            self.slide_info["y_hat"] = torch.cat((self.slide_info["y_hat"], y_hat), 0)
            self.slide_info["true"] = torch.cat((self.slide_info["true"], y.int()), 0)

        pred = torch.sigmoid(y_hat)
        # if batch_idx % 100 == 0 and self.trainer.training_type_plugin.global_rank == 0:
        #     self.log_images(x, y, y_hat, batch_idx)

        self.update_metrics(pred, y)

    def validation_epoch_end(self, outputs: Dict[str, Tensor]):
        print(self.metrics.compute())
        # self.log_dict(self.metrics.compute(), sync_dist=True)
        self.log_metrics(self.metrics)
        self.compute_slide_metrics()

        # self.temp_dict = {}

    def compute_slide_metrics(self):

        y_hat_slide = {int(i.cpu().numpy()): [] for i in self.slide_info["idx"]}
        target_slide = {int(i.cpu().numpy()): -1 for i in self.slide_info["idx"]}
        print(target_slide)
        print(self.slide_info["y_hat"])
        for i, idx in enumerate(self.slide_info["idx"]):

            cpu_idx = int(idx.cpu().numpy())
            if isinstance(y_hat_slide[cpu_idx], list):
                y_hat_slide[cpu_idx] = self.slide_info["y_hat"][i : i + 1]

            else:
                y_hat_slide[cpu_idx] = torch.cat(
                    (y_hat_slide[cpu_idx], self.slide_info["y_hat"][i : i + 1]), dim=0
                )
            target_slide[cpu_idx] = self.slide_info["true"][i]

        pred_slide_mean = []
        pred_slide_vote = []
        for y_hat in y_hat_slide.values():
            pred_slide_mean.append(y_hat.mean(0))

        pred_slide_mean = torch.Tensor(pred_slide_mean)

        targets = torch.LongTensor(list(target_slide.values()))

        print("pred_slide_mean", pred_slide_mean)

        print(pred_slide_mean, targets)
        print(torch.sigmoid(pred_slide_mean), targets)
        pred = torch.sigmoid(pred_slide_mean)

        self.metrics.reset()
        self.cm.reset()
        self.roc.reset()

        self.metrics(pred, targets)
        # self.roc(pred_slide_mean, targets)
        # self.cm(pred_slide_mean, targets)

        # self.log_dict(self.metrics.compute(), sync_dist=True)
        self.log_metrics(self.metrics, suffix="slide")

        # self.metrics(pred_slide_vote, targets)
        # self.roc(pred_slide_vote, targets)
        # self.cm(pred_slide_vote, targets)
        # # self.log_dict(self.metrics.compute(), sync_dist=True)
        # self.log_metrics(self.metrics, self.cm, self.roc, suffix="slide_vote")

    # def log_slide_metrics(self, preds: torch.Tensor, labels: torch.Tensor):
    #     if not self.trainer.sanity_checking:
    #         items = self.trainer.datamodule.data.valid.items
    #         patch_slides = np.vectorize(lambda x: x.parent.name)(items)
    #         slides = np.unique(patch_slides)
    #         slide_labels = []
    #         slide_preds = []
    #         roc = ROC(num_classes=self.hparams.n_classes, compute_on_step=False)
    #         cm = ConfusionMatrix(self.hparams.n_classes, compute_on_step=False)
    #         metrics = ClassifMetrics(
    #             n_classes=self.hparams.n_classes, compute_on_step=False
    #         )
    #         for slide in slides:
    #             idxs = np.argwhere(patch_slides == slide).squeeze()
    #             label = labels[idxs][0]
    #             pred = preds[idxs].mean(0)
    #             slide_labels.append(label)
    #             slide_preds.append(pred)
    #         slide_labels = torch.stack(slide_labels)
    #         slide_preds = torch.stack(slide_preds)
    #         cm(slide_preds, slide_labels)
    #         metrics(slide_preds, slide_labels)
    #         roc(slide_preds, slide_labels)
    #         self.log_metrics(metrics, cm, roc, suffix="slide")

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
            tamp_sched = {
                "optimizer": self.opt,
                "scheduler": self.scheduler_func(self.opt),
                "monitor": "val_loss",
                "interval": "epoch",
            }
            return tamp_sched

    # def log_images(self, x: Tensor, y: Tensor, y_hat: Tensor, batch_idx: int):
    #     y = y[:, None].repeat(1, 3, 1, 1)
    #     y_hat = y_hat[:, None].repeat(1, 3, 1, 1)
    #     sample_imgs = torch.cat((x, y, y_hat))
    #     grid = make_grid(sample_imgs, y.shape[0])
    #     self.logger.experiment.log_image(
    #         to_pil_image(y),
    #         f"val_image_sample_{self.current_epoch}_{batch_idx}",
    #         step=self.current_epoch,
    #     )

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

    # def freeze_encoder(self):
    #     for m in named_leaf_modules(self.model):
    #         if "encoder" in m.name and not isinstance(m, nn.BatchNorm2d):
    #             for param in m.parameters():
    #                 param.requires_grad = False
