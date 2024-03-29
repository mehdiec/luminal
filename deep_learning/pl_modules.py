import json
import cv2
import numpy as np
import pytorch_lightning as pl
import torch

from pathaia.util.basic import ifnone
from torch import Tensor, nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
import torch.nn.functional as F
from torchmetrics import ROC, ConfusionMatrix
from torchmetrics.functional import auc
from torchvision.transforms.functional import to_pil_image
from torchmetrics import Metric, MetricCollection
from typing import Optional, Dict, Callable, Sequence, Tuple, Union


from deep_learning.losses import get_loss_name


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
                "scheduler": ReduceLROnPlateau(
                    opt,
                    patience=2,
                ),
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
        num_classes=2,
        device=1,
        train_dl=None,
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
        self.num_classes = num_classes
        self.loss = loss
        self.lr = lr
        self.wd = wd
        self.scheduler_func = scheduler_func
        self.metrics = MetricCollection(ifnone(metrics, []))
        self.cm = ConfusionMatrix(num_classes=num_classes, compute_on_step=False)
        self.roc = ROC(num_classes=num_classes, compute_on_step=False)
        self.temp_dict = {}
        self.scheduler_name = scheduler_name
        self.slide_info = {
            "idx": [],
            "y_hat": None,
            "true": None,
            "pos_x": None,
            "pos_y": None,
            "target": None,
        }
        self.logdir = logdir

        self.main_device = f"cuda:{device}"
        self.count_lumA_0 = {i: 0 for i in range(10)}
        self.count_lumA_1 = {i: 0 for i in range(10)}
        self.count_lumB_0 = {i: 0 for i in range(10)}
        self.count_lumB_1 = {i: 0 for i in range(10)}
        self.train_dl = train_dl
        if num_classes == 3:
            self.loss_ab = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 1.0, 0])
            )  # weight=torch.tensor([1.0, 1.0, 0.8])

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x.squeeze(1).float())  # .squeeze(1)

    def on_fit_start(self):

        x = next(iter(self.train_dl))
        for img in x["image"]:
            out = F.interpolate(img, size=256)  # The resize operation on tensor.
            self.logger.experiment.log_image(
                out, image_channels="first", image_minmax=(0.0, 1.0)
            )

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        loss, y_hat, y, _, _, _, _, _ = self.common_step(batch)
        if self.num_classes == 3:
            val_loss_ab = self.loss_ab(y_hat, y)
            self.log(f"train_loss_ab", val_loss_ab, sync_dist=True)
        self.log(f"train_loss_{get_loss_name(self.loss)}", loss)
        self.log("learning_rate", self.opt.param_groups[0]["lr"])

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):

        loss, y_hat, y, slide_idx, images, p_x, p_y, y_slide = self.common_step(batch)
        if self.num_classes == 3:
            val_loss_ab = self.loss_ab(y_hat, y)
            self.log(f"val_loss_ab", val_loss_ab, sync_dist=True)

        nn.CrossEntropyLoss()  # weight=torch.tensor([1.0, 1.0, 0.8])
        self.log(f"val_loss", loss, sync_dist=True)
        self.concat_info(slide_idx, y_hat, y.int(), p_x, p_y, y_slide, y)

        preds = torch.softmax(y_hat, 1)  # torch.softmax(y_hat)
        count = 0

        for i in range(10):
            count += self.count_lumB_0[i]
            count += self.count_lumA_1[i]
            count += self.count_lumB_1[i]
            count += self.count_lumA_0[i]

        if count < 0:
            self.log_image_check(preds, y, images, slide_idx)

        self.update_metrics(preds, y)

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
        y_slide = batch["target_slide"]
        y_hat = self(image)
        loss = self.loss(y_hat, target)  # .squeeze(),

        return loss, y_hat, target, slide_idx, image, p_x, p_y, y_slide

    def concat_info(self, slide_idx, y_hat, y, p_x, p_y, y_slide, target):
        if len(self.slide_info["idx"]) == 0:
            self.slide_info["idx"] = slide_idx
            self.slide_info["y_hat"] = y_hat
            self.slide_info["true"] = y_slide.int()
            self.slide_info["pos_x"] = p_x
            self.slide_info["pos_y"] = p_y
            self.slide_info["target"] = target
        # after first checking it is possible concatenate the tensors
        else:
            self.slide_info["idx"] = torch.cat((self.slide_info["idx"], slide_idx), 0)
            self.slide_info["y_hat"] = torch.cat((self.slide_info["y_hat"], y_hat), 0)
            self.slide_info["target"] = torch.cat(
                (self.slide_info["target"], target), 0
            )
            self.slide_info["true"] = torch.cat(
                (self.slide_info["true"], y_slide.int()), 0
            )
            self.slide_info["pos_x"] = torch.cat((self.slide_info["pos_x"], p_x), 0)
            self.slide_info["pos_y"] = torch.cat((self.slide_info["pos_y"], p_y), 0)

    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor):
        self.roc(y_hat, y)
        self.cm(y_hat, y)
        self.metrics(y_hat, y.int())

    def compute_slide_metrics(self):

        # all the slide ids are put in a dictionary with empty values
        y_hat_slide = {int(i.cpu().numpy()): [] for i in self.slide_info["idx"]}
        target_patch = {int(i.cpu().numpy()): [] for i in self.slide_info["idx"]}
        target_slide = {int(i.cpu().numpy()): -1 for i in self.slide_info["idx"]}
        pos_x = {int(i.cpu().numpy()): -1 for i in self.slide_info["idx"]}
        pos_y = {int(i.cpu().numpy()): -1 for i in self.slide_info["idx"]}

        # the two dictionaries are populated by the info
        for i, idx in enumerate(self.slide_info["idx"]):
            cpu_idx = int(idx.cpu().numpy())
            y_hat_tmp = self.slide_info["y_hat"][i : i + 1][:, :3]
            pred_tmp = torch.softmax(y_hat_tmp, 1)

            if torch.argmax(pred_tmp, 1) != 2:
                if isinstance(y_hat_slide[cpu_idx], list):

                    y_hat_slide[cpu_idx] = self.slide_info["y_hat"][i : i + 1][:, :2]
                    target_patch[cpu_idx] = self.slide_info["target"][i : i + 1]

                    pos_x[cpu_idx] = self.slide_info["pos_x"][i : i + 1]
                    pos_y[cpu_idx] = self.slide_info["pos_y"][i : i + 1]

                else:
                    y_hat_slide[cpu_idx] = torch.cat(
                        (
                            y_hat_slide[cpu_idx],
                            self.slide_info["y_hat"][i : i + 1][:, :2],
                        ),
                        dim=0,
                    )
                    target_patch[cpu_idx] = torch.cat(
                        (
                            target_patch[cpu_idx],
                            self.slide_info["target"][i : i + 1],
                        ),
                        dim=0,
                    )
                    pos_x[cpu_idx] = torch.cat(
                        (pos_x[cpu_idx], self.slide_info["pos_x"][i : i + 1]), dim=0
                    )
                    pos_y[cpu_idx] = torch.cat(
                        (pos_y[cpu_idx], self.slide_info["pos_y"][i : i + 1]), dim=0
                    )
            target_slide[cpu_idx] = self.slide_info["true"][i]

        pred_slide_mean = []
        t = self.current_epoch

        for y_hat in y_hat_slide.values():

            if len(y_hat) < 1:
                continue
            y_hat = torch.tensor(y_hat)
            y_hat = torch.softmax(y_hat, 1)
            pred_slide_mean.append(y_hat.mean(0))

        pred_slide_mean = torch.stack(pred_slide_mean)
        # print(pred_slide_mean.shape)
        patches_predictions_hashable = {}
        pos_x_hash = {}
        pos_y_hash = {}
        # slide_prediction = pred_slide_mean.cpu.numpy.to_list()

        for cpu_idx, y_hat in y_hat_slide.items():
            if len(y_hat) > 0:
                patches_predictions_hashable[cpu_idx] = y_hat.cpu().numpy().tolist()
                pos_x_hash[cpu_idx] = y_hat.cpu().numpy().tolist()

                pos_y_hash[cpu_idx] = y_hat.cpu().numpy().tolist()

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
        if pred_slide_mean.shape[0] == 3:
            pred_slide_mean = pred_slide_mean.unsqueeze(0)
            # targets = targets.unsqueeze(0)

        pred = pred_slide_mean

        # the metrics are reseted then compute them

        self.cm.reset()
        self.roc.reset()

        if pred.shape[0] != targets.shape[0]:
            pred = pred.squeeze()

        print("\n")
        print("\n")

        try:
            self.metrics(pred.to(self.main_device), targets.to(self.main_device))

            log = {}
            suffix = "slide_mean"
            app = f"_{suffix}" if suffix is not None else ""
            metric_dict = self.metrics.compute()
            print(self.metrics.compute())
            for metric in metric_dict:
                val = metric_dict[metric]
                log[metric + app] = val

            self.log_dict(log, on_step=False, on_epoch=True)

            self.metrics.reset()

            pred_filtered = []
            targetu = []
            for y_hat, y_target in zip(y_hat_slide.values(), target_patch.values()):
                if len(y_hat) < 1:
                    continue
                y_hat = torch.tensor(y_hat)
                y_hat = torch.softmax(y_hat, 1)

                pred_filtered.append(y_hat)
                # for i in range(y_hat.shape[0]):
                targetu.append(y_target)
            pred_filtered = torch.cat(pred_filtered, dim=0)
            targetu = torch.cat(targetu, dim=0)

            print((pred_filtered.shape, targetu.shape))

            self.metrics(pred_filtered, targetu)
            log = {}
            suffix = "patch_ab"
            app = f"_{suffix}" if suffix is not None else ""
            metric_dict = self.metrics.compute()
            print(self.metrics.compute())
            for metric in metric_dict:
                val = metric_dict[metric]
                log[metric + app] = val

            self.log_dict(log, on_step=False, on_epoch=True)
        except:
            log = {}
            log["Accuracy_slide_mean"] = 0
            self.log_dict(log, on_step=False, on_epoch=True)

            self.metrics.reset()
        self.metrics.reset()
        self.slide_info = {
            "idx": [],
            "y_hat": None,
            "true": None,
            "pos_x": None,
            "pos_y": None,
        }

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

    def log_metrics(self, metrics, cm, roc, suffix: str = None):
        if self.num_classes == 3:
            classes = ["luminal A", "luminal B", "other"]
        else:
            classes = ["luminal A", "luminal B"]
        log = {}
        app = f"_{suffix}" if suffix is not None else ""
        metric_dict = metrics.compute()
        for metric in metric_dict:
            val = metric_dict[metric]
            if val.numel() == 1:
                log[metric + app] = val
            else:
                for k, cl in enumerate(classes):
                    c_name = f"{metric}{app}_{cl}"
                    log[c_name] = val[k]
                log[f"{metric}{app}_mean"] = val.mean()
        if not self.trainer.sanity_checking:
            mat = cm.compute().cpu().numpy()
            self.logger.experiment.log_confusion_matrix(
                labels=classes,
                matrix=mat,
                step=self.global_step,
                epoch=self.current_epoch,
                file_name=f"confusion_matrix{app}_{self.current_epoch}.json",
            )
            fprs, tprs, _ = roc.compute()
            for cl, fpr, tpr in zip(classes[::-1], fprs[::-1], tprs[::-1]):
                fpr = fpr
                tpr = tpr
                self.logger.experiment.log_curve(
                    f"ROC{app}_{cl}_{self.current_epoch}",
                    x=fpr.tolist(),
                    y=tpr.tolist(),
                    step=self.current_epoch,
                    overwrite=False,
                )
                log[f"AUC{app}_{cl}"] = auc(fpr, tpr)

        metrics.reset()
        cm.reset()
        roc.reset()
        self.log_dict(log, on_step=False, on_epoch=True)

    # def log_image_check(self, preds, y, images, slide_idx):
    #     for pred, taget, image, slide_id in zip(preds, y, images, slide_idx):
    #         slide_id = int(slide_id.cpu().numpy())
    #         taget = taget.item()
    #         pred_A = pred[0].item()
    #         pred_B = pred[0].item()
    #         pred_C = pred[0].item()
    #         if taget == 1 and pred_A > 0.5:
    #             if self.count_lumB_0[slide_id] < 10:
    #                 self.log_images(
    #                     image * 255,
    #                     slide_id,
    #                     title=f"/BA luminal B classe comme A| prediction:{pred_B}",
    #                 )
    #                 self.count_lumB_0[slide_id] += 1

    #         if taget == 1 and pred_B > 0.5:
    #             if self.count_lumB_1[slide_id] < 10:
    #                 self.log_images(
    #                     image * 255,
    #                     slide_id,
    #                     title=f"/BB luminal B classe comme B| prediction:{pred_B}",
    #                 )
    #                 self.count_lumB_1[slide_id] += 1
    #         if taget == 0 and pred_A > 0.5:
    #             if self.count_lumA_0[slide_id] < 10:
    #                 self.log_images(
    #                     image * 255,
    #                     slide_id,
    #                     title=f"/AA luminal A classe comme A| prediction:{pred_A}",
    #                 )
    #                 self.count_lumA_0[slide_id] += 1
    #         if taget == 0 and pred_B > 0.5:
    #             if self.count_lumA_1[slide_id] < 10:
    #                 self.log_images(
    #                     image * 255,
    #                     slide_id,
    #                     title=f"/AB luminal A classe comme B| prediction:{pred_A}",
    #                 )
    #                 self.count_lumA_1[slide_id] += 1
    #         if taget == 0 and pred_C > 0.5:
    #             if self.count_lumA_1[slide_id] < 10:
    #                 self.log_images(
    #                     image * 255,
    #                     slide_id,
    #                     title=f"/luminal A classe comme autre| prediction:{pred_C}",
    #                 )
    #                 self.count_lumA_1[slide_id] += 1
    #         if taget == 1 and pred_C > 0.5:
    #             if self.count_lumA_1[slide_id] < 10:
    #                 self.log_images(
    #                     image * 255,
    #                     slide_id,
    #                     title=f"/luminal B classe comme autre| prediction:{pred_C}",
    #                 )
    #                 self.count_lumA_1[slide_id] += 1

    # def log_images(self, x: Tensor, slide_id: int, title: str):
    #     a = self.logdir + f"/{self.current_epoch}/{slide_id}" + title + ".png"
    #     if not cv2.imwrite(a, np.transpose(x.cpu().numpy(), (1, 2, 0))):
    #         raise Exception("Could not write image")
