import os
import comet_ml
import torch
import sys
import yaml
import argparse
from albumentations import (
    RandomRotate90,
    Flip,
    Transpose,
    RandomBrightnessContrast,
)
from math import ceil

import pandas as pd
from pathlib import Path
from pathaia.util.paths import get_files
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torchmetrics import JaccardIndex, Precision, Recall, Specificity, Accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src import models, data_loader, pl_modules, losses, utils

# Init the parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

seed_everything(workers=True)
# Add path to the config file to the command line arguments
parser.add_argument(
    "--path_to_config",
    type=str,
    required=True,
    default="./config.yaml",
    help="path to config file",
)
args = parser.parse_args()


# parser.add_argument(
#     "--maskfolder",
#     type=Path,
#     help="Input folder containing tif mask files.",
#     required=True,
# )


# parser.add_argument(
#     "--resume-version", help="Version id of a model to load weights from. Optional."
# )
# parser.add_argument(
#     "--seed",
#     type=int,
#     help=(
#         "Specify seed for RNG. Can also be set using PL_GLOBAL_SEED environment "
#         "variable. Optional."
#     ),
# )
experiment = comet_ml.Experiment(
    api_key="57zOARA0d8ftliPTpL3pXTeVc",
    project_name="luminal",
)


def _collate_fn(batch):
    xs = []
    ys = []
    for x, y in batch:
        xs.append(x)
        ys.append(y)
    return xs, ys


def main(cfg, path_to_cfg=""):
    # if args.horovod:
    #     hvd.init()
    print("main")

    seed_everything(workers=True)

    # if args.stain_matrices_folder is not None:
    #     stain_matrices_paths = mask_paths.map(
    #         lambda x: args.stain_matrices_folder / x.with_suffix(".npy").name
    #     )
    #     stain_matrices_paths = stain_matrices_paths[train_idxs]
    # else:
    #     stain_matrices_paths = None
    transforms = None
    if cfg["transform"]:
        transforms = [
            Flip(),
            Transpose(),
            RandomRotate90(),
            RandomBrightnessContrast(),
            ToTensor(),
        ]
    print("########################## dataset ##########################")
    train_ds = data_loader.ClassificationDataset(
        cfg["slide_file"], transforms=transforms
    )
    val_ds = data_loader.ClassificationDataset(
        cfg["slide_file"], split="valid", transforms=transforms
    )

    # sampler = data_loader.BalancedRandomSampler(train_ds, p_pos=1)
    print("########################## loader ##########################")
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=args.batch_size,
    #     pin_memory=True,
    #     num_workers=args.num_workers,
    #     drop_last=True,
    #     sampler=sampler,
    #     collate_fn=_collate_fn,
    #     persistent_workers=True,
    # )
    # val_dl = DataLoader(
    #     val_ds,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=args.num_workers,
    #     collate_fn=_collate_fn,
    #     persistent_workers=True,
    # )

    val_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    )
    print("loaded")
    scheduler_func = pl_modules.get_scheduler_func(
        cfg["scheduler"],
        total_steps=ceil(len(train_dl) / (cfg["grad_accumulation"])) * cfg["epochs"],
        lr=cfg["lr"],
    )

    # model = maskrcnn_resnet50_fpn(num_classes=2)
    # Init model, loss, optimizer
    model = models.build_model(cfg["model"], 1)

    plmodule = pl_modules.BasicClassificationModule(
        model,
        lr=cfg["lr"],
        wd=cfg["wd"],
        loss=losses.get_loss(cfg["loss"]),
        scheduler_func=scheduler_func,
        metrics=[
            Accuracy(),
            Precision(),
            Recall(),
            Specificity(),
        ],
    )
    logfolder_temp = Path(cfg["logfolder"])
    logfolder = logfolder_temp / "luninal/"  # Path
    logdir = utils.generate_unique_logpath(logfolder, cfg["model"])
    print("Logging to {}".format(logdir))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace="mehdiec",  # changer nom du compte
        save_dir=logdir,  # dossier du log local data nom du compte
        project_name="luninal",  # changer nom
        auto_metric_logging=False,
    )

    # if not args.horovod or hvd.rank() == 0:
    #     logger.experiment.add_tag(args.ihc_type)
    loss = cfg["loss"]
    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"val_loss_{loss}",
        save_last=True,
        mode="min",
        filename=f"{{epoch}}-{{val_loss_{loss}:.3f}}",
    )

    trainer = pl.Trainer(
        gpus=1 if cfg["horovod"] else [cfg["gpu"]],
        min_epochs=cfg["epochs"],
        max_epochs=cfg["epochs"],
        logger=logger,
        precision=16,
        accumulate_grad_batches=cfg["grad_accumulation"],
        callbacks=[ckpt_callback],
        # strategy="horovod" if args.horovod else None,
    )

    # if cfg["resume_version"]  is not None:
    #     ckpt_path = (
    #         cfg["logfolder"]   / f"luninal/{cfg["resume_version"]}/checkpoints/last.ckpt" #Pqth
    #     )
    #     checkpoint = torch.load(ckpt_path)
    #     missing, unexpected = plmodule.load_state_dict(
    #         checkpoint["state_dict"], strict=False
    #     )
    trainer.fit(
        plmodule,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        # ckpt_path=ckpt_path,
    )


if __name__ == "__main__":

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)

    main(cfg=config_file)
