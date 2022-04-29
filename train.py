import argparse
import os
import yaml
import pytorch_lightning as pl
from albumentations import (
    RandomRotate90,
    Flip,
    Transpose,
    RandomBrightnessContrast,
)
from math import ceil

from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, Specificity, Accuracy

from src.transforms import ToTensor
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

    # check for transformation
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
    # load the dataset
    train_ds = data_loader.ClassificationDataset(
        cfg["slide_file"], transforms=transforms, noted=cfg["noted"]
    )
    val_ds = data_loader.ClassificationDataset(
        cfg["slide_file"],
        split="valid",
        transforms=[
            ToTensor(),
        ],
        noted=cfg["noted"],
    )

    # sampler = data_loader.BalancedRandomSampler(train_ds, p_pos=1)
    print("########################## loader ##########################")
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print("loaded")

    # initialize the scheduler
    scheduler_func = pl_modules.get_scheduler_func(
        cfg["scheduler"],
        total_steps=ceil(len(train_dl) / (cfg["grad_accumulation"])) * cfg["epochs"],
        lr=cfg["lr"],
    )

    # model = maskrcnn_resnet50_fpn(num_classes=2)
    # Init model
    model = models.build_model(cfg["model"], 1, cfg["freeze"])

    # creating unique log folder
    logfolder_temp = Path(cfg["logfolder"])
    logfolder = logfolder_temp / "luminal/"  # Path
    logdir = utils.generate_unique_logpath(logfolder, cfg["model"])
    print("Logging to {}".format(logdir))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Init pl module
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
        scheduler_name=cfg["scheduler"],
        logdir=logdir,
    )
    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace="mehdiec",
        save_dir=logdir,
        project_name="luminal",
        auto_metric_logging=True,
    )
    cfg["logdir"] = logdir
    logger.log_hyperparams(cfg)

    # if not args.horovod or hvd.rank() == 0:
    #     logger.experiment.add_tag(args.ihc_type)
    loss = cfg["loss"]
    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        save_last=True,
        mode="min",
        filename=f"{{epoch}}-{{val_loss_{loss}:.3f}}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        gpus=1 if cfg["horovod"] else [cfg["gpu"]],
        min_epochs=3,
        max_epochs=20,
        logger=logger,
        precision=16,
        accumulate_grad_batches=cfg["grad_accumulation"],
        callbacks=[ckpt_callback, early_stop_callback],
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
        # ckpt_path=logdir,
    )


if __name__ == "__main__":

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)

    main(cfg=config_file)
