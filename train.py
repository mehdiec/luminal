import argparse
import os
import yaml
import pytorch_lightning as pl
from albumentations import (
    Normalize,
    RandomRotate90,
    Flip,
    Transpose,
    RandomBrightnessContrast,
    GlassBlur,
    CenterCrop,
    Resize,
    CoarseDropout,
    Blur,
    Cutout,
    Rotate,
    ChannelShuffle,
    RGBShift,
)
from math import ceil

from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from torchmetrics import F1Score, Precision, Recall, Specificity, Accuracy
from src.preprocess import load_patches

from src.transforms import StainAugmentor, ToTensor
from src import models, pl_modules_temp, losses, utils

shuft = 30 / 255

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


def main(cfg):
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
    transforms = [ToTensor()]
    if cfg["transform"]:
        if cfg["patch_size"] == 2048:
            transforms = [
                Resize(1024, 1024),
                # Resize(256, 256),
                # CenterCrop(224,224)
                Flip(),
                Transpose(),
                RandomRotate90(),
                ChannelShuffle(always_apply=False, p=0.5),
                RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=(-0.3, 0.05),
                    brightness_by_max=True,
                    always_apply=False,
                    p=0.8,
                ),
                RGBShift(
                    always_apply=False,
                    p=0.5,
                    r_shift_limit=(-shuft, shuft),
                    g_shift_limit=(-shuft, shuft),
                    b_shift_limit=(-shuft, shuft),
                ),
                Blur(always_apply=False, p=0.5, blur_limit=(3, 4)),
                # Cutout(
                #     always_apply=False,
                #     p=0.5,
                #     num_holes=40,
                #     max_h_size=60,
                #     max_w_size=60,
                #     fill_value=(1, 1, 1),
                # ),
                # Rotate(always_apply=False, p=.5, limit=(-10, 10), interpolation=4, border_mode=2, value=(0, 0, 0), mask_value=None),
                # Rotate(
                #     always_apply=False,
                #     p=0.8,
                #     limit=(-90, 314),
                #     interpolation=2,
                #     border_mode=0,
                #     value=(1, 1, 1),
                #     mask_value=None,
                # ),
                # StainAugmentor(),
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor(),
                # RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0.5, p=1,src_color=(255, 255, 255)),
            ]
        else:
            transforms = [
                # Cutout(
                #     always_apply=False,
                #     p=0.5,
                #     num_holes=40,
                #     max_h_size=60,
                #     max_w_size=60,
                #     fill_value=(1, 1, 1),
                # ),
                # Resize(256, 256),
                # CenterCrop(224, 224),
                Flip(),
                Transpose(),
                RandomRotate90(p=0.5),
                # StainAugmentor(),
                # RandomBrightnessContrast(
                #     brightness_limit=0.2,
                #     contrast_limit=(-0.3, 0.05),
                #     brightness_by_max=True,
                #     always_apply=False,
                #     p=0.5,
                # ),
                Blur(always_apply=False, p=0.7, blur_limit=(3, 4)),
                # ChannelShuffle(always_apply=False, p=0.5),
                RGBShift(
                    always_apply=False,
                    p=0.5,
                    r_shift_limit=(-shuft, shuft),
                    g_shift_limit=(-shuft, shuft),
                    b_shift_limit=(-shuft, shuft),
                ),
                # # Rotate(always_apply=False, p=.5, limit=(-10, 10), interpolation=4, border_mode=2, value=(0, 0, 0), mask_value=None),
                # Rotate(
                #     always_apply=False,
                #     p=0.8,
                #     limit=(-90, 314),
                #     interpolation=2,
                #     border_mode=0,
                #     value=(1, 1, 1),
                #     mask_value=None,
                # ),
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor(),
            ]

    train_dl, val_dl = load_patches(
        slide_file=cfg["slide_file"],
        noted=cfg["noted"],
        level=cfg["level"],
        transforms=transforms,
        normalize=cfg["normalize"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        patch_size=cfg["patch_size"],
        num_classes=cfg["num_classes"],
        balance=cfg["balance"],
    )
    print("loaded")

    # initialize the scheduler
    scheduler_func = pl_modules_temp.get_scheduler_func(
        cfg["scheduler"],
        total_steps=ceil(len(train_dl) / (cfg["grad_accumulation"])) * cfg["epochs"],
        lr=cfg["lr"],
    )

    # model = maskrcnn_resnet50_fpn(num_classes=2)
    # Init model
    model = models.build_model(
        cfg["model"],
        cfg["num_classes"],
        cfg["freeze"],
        cfg["pretrained"],
    )

    # creating unique log folder
    logfolder_temp = Path(cfg["logfolder"])
    logfolder = logfolder_temp / "luminal/"  # Path
    logdir = utils.generate_unique_logpath(logfolder, cfg["model"])
    print("Logging to {}".format(logdir))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        for i in range(cfg["epochs"]):
            log_path = os.path.join(logdir, str(i))
            os.mkdir(log_path)
            for j in range(10):
                log_path_ind = os.path.join(log_path, str(j))
                os.mkdir(log_path_ind)

    # Init pl module
    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace="mehdiec",
        save_dir=logdir,
        project_name="luminal",
        auto_metric_logging=True,
    )
    plmodule = pl_modules_temp.BasicClassificationModule(
        model,
        lr=cfg["lr"],
        wd=cfg["wd"],
        loss=losses.get_loss(cfg["loss"]),
        scheduler_func=scheduler_func,
        metrics=[Accuracy(), Precision(), Recall(), Specificity(), F1Score()],
        scheduler_name=cfg["scheduler"],
        logdir=logdir,
        num_classes=cfg["num_classes"],
        device=cfg["gpu"],
    )
    cfg["logdir"] = logdir
    logger.log_hyperparams(cfg)

    # if not args.horovod or hvd.rank() == 0:
    #     logger.experiment.add_tag(args.ihc_type)
    loss = cfg["loss"]

    # checkpoint model to save
    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="Accuracy_patch",
        save_last=True,
        mode="max",
        filename=f"{{epoch}}-{{val_loss_{loss}:.3f}}",
    )

    # earlystopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        gpus=[cfg["gpu"]],
        min_epochs=10,
        max_epochs=50,
        logger=logger,
        precision=16,
        accumulate_grad_batches=cfg["grad_accumulation"],
        auto_select_gpus=True,
        callbacks=[ckpt_callback, early_stop_callback],
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
