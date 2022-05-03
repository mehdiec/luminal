import argparse
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import models
from albumentations import (
    Normalize,
    CenterCrop,
    Resize,
    RandomRotate90,
    Flip,
    Transpose,
    RandomBrightnessContrast,
    HueSaturationValue,
)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import utils
import data_loader
import transforms
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import json


parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU")

parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Where to store the downloaded dataset",
    default="/mounts/Datasets1/ChallengeDeep/",
)

parser.add_argument(
    "--num_workers", type=int, default=1, help="The number of CPU threads used"
)

parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")

parser.add_argument(
    "--data_augment",
    help="Specify if you want to use data augmentation",
    action="store_true",
)

parser.add_argument(
    "--normalize",
    help="Which normalization to apply to the input data",
    action="store_true",
)

parser.add_argument(
    "--logdir",
    type=str,
    default="./logs",
    help="The directory in which to store the logs",
)

parser.add_argument(
    "--model",
    choices=["vanilla", "fancyCNN", "PenCNN", "resnet", "densenet"],
    action="store",
    required=True,
)

args = parser.parse_args()


num_classes = 86
batch_size = 32
epochs = 15
valid_ratio = 0.2


def main(cfg):
    # print("Using GPU{}".format(torch.cuda.current_device()))
    device = torch.device("cuda:0")

    # Where to store the logs
    logfolder_temp = Path(cfg["logfolder"])
    logfolder = logfolder_temp / "luminal/"  # Path
    logdir = utils.generate_unique_logpath(logfolder, cfg["model"])
    print("Logging to {}".format(logdir))
    print("Logging to {}".format(logdir))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        for i in range(7):
            log_path = os.path.join(logdir, str(i))
            os.mkdir(log_path)

    # Data augmentation
    train_augment_transforms = None
    if args.data_augment:
        transforms = [
            Normalize(mean=[0.0, 0.0, 0.0], std=[1, 1, 1]),
            # Resize(256, 256),
            # CenterCrop(224, 224),
            Flip(),
            Transpose(),
            RandomRotate90(),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.6),
            # transforms.CLAHE(p=0.8),
            # HueSaturationValue(
            #     hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.6
            # ),
            transforms.ToTensor(),
        ]

    train_ds = data_loader.ClassificationDataset(
        cfg["slide_file"], transforms=transforms, noted=cfg["noted"], level=cfg["level"]
    )
    val_ds = data_loader.ClassificationDataset(
        cfg["slide_file"],
        split="valid",
        transforms=[
            Normalize(
                mean=[0.0, 0.0, 0.0], std=[1, 1, 1]
            ),  # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Resize(256, 256),
            # CenterCrop(224, 224),
            transforms.ToTensor(),
        ],
        noted=cfg["noted"],
        level=cfg["level"],
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

    # Init model, loss, optimizer
    model = models.build_model(cfg["model"], 1, cfg["freeze"], cfg["pretrained"])

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=0)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=6, gamma=0.1
    )

    # Where to save the logs of the metrics

    logger = Experiment(
        api_key=os.environ["COMET_API_KEY"],
        workspace="mehdiec",
        save_dir=logdir,
        project_name="no-luminal",
        auto_metric_logging=True,
    )
    cfg["logdir"] = logdir
    logger.log_hyperparams(cfg)
    history_file = open(logdir + "/history", "w", 1)
    history_file.write(
        "Epoch\tTrain loss\tTrain acc\tVal loss\tVal acc\n"  # \tTest loss\tTest acc\n"
    )

    # Generate and dump the summary of the model
    model_summary = utils.torch_summarize(model)

    summary_file = open(logdir + "/summary.txt", "w")
    summary_text = """
    Executed command
    ===============
    {}
    Dataset
    =======
    Train transform : {}
    Normalization : {}
    Model summary
    =============
    {}
    {} trainable parameters
    Optimizer
    ========
    {}
    """.format(
        " ".join(sys.argv),
        train_augment_transforms,
        args.normalize,
        str(model).replace("\n", "\n\t"),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        str(optimizer).replace("\n", "\n\t"),
    )
    summary_file.write(summary_text)
    summary_file.close()

    tensorboard_writer = SummaryWriter(log_dir=logdir)
    tensorboard_writer.add_text("Experiment summary", summary_text)
    logger.log_text(summary_text)
    logger.log_parameters(cfg)

    # Add the graph of the model to the tensorboard
    inputs, _ = next(iter(train_dl))
    inputs = inputs.to(device)
    inputs = inputs.float()
    tensorboard_writer.add_graph(model, inputs)
    ########################################### Main Loop ###########################################
    for t in range(epochs):
        print("Epoch {}".format(t))
        train_loss, train_acc = utils.train(model, train_dl, loss, optimizer, device)
        logger.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
        )

        val_loss, val_acc, val_f1 = utils.test(model, val_dl, loss, device)

        logger.log_metrics({"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        print(
            "Loss : {:.4f}, Acc : {:.4f}, macro F1 :  {:.4f}".format(
                val_loss, val_acc, val_f1
            )
        )

        history_file.write(
            "{}\t{}\t{}\t{}\t{}\n".format(t, train_loss, train_acc, val_loss, val_acc)
        )
        # model_checkpoint.update(val_loss)
        tensorboard_writer.add_scalar("metrics/train_loss", train_loss, t)
        tensorboard_writer.add_scalar("metrics/train_acc", train_acc, t)
        tensorboard_writer.add_scalar("metrics/val_loss", val_loss, t)
        tensorboard_writer.add_scalar("metrics/val_acc", val_acc, t)
        tensorboard_writer.add_scalar("metrics/val_f1", val_f1, t)

    # Loading the best model found
