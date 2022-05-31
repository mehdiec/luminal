import argparse
import os.path
import torch
import yaml

from pathaia.util.types import Slide
from pytorch_lightning.utilities.seed import seed_everything

from src import models
from utils import gradcam, patch_from_coord


MAPPING = {
    "luminal A": 0,
    "luminal B": 1,
}
MAPPING_inv = {
    0: "luminal A",
    1: "luminal B",
}


# Init the parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

seed_everything(workers=True)
# Add path to the config file to the command line arguments
parser.add_argument(
    "--path_to_config",
    type=str,
    required=True,
    default="./config_gc.yaml",
    help="path to config file",
)
args = parser.parse_args()


def main(cfg):

    print("main")
    lesnet = torch.load(cfg["model"])
    resnet = {k.replace("model.", ""): v for k, v in lesnet["state_dict"].items()}
    resnet_ = {
        k.replace("model.", "resnet."): v for k, v in lesnet["state_dict"].items()
    }
    sttdict = {**resnet, **resnet_}
    sttdict = {k.replace("resnet.fc.1", "resnet.fc"): v for k, v in sttdict.items()}
    model = models.ResNet(cfg["num_classes"])

    # Load the models for inference
    print(sttdict.keys())
    model.load_state_dict(sttdict, strict=True)
    _ = model.eval()

    logdir = "/home/mehdi/code/luminal" + "/results"

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    slide = Slide("/media/AprioricsSlides/" + cfg["slide"], backend="cucim")
    img = patch_from_coord(slide, cfg["x"], cfg["y"], level=1, patch_size=512)

    gradcam(img, model, logdir, num_classes=3)


if __name__ == "__main__":

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)

    main(cfg=config_file)
