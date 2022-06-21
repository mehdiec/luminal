import argparse
import os.path
import torch
import yaml

from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from deep_learning.data_loader import SingleSlideInference
from deep_learning.transforms import ToTensor
from deep_learning import models
from utils import hist, top, test, heatmap, piechart


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
    default="./config.yaml",
    help="path to config file",
)
args = parser.parse_args()


def main(cfg):

    print("main")

    seed_everything(workers=True)

    # check for transformation
    transforms = [ToTensor()]
    name = cfg["checkpoint_name"]

    lesnet = torch.load(name)
    resnet = {k.replace("model.", ""): v for k, v in lesnet["state_dict"].items()}

    # Init model
    model = models.build_model(
        cfg["model"],
        cfg["num_classes"],
        cfg["freeze"],
        cfg["pretrained"],
    )
    model.load_state_dict(resnet, strict=True)
    model.to("cuda:0")

    _ = model.eval()

    slide_name = cfg["slide_name"]

    infds = SingleSlideInference(
        slide_name, level=1, patch_size=512, transforms=transforms
    )

    val_dl = DataLoader(
        infds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print("loaded")
    # creating unique log folder

    size = cfg["size"]  # 2
    resize_ratio = cfg["resize_ratio"]  # 32# args.resize_ratio
    blend_alpha = cfg["blend_alpha"]  # 0.4# args.blend_alpha

    logdir = "/home/mehdi/code/luminal" + "/results"

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    result = test(model, val_dl)

    top(result, logdir)

    piechart(result, logdir)
    hist(result, logdir)
    heatmap(result, size, resize_ratio, blend_alpha, logdir, slide_name)


if __name__ == "__main__":

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)

    main(cfg=config_file)
