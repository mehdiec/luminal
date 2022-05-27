import argparse
import json
import torch
from tqdm import tqdm
import yaml
from pytorch_lightning.utilities.seed import seed_everything
from src.preprocess import load_patches

from src.transforms import ToTensor
from src import models
from src.utils import progress_bar


def infer(model, loader, device=None):
    slide_info = {
        "idx": [],
        "y_hat": [],
        "true": [],
        "pos_x": [],
        "pos_y": [],
    }
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..

        for batch in tqdm(loader):
            image = batch["image"]
            slide_idx = batch["idx"]
            p_x = batch["pos_x"]
            p_y = batch["pos_y"]
            y_slide = batch["target_slide"]

            inputs = image.float()

            outputs = model(inputs)
            preds = torch.softmax(outputs, 1)
            print(slide_idx)
            print(p_x)

            if len(slide_info["idx"]) == 0:
                slide_info["idx"] = slide_idx
                slide_info["y_hat"] = preds
                slide_info["true"] = y_slide.int()
                slide_info["pos_x"] = p_x
                slide_info["pos_y"] = p_y
            # after first checking it is possible concatenate the tensors
            else:
                slide_info["idx"] = torch.cat((slide_info["idx"], slide_idx), 0)
                slide_info["y_hat"] = torch.cat((slide_info["y_hat"], preds), 0)
                slide_info["true"] = torch.cat((slide_info["true"], y_slide.int()), 0)
                slide_info["pos_x"] = torch.cat((slide_info["pos_x"], p_x), 0)
                slide_info["pos_y"] = torch.cat((slide_info["pos_y"], p_y), 0)

        sample = {
            "slide_idx": slide_info["idx"].cpu().numpy().tolist(),
            "prediction_patch": slide_info["y_hat"].cpu().numpy().tolist(),
            "pos_x": slide_info["pos_x"].cpu().numpy().tolist(),
            "pos_y": slide_info["pos_y"].cpu().numpy().tolist(),
        }
        # the results are saved in a json
        with open(f"./result.json", "w") as fp:
            json.dump(sample, fp)


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

    _, val_dl = load_patches(
        slide_file=cfg["slide_file"],
        noted=cfg["noted"],
        level=cfg["level"],
        transforms=transforms,
        normalize=cfg["normalize"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        patch_size=cfg["patch_size"],
        num_classes=cfg["num_classes"],
    )
    print("loaded")
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

    _ = model.eval()
    # creating unique log folder

    infer(model, val_dl)


if __name__ == "__main__":

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)

    main(cfg=config_file)
