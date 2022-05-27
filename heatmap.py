import argparse
import cv2
import json
import numpy as np
import os.path
from pathaia.util.types import Slide
from PIL import Image

DICT_SVS = {
    "0": "21I000004-1-03-1_135435.svs",
    "1": "21I000005-1-16-1_135140.svs",
    "2": "21I000006-1-09-1_134911.svs",
    "3": "21I000007-1-14-1_134640.svs",
    "4": "21I000008-1-02-1_134423.svs",
    "5": "21I000009-1-03-1_145327.svs",
    "6": "21I000245-1-14-1_134026.svs",
    "7": "21I000249-1-08-1_161359.svs",
    "8": "21I000263-1-06-1_133451.svs",
    "9": "21I000268-1-09-1_152552.svs",
}
DICT_T = {
    "0": "luminal_b",
    "1": "luminal_a",
    "2": "luminal_b",
    "3": "luminal_a",
    "4": "luminal_b",
    "5": "luminal_a",
    "6": "luminal_a",
    "7": "luminal_a",
    "8": "luminal_b",
    "9": "luminal_b",
}


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


# Add path to the config file to the command line arguments
# /data/DeepLearning/mehdi/log/luminal/resnet_293/luminal/f3c5a77006b74e168a89605663d65022/checkpoints/epoch=15-val_loss_ce=0.000.ckpt

parser.add_argument(
    "--model_name",
    type=str,
    default="resnet_305",
    help="size of the patches",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=27,
    help=" ",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=5,
    help=" ",
)
parser.add_argument(
    "--size",
    type=int,
    default=2,
    help=" ",
)
parser.add_argument(
    "--resize_ratio",
    type=int,
    default=32,
    help=" ",
)
parser.add_argument(
    "--blend_alpha",
    type=int,
    default=0.3,
    help=" ",
)
args = parser.parse_args()

epoch = args.epoch
model = args.model_name

size = args.size
resize_ratio = args.resize_ratio
blend_alpha = args.blend_alpha

if __name__ == "__main__":
    data_root = f"/data/DeepLearning/mehdi/log/luminal/{model}/"
    result_file = f"/data/DeepLearning/mehdi/log/luminal/{model}/result__{epoch}.json"  # "/home/mehdi/code/luminal/result.json"  # f"/data/DeepLearning/mehdi/log/luminal/{model}/result__{epoch}.json"

    with open(result_file, "r") as fp:
        result = json.load(fp)
    logdir = data_root + "/heatmap"

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # create heatmap for all the slides
    for i, slide_name in DICT_SVS.items():

        slide = f"/media/AprioricsSlides/{slide_name}"
        slide_he = Slide(slide, backend="cucim")

        preds = np.array(result["prediction_patch"][i]).squeeze()

        dict_pred = {
            "luminal_a": preds[::, 0],
            "luminal_b": preds[::, 1],
            "trash": preds[::, 2],
        }
        y_coords = np.array(result["pos_y"][i])
        x_coords = np.array(result["pos_x"][i])

        unique_x = np.unique(x_coords)
        unique_x.sort()
        delta = (unique_x[1:] - unique_x[:-1]).min()
        w = int((slide_he.dimensions[0]) / delta)
        h = int((slide_he.dimensions[1]) / delta)
        x_coords = x_coords // delta
        y_coords = y_coords // delta
        mask = np.full((h, w), 0, dtype=np.float64)

        # one heatmap by class
        for name, pred in dict_pred.items():
            hues = pred
            mask[(y_coords, x_coords)] = hues

            mask /= mask.max()

            heatmap = mask

            heatmap = cv2.resize(
                heatmap,
                (w * resize_ratio, h * resize_ratio),
                # interpolation=cv2.INTER_NEAREST,
            )
            img = slide_he.get_thumbnail((heatmap.shape[1], heatmap.shape[0])).resize(
                (heatmap.shape[1], heatmap.shape[0])
            )
            a = heatmap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            ii, jj = np.nonzero(a == 0)
            heatmap[ii, jj] = [255, 255, 255]

            superimposed_img = cv2.addWeighted(heatmap, 0.4, np.array(img), 0.6, 0)
            cv2.imwrite(
                f"{logdir}/{slide_name}_heatmap_for_{name}_true_value_{DICT_T[i]}_map.jpg",
                superimposed_img,
            )
            cv2.imwrite(
                f"{logdir}/{slide_name}.jpg",
                np.array(img),
            )

        mask = np.full((h, w), -1, dtype=np.float64)
        hues = ((preds.argmax(axis=1)) + 1) % 3
        mask[(y_coords, x_coords)] = hues

        heatmap = np.stack((mask,) * 3, axis=-1)

        # interpolate the heatmap
        heatmap = cv2.resize(
            heatmap,
            (w * resize_ratio, h * resize_ratio),
            interpolation=cv2.INTER_NEAREST,
        )
        img = slide_he.get_thumbnail((heatmap.shape[1], heatmap.shape[0])).resize(
            (heatmap.shape[1], heatmap.shape[0])
        )
        a = heatmap
        heatmap = np.uint8(85 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        print(len(np.nonzero(a == -1)))

        ii, jj, _ = np.nonzero(a == -1)
        heatmap[ii, jj] = [255, 255, 255]
        # heatmap = Image.fromarray(heatmap, "RGB")

        superimposed_img = cv2.addWeighted(heatmap, 0.4, np.array(img), 0.6, 0)
        cv2.imwrite(
            f"{logdir}/{slide_name}_heatmap_for_all_true_value_{DICT_T[i]}_map.jpg",
            # heatmap
            superimposed_img,
        )
        cv2.imwrite(
            f"{logdir}/{slide_name}.jpg",
            np.array(img),
        )
