import csv
import json
import cv2
import numpy as np
import pandas as pd
import argparse
import yaml
from pathaia.util.types import Slide
from tqdm import tqdm

from glob import glob
from constante import COLORS, EPSILON, GEOMETRIES
from utils import (
    blue,
    bounding_box,
    contour_centered,
    convex_area,
    e_mean,
    e_std,
    elipsea,
    elipseb,
    get_color_nuc,
    get_gray_scale_nuc,
    get_hu,
    glcm,
    green,
    h_mean,
    h_std,
    hue_mean,
    hue_std,
    in_patch,
    length,
    radiuses,
    red,
    saturation_mean,
    saturation_std,
    valid_bbox,
    value_hsv_mean,
    value_hsv_std,
    get_displayde_imaged,
)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--path_to_config",
    type=str,
    required=False,
    default="configs/slide_ft_config.yaml",
    help="path to config file",
)
args = parser.parse_args()


def main(
    cfg,
    x_patch,
    y_patch,
    IMAGE,
    IMAGE_GRAY,
    df,
):

    p_size = cfg["patch_size"]
    dx = x + p_size
    dy = y + p_size
    countour_patchs = np.array([[x, y], [x, dy], [dx, dy], [dx, y]])
    contours_np = [np.array(elt) for elt in df["contour"].tolist()]

    # Preprocess
    df["contours_np"] = contours_np
    bboxs_np = [np.array(elt) for elt in df["bbox"].tolist()]
    df["bbox_np"] = bboxs_np
    df["contour_centered"] = df["contours_np"].apply(
        contour_centered, x_patch=x_patch, y_patch=y_patch
    )
    df["bbox_centered"] = df["contour_centered"].apply(cv2.boundingRect)
    df = df[df["centroid"].apply(in_patch, countour_patch=countour_patchs)]
    df["x,y,w,h"] = df["bbox_np"].apply(bounding_box, x_patch=x_patch, y_patch=y_patch)
    df = df[df["x,y,w,h"].apply(valid_bbox, p_size=p_size)]

    # Geom
    df["area"] = df["contours_np"].apply(cv2.contourArea)
    df["gd_ax"] = df["contours_np"].apply(elipsea)
    df["pt_ax"] = df["contours_np"].apply(elipseb)
    df["radius"] = df["x,y,w,h"].apply(radiuses)
    df["length"] = df["contours_np"].apply(length)
    df["compactness"] = (4 * np.pi * df["area"]) / ((df["length"]) ** 2 + EPSILON)
    df["eliptic_fit"] = df["area"] / (np.pi * df["gd_ax"] * df["pt_ax"] + EPSILON)
    df["enclosing_circle_overlap"] = df["area"] / (
        np.pi * df["radius"] * df["radius"] + EPSILON
    )
    df["length_width_ratio"] = df["gd_ax"] / df["pt_ax"]
    df["Eccentricity"] = np.sqrt(1 - 1 / df["length_width_ratio"])
    df["smoothmess"] = df["length"] / (4 * np.sqrt(df["area"]))
    df["convex_area"] = df["contours_np"].apply(convex_area)
    df["iou"] = df["area"] / df["convex_area"]

    # nuc images
    df["nuc_img"] = df["bbox_centered"].progress_apply(
        get_color_nuc, image=IMAGE, p_size=p_size
    )
    df["gscale_img"] = df["bbox_centered"].progress_apply(
        get_gray_scale_nuc, image=IMAGE_GRAY, p_size=p_size
    )
    if cfg["display"] == True:
        df["image"] = df.progress_apply(
            get_displayde_imaged, image=IMAGE, p_size=p_size
        )

    # color feature
    df["hue_mean"] = df["nuc_img"].progress_apply(hue_mean)
    df["hue_std"] = df["nuc_img"].progress_apply(hue_std)
    df["saturation_mean"] = df["nuc_img"].progress_apply(saturation_mean)
    df["value_hsv_mean"] = df["nuc_img"].progress_apply(value_hsv_mean)
    df["saturation_std"] = df["nuc_img"].progress_apply(saturation_std)
    df["value_hsv_std"] = df["nuc_img"].progress_apply(value_hsv_std)
    df["normalized_red_intensity"] = df["nuc_img"].progress_apply(red)
    df["normalized_green_intensity"] = df["nuc_img"].progress_apply(green)
    df["normalized_blue_intensity"] = df["nuc_img"].progress_apply(blue)
    df["h_mean"] = df["nuc_img"].progress_apply(h_mean)
    df["e_mean"] = df["nuc_img"].progress_apply(e_mean)
    df["h_std"] = df["nuc_img"].progress_apply(h_std)
    df["e_std"] = df["nuc_img"].progress_apply(e_std)

    # texture feature
    df["hus"] = df["gscale_img"].progress_apply(get_hu)
    df["flcm"] = df["gscale_img"].progress_apply(glcm)

    return df


if __name__ == "__main__":
    tqdm.pandas()

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)
    p_size = config_file["patch_size"]

    input_folder = config_file["input_folder"]  # /media/AprioricsSlides/
    output_folder = config_file["output_folder"]  # //data/DeepLearning/mehdi/_new/
    image_folder = config_file["image_folder"]  # /data/DeepLearning/mehdi/_new/
    image_extension = config_file["image_extension"]  # ".svs"
    segmentation_folder = config_file["segmentation_folder"]  # ".svs"
    # /data/DeepLearning/mehdi/patch_csv/patch_csvs/0/{p_size}/
    csv_files = glob(input_folder + f"*")
    for csv_file in csv_files[:]:

        print(csv_file)
        file = csv_file.split("/")[-1].split(".")[0]
        slide_he = Slide(image_folder + file + image_extension)
        with open(
            segmentation_folder + f"{file}.json",
            "r",
        ) as f:
            shape_dict = json.load(f)["nuc"]
        df = pd.json_normalize(
            shape_dict.values(),
        )
        with open(csv_file, "r") as patch_file:

            reader = csv.DictReader(patch_file, delimiter=",")

            for j, row in enumerate(reader):
                x, y = int(row.get("x")), int(row.get("y"))
                out = slide_he.read_region(
                    (x, y), 0, (config_file["patch_size"], config_file["patch_size"])
                ).convert("RGB")

                out = cv2.cvtColor(
                    np.array(out),
                    cv2.COLOR_RGB2BGR,
                )

                IMAGE = np.array(out)

                IMAGE_GRAY = cv2.cvtColor(
                    IMAGE,
                    cv2.COLOR_BGR2GRAY,
                )  # grayscale

                for transformation in [
                    "",
                ]:
                    print(row)
                    if row.get("info") == 2:
                        continue
                    if output_folder + f"{file}_{x}_{y}_{transformation}.csv" in glob(
                        output_folder + "*"
                    ):
                        continue

                    x, y = int(row.get("x")), int(row.get("y"))

                    config_file["transform"] = transformation

                    result = main(
                        cfg=config_file,
                        x_patch=x,
                        y_patch=y,
                        IMAGE=IMAGE,
                        IMAGE_GRAY=IMAGE_GRAY,
                        df=df.copy(),
                    )
                    result["slide"] = file

                    col = (
                        GEOMETRIES
                        + COLORS
                        + [
                            "centroid",
                            "type_prob",
                            "type",
                            "radius",
                            "flcm",
                            "hus",
                            "slide",
                        ]
                    )
                    result[col].to_csv(
                        output_folder + f"{file}_{x}_{y}_{transformation}.csv"
                    )

                    print("saved")
