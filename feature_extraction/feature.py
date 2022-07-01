# adding Folder_2/subfolder to the system path
import base64
import sys

from matplotlib.pyplot import gray

sys.path.insert(0, "..")
import csv
from glob import glob
import json
import cv2
import mahotas
import numpy as np
import pandas as pd
import shapely.geometry as sg
from shapely import affinity
import argparse
import geopandas as gpd
import yaml
from tqdm import tqdm
import skimage.color
from pathaia.util.types import Slide
from deep_learning.transforms import StainAugmentor
import rasterio.features
from feature_extraction.utils import coarseness, contrast, directionality, iou
from albumentations import (
    Flip,
    Transpose,
    RandomBrightnessContrast,
    Compose,
    BasicTransform,
)
from pathaia.util.basic import ifnone

patch_size = 0
file = ""
x, y = 0, 0
har_ft = [
    "Angular Second Moment",
    "Contrast",
    "Correlation",
    "Sum of Squares: Variance",
    "Inverse Difference Moment",
    "Sum Average",
    "Sum Variance",
    "Sum Entropy",
    "Entropy",
    "Difference Variance",
    "Difference Entropy",
    "Information Measure of Correlation 1",
    "Information Measure of Correlation 2",
]
hu_ft = [f"hu_{i}" for i in range(7)]

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


def export_img_cv2(img):
    _, buff = cv2.imencode(".png", img)
    return base64.b64encode(buff).decode("ascii")


# Add path to the config file to the command line arguments
parser.add_argument(
    "--path_to_config",
    type=str,
    required=False,
    default="./slide_ft_config.yaml",
    help="path to config file",
)
args = parser.parse_args()


def main(cfg, file, states, x, y):
    result = {}
    har = {key: [] for key in har_ft}
    hu = {key: [] for key in hu_ft}

    df_label = pd.read_csv("/data/DeepLearning/mehdi/csv/luminal_data_split.csv")
    print(df_label[df_label["id"] == f"/media/AprioricsSlides/{file}.svs"].ab.values)
    if (
        len(df_label[df_label["id"] == f"/media/AprioricsSlides/{file}.svs"].ab.values)
        > 0
    ):
        print(
            "ddd,",
            df_label[df_label["id"] == f"/media/AprioricsSlides/{file}.svs"].ab.values,
        )

        label = df_label[
            df_label["id"] == f"/media/AprioricsSlides/{file}.svs"
        ].ab.values[0]
    else:
        label = -1
    print(label)
    states = states
    slide_he = Slide(
        "/media/AprioricsSlides/" + file + ".svs",
        backend="cucim",
    )
    print(x, y)
    out = slide_he.read_region(
        (x, y), 0, (cfg["patch_size"], cfg["patch_size"])
    ).convert("RGB")

    if cfg["transform"] == "bright":
        transforms = [
            RandomBrightnessContrast(
                brightness_limit=[0.19, 0.2],
                contrast_limit=0,
                brightness_by_max=True,
                always_apply=False,
                p=1,
            )
        ]
        transforms = Compose(ifnone(transforms, []))
        image = transforms(image=np.array(out))["image"]

    elif cfg["transform"] == "contrast":

        transforms = [
            RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=[0.19, 0.2],
                brightness_by_max=True,
                always_apply=False,
                p=1,
            ),
        ]
        transforms = Compose(ifnone(transforms, []))
        image = transforms(image=np.array(out))["image"]
    elif cfg["transform"] == "stain":

        transforms = [
            StainAugmentor(),
        ]
        transforms = Compose(ifnone(transforms, []))
        image = transforms(image=np.array(out))["image"]

    elif cfg["transform"] == "transpose":
        image = Transpose(
            always_apply=True,
            p=1,
        ).apply(np.array(out))
    elif cfg["transform"] == "flip":
        image = Flip(
            always_apply=True,
            p=1,
        ).apply(np.array(out))
    else:

        image = np.array(out)

    pt1 = sg.Point(x, y)
    dx = pt1.x + cfg["patch_size"]
    dy = pt1.y + cfg["patch_size"]
    pt2 = sg.Point(dx, pt1.y)
    pt3 = sg.Point(pt1.x, dy)
    pt4 = sg.Point(dx, dy)
    patch_shape = sg.Polygon([pt1, pt2, pt4, pt3])
    geom = []
    for i, roi_shape in enumerate(states.geometry):
        intersect = roi_shape.intersection(patch_shape)
        if roi_shape.intersects(patch_shape):
            if intersect.area / roi_shape.area > 0.99:
                geom.append(i)
    dff = states.iloc[np.array(geom)]
    df = dff.reset_index(drop=True)

    angle = []
    lista = []
    listb = []
    compactness = []
    geom = []
    smoothmess = []
    area = []
    lenwit_ratio = []
    nb_bump = []
    coarseness_ = []
    contrast_ = []
    directionality_ = []
    iou_ = []
    h_mean = []
    h_std = []
    e_mean = []
    e_std = []
    hue_mean = []
    hue_std = []
    letters = []

    for i in tqdm(range(len(df))):

        poly = df.geometry[i]  # .simplify(0.05, preserve_topology=False)
        poly_0 = affinity.translate(
            sg.Polygon(poly),
            xoff=-pt1.x,
            yoff=-pt1.y,
        )
        geom.append(poly_0.boundary)
        temp = poly_0.wkt[10:-2].split(", ")
        x_cor = []
        y_cor = []
        coor = []
        for xy in temp:
            x, y = xy.split(" ")
            try:
                y_ = float(y)
            except:
                y_ = float(y[:-1])
            try:
                x_ = float(x)
            except:
                x_ = float(x[1:])

            y_cor.append(y_)
            x_cor.append(x_)
            coor.append([[x_, y_]])

        _, l, orientation = cv2.fitEllipse(np.array(coor, dtype=np.int32))
        # print("open cv methode", l, orientation)
        [x, y, w, h] = cv2.boundingRect(np.array(coor, dtype=np.int32))
        # discard areas that are too large

        if w < 1 or h < 1 or x < 0 or y < 0:

            angle.append(np.nan)
            lista.append(np.nan)
            listb.append(np.nan)
            compactness.append(np.nan)
            geom.append(np.nan)
            area.append(np.nan)
            lenwit_ratio.append(np.nan)
            nb_bump.append(np.nan)
            coarseness_.append(np.nan)
            contrast_.append(np.nan)
            directionality_.append(np.nan)
            smoothmess.append(np.nan)
            iou_.append(np.nan)
            h_mean.append(np.nan)
            h_std.append(np.nan)
            e_mean.append(np.nan)
            e_std.append(np.nan)
            hue_mean.append(np.nan)
            hue_std.append(np.nan)
            letters.append(np.nan)
            for key in har_ft:
                har[key].append(np.nan)
            for key in hu_ft:
                hu[key].append(np.nan)

        else:

            img = rasterio.features.rasterize([poly_0], out_shape=(2000, 2000))[
                y : y + h, x : x + w
            ]

            image_gray = cv2.cvtColor(
                np.array(
                    slide_he.read_region(
                        (x, y), 0, (cfg["patch_size"], cfg["patch_size"])
                    ).convert("RGB")
                )[y : y + h, x : x + w],
                cv2.COLOR_BGR2GRAY,
            )  # grayscale

            mask = img == 0
            gray_0 = image_gray
            a_gray = np.ma.masked_array(data=image_gray, mask=mask, fill_value=999999)

            gray_0 = image_gray
            gray_0[mask] = 0

            mask = np.stack(((img == 0),) * 3, axis=-1)
            a = np.ma.masked_array(
                data=image[y : y + h, x : x + w], mask=mask, fill_value=999999
            )
            # discard areas that are too large

            letter = a
            dezoom = 70
            ymin = max(y - dezoom, 0)
            ymax = min(y + dezoom + h, 1999)
            xmin = max(x - dezoom, 0)
            xmax = min(x + w + dezoom, 1999)

            img_letter = rasterio.features.rasterize(
                [poly_0.boundary], out_shape=(2000, 2000)
            )
            mask = np.stack(((img_letter == 1),) * 3, axis=-1)
            image_tmp = np.array(
                slide_he.read_region(
                    (x, y), 0, (cfg["patch_size"], cfg["patch_size"])
                ).convert("RGB")
            )

            image_tmp[mask] = 0

            letterr = image_tmp[ymin:ymax, xmin:xmax]
            grey_nuc = a_gray

            # HUMoments for shape
            image_hu = cv2.HuMoments(cv2.moments(grey_nuc)).flatten()
            # Haralick for texture

            if grey_nuc.shape[0] > 1 and grey_nuc.shape[1] > 1:

                image_har = mahotas.features.haralick(gray_0.astype(int)).mean(axis=0)
            else:
                image_har = [np.nan for _ in range(len(har_ft))]
            for k, key in enumerate(har_ft):

                har[key].append(image_har[k])
            for k, key in enumerate(hu_ft):
                hu[key].append(image_hu[k])

            aa = max(l)
            bb = min(l)

            aa = max(l)
            bb = min(l)
            angle.append(orientation)
            lista.append(aa)
            listb.append(bb)
            compactness.append((4 * np.pi * poly_0.area) / (poly_0.length) ** 2)
            area.append(poly_0.area)
            lenwit_ratio.append(aa / bb)
            smoothmess.append(poly_0.length / (4 * np.sqrt(poly_0.area)))
            c = np.array(coor, dtype=np.int32)

            try:
                tmp = len(
                    cv2.convexityDefects(c, cv2.convexHull(c, returnPoints=False))
                )
            except:
                tmp = np.nan

            nb_bump.append(tmp)
            coarseness_.append(coarseness(grey_nuc, 5))
            contrast_.append(contrast(grey_nuc))
            directionality_.append(directionality(grey_nuc))
            cvx_shape = sg.Polygon(cv2.convexHull(c).squeeze())
            shape = sg.Polygon(c.squeeze())
            iou_.append(iou(shape, cvx_shape))
            he_image = skimage.color.rgb2hed(np.array(letter))
            h_mean.append(np.mean(he_image[:, :, 0].flatten()))
            h_std.append(np.std(he_image[:, :, 0].flatten()))
            e_mean.append(np.mean(he_image[:, :, 1].flatten()))
            e_std.append(np.std(he_image[:, :, 1].flatten()))
            try:
                lab_image = cv2.cvtColor(np.array(letter), cv2.COLOR_RGB2HSV)
                hue_mean.append(np.mean(lab_image[:, :, 0].flatten()))
                hue_std.append(np.std(lab_image[:, :, 0].flatten()))
            except:
                print(np.array(letter).shape)
                hue_mean.append(np.nan)
                hue_std.append(np.nan)
            letters.append(export_img_cv2(letterr))

    lw = np.array(lenwit_ratio) + 0.0000000000001
    df["orientation"] = list(angle)
    df["gd_ax"] = list(lista)
    df["pt_ax"] = list(listb)
    df["compactness"] = list(compactness)
    df["area"] = list(area)
    df["length_width_ratio"] = list(lenwit_ratio)
    df["Eccentricity"] = list(np.sqrt(1 - 1 / lw))
    df["Assymetry"] = list(1 - np.sqrt(1 / lw))
    df["smoothmess"] = list(smoothmess)
    df["coarseness"] = coarseness_
    df["contrast"] = contrast_
    df["directionality"] = directionality_
    df["nb_bump"] = nb_bump
    df["iou"] = iou_
    df["h_mean"] = h_mean
    df["h_std"] = h_std
    df["e_mean"] = e_mean
    df["e_std"] = e_std
    df["hue_mean"] = hue_mean
    df["hue_std"] = hue_std
    df["image"] = letters

    for key in har_ft:
        df[key] = har[key]
    for key in hu_ft:
        df[key] = hu[key]
    result["slide"] = file
    result["coordinate"] = [x, y]
    result["label"] = label
    result["features"] = df[df.columns[2:]].to_dict()
    return result


if __name__ == "__main__":

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)

    with open("/data/DeepLearning/mehdi/csv/patches.csv", "r") as patch_file:
        # reader = csv.DictReader(patch_file)
        reader = csv.DictReader(patch_file, delimiter=",")
        for j, row in enumerate(reader):
            file = row.get("id").split("/")[-1].split(".")[0]
            try:
                states = gpd.read_file(
                    f"/media/AprioricsSlides/luminal/hovernet_outputs/geojson/{file}.geojson"
                )
            except:
                continue
            print(file)

            # glob("/media/AprioricsSlides/geojsons/" + file.split("_")[0] + "*")[0]
            # print(state)
            # states = gpd.read_file(state)
            for transformation in [
                "",
            ]:

                x, y = int(row.get("x")), int(row.get("y"))
                print(row)
                config_file["transform"] = transformation

                result = main(config_file, file, states, x, y)
                result["info"] = row.get("info")
                print(row.get("info"))

                with open(
                    f"/data/DeepLearning/mehdi/features/{file}_{x}_{y}_{transformation}.json",
                    "w",
                ) as fp:
                    json.dump(result, fp)
