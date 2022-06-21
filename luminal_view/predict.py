from re import X
import sys
from unittest import result

from utils import ellipse_axis_length, fitEllipse

# adding Folder_2/subfolder to the system path
sys.path.insert(0, "..")

import os.path
import torch


from torch.utils.data import DataLoader

from deep_learning.data_loader import SingleSlideInference
from deep_learning.transforms import ToTensor
from deep_learning import models


import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path
import seaborn as sns
import torch
from tqdm import tqdm
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from pathaia.patches.functional_api import slide_rois
from pathaia.util.types import Slide
from pathaia.patches import filter_thumbnail
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import to_pil_image
from pathaia.util.types import NDImage, NDGrayImage, NDByteImage
from torchvision.transforms.functional import to_tensor
from io import BytesIO
import base64
from typing import Callable, Optional, Any, List, Sequence, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
import shapely.geometry as sg
import shapely.ops as so
from shapely import affinity

import geopandas as gpd
import numpy.linalg as linalg
import csv
from PIL import Image

ID_TO_CLASS = {0: "Luminal A", 1: "Luminal B", 2: "other"}


class ToTensor(DualTransform):
    def __init__(
        self, transpose_mask: bool = False, always_apply: bool = True, p: float = 1
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> Dict[str, Callable[[NDImage], torch.Tensor]]:
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img: NDImage, **params) -> torch.Tensor:
        return to_tensor(img)

    def apply_to_mask(self, mask: NDImage, **params) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("transpose_mask",)


MAPPING = {
    "luminal A": 0,
    "luminal B": 1,
}
MAPPING_inv = {
    0: "luminal A",
    1: "luminal B",
}


def export_img(fig):
    img = BytesIO()
    fig.savefig(img, format="png")
    plt.close()
    plt.figure().clear()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode("ascii")


def export_img_cv2(img):
    _, buff = cv2.imencode(".png", img)
    return base64.b64encode(buff).decode("ascii")


def top(result, slide):

    top = {ID_TO_CLASS.get(i): [] for i in range(3)}

    for label in range(3):
        count = 0
        for i, pred in enumerate(result["prediction_patch"]):
            if np.array(pred).argmax() == label:

                top[ID_TO_CLASS.get(label)].append(
                    {
                        "prediction": {
                            ID_TO_CLASS.get(i): pred[i] for i in range(len(pred))
                        },
                        "image": export_img_cv2(
                            np.array(
                                slide.read_region(
                                    (result["pos_x"][i], result["pos_y"][i]),
                                    1,
                                    (512, 512),
                                )
                            )
                        ),
                        "pos_x": result["pos_x"][i],
                        "pos_y": result["pos_y"][i],
                    }
                )
                count += 1
            if count == 20:
                break
    return top
    # with open(logdir + "/result.json", "w") as fp:
    #     json.dump(top, fp)


def test(model, loader, device="cuda:0"):
    slide_info = {
        "y_hat": [],
        "pos_x": [],
        "pos_y": [],
    }
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..

        for batch in tqdm(loader):
            image = batch["image"].to(device)
            p_x = batch["pos_x"].to(device)
            p_y = batch["pos_y"].to(device)
            inputs = image.float()

            outputs = model(inputs)
            preds = torch.softmax(outputs, 1)

            if len(slide_info["y_hat"]) == 0:
                slide_info["y_hat"] = preds
                slide_info["pos_x"] = p_x
                slide_info["pos_y"] = p_y
            # after first checking it is possible concatenate the tensors
            else:
                slide_info["y_hat"] = torch.cat((slide_info["y_hat"], preds), 0)
                slide_info["pos_x"] = torch.cat((slide_info["pos_x"], p_x), 0)
                slide_info["pos_y"] = torch.cat((slide_info["pos_y"], p_y), 0)

        sample = {
            "prediction_patch": slide_info["y_hat"].cpu().numpy().tolist(),
            "pos_x": slide_info["pos_x"].cpu().numpy().tolist(),
            "pos_y": slide_info["pos_y"].cpu().numpy().tolist(),
        }

        # # the results are saved in a json
        # with open(f"./result.json", "w") as fp:
        #     json.dump(sample, fp)
        return sample


def piechart(result):

    l = np.array(result["prediction_patch"])
    tt = []
    for x in l:
        tt.append(x[:2])
    tt = np.array(tt)
    max_tt = np.array(tt).argmax(1)
    nb_dict = np.count_nonzero(max_tt > 0.5)
    data = [nb_dict, len(l) - nb_dict]

    fig = plt.figure(figsize=(6, 5))

    labels = ["luminal B", "luminal A"]

    # define Seaborn color palette to use
    colors = sns.color_palette("pastel")[0:5]

    # create pie chart

    plt.pie(data, labels=labels, colors=colors, autopct="%.0f%%")
    plt.tight_layout()

    # plt.title(f"Vrai valeur :    ")
    mean = tt.mean(0)
    mean = mean[:2]

    pred = mean / mean.sum()

    return {
        "prediction": {ID_TO_CLASS.get(i): pred[i] for i in range(len(pred))},
        "image": export_img(fig),
    }


def hist(result):

    temp_result = {}
    li = [np.array(result["prediction_patch"])[::, i] for i in range(3)]
    dict_a = {"Luminal A": li[0], "Luminal B": li[1], "other": li[2]}
    for i, l in enumerate(li):

        fig, ax = plt.subplots()
        sns.histplot(l, kde=True, stat="percent", ax=ax)
        ax.axvline(0.4, 0, 1, color="r", ls="--")
        plt.tight_layout()
        temp_result[ID_TO_CLASS.get(i)] = export_img(fig)
    fig, ax = plt.subplots()
    sns.histplot(dict_a, kde=True, stat="percent", ax=ax)
    ax.axvline(0.4, 0, 1, color="r", ls="--")
    plt.tight_layout()
    temp_result["wfull"] = export_img(fig)
    return temp_result


def heatmap(result, slide_name, resize_ratio=16):

    temp_result = {}

    slide = f"/media/AprioricsSlides/{slide_name}"
    slide_he = Slide(slide, backend="cucim")

    preds = np.array(result["prediction_patch"])

    dict_pred = {
        "Luminal A": preds[::, 0],
        "Luminal B": preds[::, 1],
        "other": preds[::, 2],
    }
    y_coords = np.array(result["pos_y"])
    x_coords = np.array(result["pos_x"])

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

        # superimposed_img = cv2.addWeighted(heatmap, 0.4, np.array(img), 0.6, 0)
        # temp_result.append(export_img_cv2(superimposed_img))
        temp_result[name] = export_img_cv2(heatmap)

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

    ii, jj, _ = np.nonzero(a == -1)
    heatmap[ii, jj] = [255, 255, 255]
    # heatmap = Image.fromarray(heatmap, "RGB")

    # superimposed_img = cv2.addWeighted(heatmap, 0.4, np.array(img), 0.6, 0)

    temp_result["Prediction"] = export_img_cv2(heatmap)
    temp_result["Raw Slide"] = export_img_cv2(np.array(img))
    return temp_result


# pas tres inteligent d appeler une classe dans une fonction
def gradcam(x, y, slide_name, model, num_classes=3):
    lesnet = torch.load(model)
    resnet = {k.replace("model.", ""): v for k, v in lesnet["state_dict"].items()}
    resnet_ = {
        k.replace("model.", "resnet."): v for k, v in lesnet["state_dict"].items()
    }
    sttdict = {**resnet, **resnet_}
    sttdict = {k.replace("resnet.fc.1", "resnet.fc"): v for k, v in sttdict.items()}
    model = models.ResNet(num_classes)

    # Load the models for inference
    model.load_state_dict(sttdict, strict=True)
    model.to("cuda:0")
    _ = model.eval()

    result_temp = {}

    target_layers = [model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    slide = Slide("/media/AprioricsSlides/" + slide_name)
    dimx, dimy = slide.dimensions
    if x < 1 and y < 1:
        x = dimx * x
        y = dimy * y

    image = slide.read_region((int(x), int(y)), 1, (512, 512)).convert("RGB")

    # Define a transform to convert PIL
    # image to a Torch tensor

    img_tensor = ToTensor()(image=np.array(image))
    img = img_tensor["image"].unsqueeze(0)

    input_tensor = img  # data_["image"] # Create an input tensor image for your model..
    input_device = input_tensor.to("cuda:0")
    outputs = model(input_device)
    preds = torch.softmax(outputs, 1)
    print(preds[0])
    prediction_list = preds[0].cpu().detach().numpy().tolist()
    print(prediction_list)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    for x in range(num_classes):
        targets = [ClassifierOutputTarget(x)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        rgb_img = to_pil_image(input_tensor.squeeze() / 255)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        img1 = visualization[:, :, [2, 1, 0]]
        img2 = np.array(to_pil_image(input_tensor.squeeze()))
        result_temp[ID_TO_CLASS.get(x)] = export_img_cv2(img1)
    result_temp["patch raw"] = export_img_cv2(img2)
    result_temp["prediction"] = {
        "Luminal A": prediction_list[0],
        "Luminal B": prediction_list[1],
        "other": prediction_list[2],
    }

    return result_temp


def predict(model_name, file_name):

    dict_result = {}

    lesnet = torch.load(model_name)
    resnet = {k.replace("model.", ""): v for k, v in lesnet["state_dict"].items()}

    # Init model
    model = models.build_model("resnet", num_classes=3)  # ,dropout=0.5)
    model.load_state_dict(resnet, strict=True)
    model.to("cuda:0")

    _ = model.eval()

    slide_name = file_name
    transforms = [ToTensor()]
    infds = SingleSlideInference(
        slide_name, level=1, patch_size=512, transforms=transforms
    )

    val_dl = DataLoader(infds, batch_size=32, num_workers=8, shuffle=True)
    print("loaded")
    # creating unique log folder

    size = 2
    resize_ratio = 32  # args.resize_ratio
    blend_alpha = 0.4  # args.blend_alpha

    result = test(model, val_dl)
    dict_result["hist"] = hist(result)
    dict_result["piechart"] = piechart(result)
    dict_result["heatmap"] = heatmap(result, slide_name)

    slide = Slide("/media/AprioricsSlides/" + slide_name)
    dict_result["top"] = top(result, slide)
    return dict_result

    # top(result, logdir)


patch_size = 2000


def patches(file, file_):
    slide = Slide("/media/AprioricsSlides/" + file + ".svs")
    slide_ = Slide("/media/AprioricsSlides/" + file_ + ".svs")

    top = {i: [] for i in range(2)}
    slides = [slide, slide_]
    files = [file, file_]
    for j, slid in enumerate(slides):

        with open(
            f"/home/mehdi/code/luminal/data/geojson_lum/{files[j]}.geojson", "r"
        ) as f:
            shape_dict = json.load(f)

        print(len(shape_dict))
        if not isinstance(shape_dict, list):
            roi_shapes = [sg.shape(shape_dict["geometry"])]
        else:
            roi_shapes = [sg.shape(shape_r["geometry"]) for shape_r in shape_dict]
            print("in")

        xmax, ymax = slid.dimensions[0], slid.dimensions[1]
        while len(top[j]) < 5:
            x = np.random.randint(0, xmax)
            y = np.random.randint(0, ymax)
            print(x, y, len(top[j]))
            pt1 = sg.Point(x, y)
            dx = pt1.x + patch_size
            dy = pt1.y + patch_size
            pt2 = sg.Point(dx, pt1.y)
            pt3 = sg.Point(pt1.x, dy)
            pt4 = sg.Point(dx, dy)
            patch_shape = sg.Polygon([pt1, pt2, pt4, pt3])
            for roi_shape in roi_shapes:

                if roi_shape.intersects(patch_shape):
                    print("in")
                    top[j].append(
                        {
                            "image": export_img_cv2(
                                np.array(
                                    slide.read_region(
                                        (x, y),
                                        0,
                                        (patch_size, patch_size),
                                    )
                                )
                            ),
                            "x": x,
                            "y": y,
                        }
                    )
                    break

    return {"top": top}
    # with open(logdir + "/result.json", "w") as fp:
    #     json.dump(top, fp)


def calculation(file, x, y):

    print(file)
    states = gpd.read_file(
        f"/media/AprioricsSlides/luminal/hovernet_outputs/geojson/{file}.geojson"
    )
    pt1 = sg.Point(int(x), int(y))
    dx = pt1.x + patch_size
    dy = pt1.y + patch_size
    pt2 = sg.Point(dx, pt1.y)
    pt3 = sg.Point(pt1.x, dy)
    pt4 = sg.Point(dx, dy)
    patch_shape = sg.Polygon([pt1, pt2, pt4, pt3])
    geom = []
    for i, roi_shape in enumerate(states.geometry):
        if roi_shape.intersects(patch_shape):
            geom.append(i)
    dff = states.iloc[np.array(geom)]
    df = dff.reset_index(drop=True)

    i = 10

    angle = []
    lista = []
    listb = []
    compactness = []

    area = []
    lenwit_ratio = []
    for i in tqdm(range(len(df))):
        # print(i)
        poly = df.geometry[i].simplify(0.05, preserve_topology=False)
        poly_0 = affinity.translate(
            sg.Polygon(poly),
            xoff=-sg.Polygon(poly).centroid.coords[0][0],
            yoff=-sg.Polygon(poly).centroid.coords[0][1],
        )
        minx, miny, maxx, maxy = poly_0.bounds
        rect_coord = [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]
        temp = poly_0.wkt[10:-2].split(", ")
        x_cor = []
        y_cor = []
        coor = []
        for xy in temp:
            x, y = xy.split(" ")

            x_cor.append(float(x))
            y_cor.append(float(y))
            coor.append([float(x), float(y)])
        X = np.array(coor)
        pca = PCA(n_components=2)
        _ = pca.fit(X).transform(X)

        # r3= Polygon(X_r)

        a, b = ellipse_axis_length(fitEllipse(np.array(x_cor), np.array(y_cor)))

        # print("a et b ", a, b)
        l = a, b

        # print("ps ", np.dot([1, 0], pca.components_[np.argmax(l)]))
        orientation = np.arccos(np.dot([1, 0], pca.components_[0])) * (180 / np.pi)
        # print("angle tah l epoque : ", orientation)
        TAH = orientation
        if pca.components_[0][1] < 0 and pca.components_[0][0] < 0:
            # print("cadrant 3")
            orientation = 180 - (
                np.arccos(np.dot([1, 0], pca.components_[0])) * (180 / np.pi)
            )
        elif pca.components_[0][0] < 0 and pca.components_[0][1] > 0:
            # print("cadrant 2")
            orientation = (
                np.arccos(np.dot([1, 0], pca.components_[0])) * (180 / np.pi)
            ) - 180

        elif pca.components_[0][0] > 0 and pca.components_[0][1] < 0:
            # print("cadrant 4")
            orientation = -np.arccos(np.dot([1, 0], pca.components_[0])) * (180 / np.pi)

        # print("angle vrai : ", orientation)
        aa = max(l)
        bb = min(l)
        angle.append(orientation)
        lista.append(aa)
        listb.append(bb)
        compactness.append((4 * np.pi * poly_0.area) / (poly_0.length) ** 2)
        area.append(poly_0.area)
        lenwit_ratio.append(aa / bb)
    lw = np.array(lenwit_ratio) + 0.0000000000001
    df["orientation"] = angle
    df["gd_ax"] = lista
    df["pt_ax"] = listb
    df["compactness"] = compactness
    df["area"] = area
    df["length_width_ratio"] = lenwit_ratio
    df["Eccentricity "] = np.sqrt(1 - 1 / lw)
    df["Assymetry "] = 1 - np.sqrt(1 / lw)
    df["smoothmess "] = poly_0.length / (4 * np.sqrt(poly_0.area))
    return df[df.columns[2:]]


def describe(df):
    return df.describe().to_html()


def make_graph(df):
    temp_result = {col: [] for col in df.columns}
    print(temp_result)

    for col in df.columns[:]:
        fig, ax = plt.subplots()
        print(col, df[col])
        sns.histplot(df[col], kde=True, stat="percent")
        ax.axvline(0.4, 0, 1, color="r", ls="--")
        plt.tight_layout()
        print(fig)
        temp_result[col] = export_img(fig)
    return temp_result


def statis(df):
    print("statuusss")
    return {"graph": make_graph(df), "describe": describe(df)}
