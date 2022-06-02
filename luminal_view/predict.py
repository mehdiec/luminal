from re import X
import sys
from unittest import result

# adding Folder_2/subfolder to the system path
sys.path.insert(0, "..")

import os.path
import torch


from torch.utils.data import DataLoader

from src.data_loader import SingleSlideInference
from src.transforms import ToTensor
from src import models


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

ID_TO_CLASS = {0:"Luminal A",1:"Luminal B",2:"other"}

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
                        "prediction": {ID_TO_CLASS.get(i):pred[i] for i in range(len(pred))},
                        "image": export_img_cv2(np.array(slide.read_region( (result["pos_x"][i], result["pos_y"][i]), 1, (512, 512))  )
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
    pred = list(tt.mean(0))
    return {"prediction": {ID_TO_CLASS.get(i):pred[i] for i in range(len(pred))},"image":export_img(fig)}


def hist(result):

   
    temp_result = {}
    li = [np.array(result["prediction_patch"])[::, i] for i in range(3)]
    for i, l in enumerate(li):

        fig, ax = plt.subplots()
        sns.histplot(l, kde=True, stat="percent", ax=ax)
        ax.axvline(0.4, 0, 1, color="r", ls="--")
        plt.tight_layout()
        temp_result[ID_TO_CLASS.get(i)] = (export_img(fig))
    return temp_result


def heatmap(result, slide_name, resize_ratio=8):

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


def patch_from_coord(slide, x, y, level=1, patch_size=512):
    patches = slide_rois(
        slide,
        level,
        psize=patch_size,
        interval=0,
        thumb_size=2000,
        slide_filters=[filter_thumbnail],
    )

 
    for patch in patches:
        if (
            x - 511 < patch[0].position.x < x + 511
            and y - 511 < patch[0].position.y < y + 511
        ):

            img = patch[1]
    return img


# pas tres inteligent d appeler une classe dans une fonction
def gradcam(x,y,slide_name, model,  num_classes=3):
    lesnet = torch.load(model)
    resnet = {k.replace("model.", ""): v for k, v in lesnet["state_dict"].items()}
    resnet_ = {
        k.replace("model.", "resnet."): v for k, v in lesnet["state_dict"].items()
    }
    sttdict = {**resnet, **resnet_}
    sttdict = {k.replace("resnet.fc.1", "resnet.fc"): v for k, v in sttdict.items()}
    model = models.ResNet(num_classes)

    # Load the models for inference
    print(sttdict.keys())
    model.load_state_dict(sttdict, strict=True)
    _ = model.eval()

    result_temp = {}

    target_layers = [model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    slide = Slide("/media/AprioricsSlides/" + slide_name)
    dimx,dimy = slide.dimensions
    if x<1 and y<1:
        x= dimx*x
        y= dimy*y

 

    image = slide.read_region( (int(x), int(y)), 1, (512, 512)).convert("RGB")

    # Define a transform to convert PIL
    # image to a Torch tensor

    img_tensor = ToTensor()(image=np.array(image))
    img = img_tensor["image"].unsqueeze(0)

    input_tensor = img  # data_["image"] # Create an input tensor image for your model..

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
        img1 = visualization
        img2 = np.array(to_pil_image(input_tensor.squeeze()))
        result_temp[ID_TO_CLASS.get(x)]=(export_img_cv2(img1))
    result_temp["patch raw"]=(export_img_cv2(img2)) 
    return result_temp


def predict(model_name, file_name):

    dict_result = {}

    lesnet = torch.load(model_name)
    resnet = {k.replace("model.", ""): v for k, v in lesnet["state_dict"].items()}

    # Init model
    model = models.build_model("resnet", num_classes=3)
    model.load_state_dict(resnet, strict=True)
    model.to("cuda:0")

    _ = model.eval()

    slide_name = file_name
    transforms = [ToTensor()]
    infds = SingleSlideInference(
        slide_name, level=1, patch_size=512, transforms=transforms
    )

    val_dl = DataLoader(
        infds,
        batch_size=32,
        num_workers=8,
    )
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
    dict_result["top"] = top(result,slide)
    return dict_result

    # top(result, logdir)

 