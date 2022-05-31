import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path
import seaborn as sns
import torch
from tqdm import tqdm

from pathaia.patches.functional_api import slide_rois
from pathaia.util.types import Slide
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

from src.transforms import ToTensor


MAPPING = {
    "luminal A": 0,
    "luminal B": 1,
}
MAPPING_inv = {
    0: "luminal A",
    1: "luminal B",
}


def top(result, logdir):
    logdir = logdir + "/top"

    top = {i: [] for i in range(3)}

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    for label in range(3):
        count = 0
        for i, pred in enumerate(result["prediction_patch"]):
            if np.array(pred).argmax() == label:
                top[label].append(
                    {
                        "prediction": pred,
                        "pos_x": result["pos_x"][i],
                        "pos_y": result["pos_y"][i],
                    }
                )
                count += 1
            if count == 20:
                break
    with open(logdir + "/result.json", "w") as fp:
        json.dump(top, fp)
    return top


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
        # the results are saved in a json
        with open(f"./result.json", "w") as fp:
            json.dump(sample, fp)
        return sample


def piechart(result, logdir):
    logdir = logdir + "/pie"

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    l = np.array(result["prediction_patch"])
    tt = []
    for x in l:
        tt.append(x[:2])
    tt = np.array(tt)
    max_tt = np.array(tt).argmax(1)
    nb_dict = np.count_nonzero(max_tt > 0.5)
    data = [nb_dict, len(l) - nb_dict]

    plt.figure(figsize=(12, 10))

    labels = ["luminal B", "luminal A"]

    # define Seaborn color palette to use
    colors = sns.color_palette("pastel")[0:5]

    # create pie chart

    plt.pie(data, labels=labels, colors=colors, autopct="%.0f%%")
    plt.title(
        "slide_name"
        + f" \n prediction moyenne: {tt.mean(0) }  {MAPPING_inv.get(tt.mean(0).argmax() )}\n   Vrai valeur :     ",
    )
    # plt.title(f"Vrai valeur :    ")
    plt.savefig(logdir + f"/output.png")

    print(
        f"Nombre de patch predit comme luminal B:{data[0]}    vrai valeur       , nombre de patch pour la slide {len(l)}     .\n"
    )
    print(
        "_____________________________________________________________________________________________________________________________"
    )

    return


def hist(result, logdir):
    logdir = logdir + "/hist"

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    li = [np.array(result["prediction_patch"])[::, i] for i in range(3)]
    for i, l in enumerate(li):
        x = l
        # x= result["prediction_slide"]
        # sns.histplot(data=x,  stat="percent")
        sns.displot(x, kde=True, stat="percent")
        plt.axvline(0.4, 0, 1, color="r", ls="--")
        # plt.savefig(lum+"_"+a+".pdf",format="pdf")
        plt.savefig(logdir + f"/output{i}.png")


def heatmap(result, size, resize_ratio, blend_alpha, logdir, slide_name):
    logdir = logdir + "/heatmap"

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    slide = f"/media/AprioricsSlides/{slide_name}"
    slide_he = Slide(slide, backend="cucim")

    preds = np.array(result["prediction_patch"])

    dict_pred = {
        "luminal_a": preds[::, 0],
        "luminal_b": preds[::, 1],
        "trash": preds[::, 2],
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

        superimposed_img = cv2.addWeighted(heatmap, 0.4, np.array(img), 0.6, 0)
        print(f"{logdir}/{slide_name}_heatmap_for_{name}_true_value__map.jpg")
        cv2.imwrite(
            f"{logdir}/{slide_name}_heatmap_for_{name}_true_value__map.jpg",
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
        f"{logdir}/{slide_name}_heatmap_for_all_true_value__map.jpg",
        # heatmap
        superimposed_img,
    )
    cv2.imwrite(
        f"{logdir}/{slide_name}.jpg",
        np.array(img),
    )


def patch_from_coord(slide, x, y, level=1, patch_size=512):
    patches = slide_rois(
        slide,
        level,
        psize=patch_size,
        interval=0,
        thumb_size=2000,
    )
    for patch in patches:
        if (
            x - 511 < patch[0].position.x < x + 511
            and y - 511 < patch[0].position.y < y + 511
        ):

            img = patch[1]
    return img


# pas tres inteligent d appeler une classe dans une fonction
def gradcam(img, model, logdir, num_classes=3):

    target_layers = [model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

    logdir = logdir + "/map"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    image = img

    # Define a transform to convert PIL
    # image to a Torch tensor

    img_tensor = ToTensor()(image=image)
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
        dst = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)

        fig, ax = plt.subplots(figsize=(20, 10))

        ax.matshow(dst)
        plt.title(f"map_label_{x}")
        plt.imsave(f"{logdir}/a_map_{x}_.png", dst)
    plt.imsave(f"{logdir}/a{x}_.png", img2)
