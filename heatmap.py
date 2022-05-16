import cv2
import json
import numpy as np

from pathaia.util.types import Slide


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


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


epoch = 6
model = "resnet_72"
epoch = 5
model = "resnet_70"
epoch = 5
model = "resnet_74"
epoch = 2
model = "resnet_76"
epoch = 8
model = "resnet_63"
epoch = 16
model = "resnet_85"
size = 299
resize_ratio = 4
blend_alpha = 0.4

with open(
    f"/data/DeepLearning/mehdi/log/luminal/{model}/result__{epoch}.json", "r"
) as fp:
    result = json.load(fp)

for i, slide_name in DICT_SVS.items():

    slide = f"/media/AprioricsSlides/{slide_name}"
    # slide = "21I000004"
    slide_he = Slide(slide, backend="cucim")

    pred = np.array(result["prediction_patch"][i]).squeeze()
    y_coords = np.array(result["pos_y"][i])
    x_coords = np.array(result["pos_x"][i])

    unique_x = np.unique(x_coords)
    unique_x.sort()
    delta = (unique_x[1:] - unique_x[:-1]).min()
    w = int((slide_he.dimensions[0] - (size - delta)) / delta)
    h = int((slide_he.dimensions[1] - (size - delta)) / delta)
    x_coords = x_coords // delta
    y_coords = y_coords // delta
    mask = np.full((h, w), -1.0, dtype=np.float64)

    hues = sigmoid(pred)
    mask[(y_coords, x_coords)] = hues

    # average the channels of the activations

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    mask /= mask.max()

    heatmap = mask
    # draw the heatmap
    # plt.matshow(heatmap)

    # make the heatmap to be a numpy array

    # interpolate the heatmap
    heatmap = cv2.resize(heatmap, (w * resize_ratio, h * resize_ratio))
    img = slide_he.get_thumbnail((heatmap.shape[1], heatmap.shape[0])).resize(
        (heatmap.shape[1], heatmap.shape[0])
    )
    a = heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * blend_alpha + img
    cv2.imwrite(
        f"/data/DeepLearning/mehdi/log/luminal/{model}/0/_map_{slide_name}.jpg",
        superimposed_img,
    )
    # plt.matshow(superimposed_img)
