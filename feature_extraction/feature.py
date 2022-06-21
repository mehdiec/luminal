import json
import cv2
import numpy as np
from sklearn.decomposition import PCA
import shapely.geometry as sg
from shapely import affinity
import argparse
import geopandas as gpd
import yaml
from tqdm import tqdm

from utils import ellipse_axis_length, fitEllipse

patch_size = 0
file = ""
x, y = 0, 0


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


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
    result = {}
    file = cfg["file"]
    states = gpd.read_file(
        f"/media/AprioricsSlides/luminal/hovernet_outputs/geojson/{file}.geojson"
    )
    pt1 = sg.Point(cfg["x"], cfg["y"])
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
    skr = []
    geom = []

    area = []
    lenwit_ratio = []
    for i in tqdm(len(df)):

        poly = df.geometry[i]  # .simplify(0.05, preserve_topology=False)
        poly_0 = affinity.translate(
            sg.Polygon(poly),
            xoff=-pt1.x,
            yoff=-pt1.y,
        )
        geom.append(poly_0.boundary)
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
            coor.append([[float(x), float(y)]])

        pca = PCA(n_components=2)
        _, l, orientation = cv2.fitEllipse(np.array(coor, dtype=np.int32))
        print("open cv methode", l, orientation)

        aa = max(l)
        bb = min(l)
        skr.append(
            sg.LineString(
                [[0, 0], pca.components_[0] * aa, [0, 0], pca.components_[1] * bb]
            )
        )
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
    df.geometry = geom
    result["slide"] = file
    result["coordinate"] = [cfg["x"], cfg["y"]]
    result["label"] = ""
    result["features"] = df[df.columns[2:]].to_dict
    return result


if __name__ == "__main__":

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    print(config_file)

    result = main(cfg=config_file)

    with open(
        f'/data/DeepLearning/mehdi/features/{config_file["slide"]}_{config_file["x"]}_{config_file["y"]}.json',
        "w",
    ) as fp:
        json.dump(result, fp)
