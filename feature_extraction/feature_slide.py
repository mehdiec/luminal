import argparse
import cv2
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from shapely.geometry import Point
from glob import glob
from sklearn.neighbors import NearestNeighbors
import yaml

from constante import COLORS, GEOMETRIES, LYMPHOCYTE, TEXTURES, TRASH


def listify(hus):
    try:
        hus = hus.replace("[ ", "[")
        hus = hus.replace("  ", ",")
        hus = hus.replace("\n", "")
        hus = hus.replace(" -", ",-")
        hus = hus.replace(" ", ",")
        hus = hus.replace(".", "")

        return json.loads(hus)
    except:

        return [np.nan for _ in range(len(TEXTURES[7:]))]


def centroid_to_int(col):

    return (int(col[0]), int(col[1]))


def in_shape(
    col,
    shape,
):
    point = Point(col)

    lap = shape[0]
    return 1 - lap.contains(point)


def pointPolygonTest(col, contours):

    for contour in contours:

        if (cv2.pointPolygonTest(contour, (col), False)) == 1:
            return 1
    return 0


def ratio(x):
    counts = x.value_counts()
    print(counts)
    return counts[1] / counts[0]


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--path_to_config",
    type=str,
    required=False,
    default="configs/slide_ft_agg_config.yaml",
    help="path to config file",
)
args = parser.parse_args()
if __name__ == "__main__":

    tqdm.pandas()
    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    input_foder = config_file["input_foder"]
    anotation_foder = config_file["anotation_foder"]
    raw_slide_path = config_file["raw_slide_path"]
    out_file = config_file["out_file"]

    list_csv = glob(input_foder + "*")[:]
    df_list = []
    for i in tqdm(range(len(list_csv))):
        df_list.append(pd.read_csv(list_csv[i]))
    df = pd.concat(df_list, ignore_index=True)

    # PREPROCESS
    har_ft = TEXTURES[7:]
    hu_ft = [f"hu_{i}" for i in range(7)]

    df["flcm"] = df["flcm"].progress_apply(listify)
    df["hus"] = df["hus"].progress_apply(listify)
    df[["hu_0", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6"]] = pd.DataFrame(
        df.hus.tolist(), index=df.index
    )
    # df['centroid']   = df['centroid'].apply(listify)
    df[har_ft] = pd.DataFrame(df.flcm.tolist(), index=df.index)

    df["centroid"] = df["centroid"].progress_apply(json.loads)
    df["centroid_point"] = df["centroid"].progress_apply(centroid_to_int)

    df["Zone"] = ""
    inside = False

    # NUC POSITION IN SLIDE
    df_list = []
    df_list_zone = []
    scale = 0.82
    for slide in df.slide.unique()[:]:

        gjson = glob(anotation_foder + slide + "*")[0]
        contours = []
        contours_scaled = []

        print(gjson)
        with open(gjson, "r") as f:
            shape_dict = json.load(f)

        if isinstance(shape_dict, list):
            for list_contour in shape_dict:
                array_contour = np.array(
                    list_contour.get("geometry").get("coordinates")[0], dtype=int
                )
                contours.append(array_contour)
                array_contour_tmp = np.array(
                    list_contour.get("geometry").get("coordinates")[0], dtype=int
                )

                array_contour_tmp[:, 0] = array_contour_tmp[:, 0] * scale
                array_contour_tmp[:, 1] = array_contour_tmp[:, 1] * scale
                contours_scaled.append(array_contour_tmp)
        elif "features" in shape_dict.keys():

            shape_dict = shape_dict.get("features")
            for list_contour in shape_dict:
                if (
                    len(list_contour.get("geometry").get("coordinates")[0]) > 1
                    and len(list_contour.get("geometry").get("coordinates")[0]) < 10
                ):
                    print(len(list_contour.get("geometry").get("coordinates")[0]))

                    continue

                array_contour = np.array(
                    list_contour.get("geometry").get("coordinates")[0], dtype=int
                )
                contours.append(array_contour)
                array_contour_tmp = np.array(
                    list_contour.get("geometry").get("coordinates")[0], dtype=int
                )

                array_contour_tmp[:, 0] = array_contour_tmp[:, 0] * scale
                array_contour_tmp[:, 1] = array_contour_tmp[:, 1] * scale
                contours_scaled.append(array_contour_tmp)

        else:
            # print((shape_dict.get("geometry").get("coordinates"))[1])
            if len(shape_dict.get("geometry").get("coordinates")) < 100:
                for list_contour in shape_dict.get("geometry").get("coordinates"):
                    array_contour = np.array(list_contour, dtype=int)
                    contours.append(array_contour)
                    array_contour_tmp = np.array(list_contour, dtype=int)

                    array_contour_tmp[:, 0] = array_contour_tmp[:, 0] * scale
                    array_contour_tmp[:, 1] = array_contour_tmp[:, 1] * scale
                    contours_scaled.append(array_contour_tmp)

            else:
                array_contour_tmp = np.array(
                    shape_dict.get("geometry").get("coordinates")[0], dtype=int
                )
                array_contour_tmp[:, 0] = array_contour_tmp[:, 0] * scale
                array_contour_tmp[:, 1] = array_contour_tmp[:, 1] * scale
                contours_scaled.append(array_contour_tmp)
                contours.append(
                    np.array(
                        shape_dict.get("geometry").get("coordinates")[0], dtype=int
                    )
                )
        df_tmp = df[df["slide"] == slide]["centroid_point"].progress_apply(
            pointPolygonTest, args=([contours])
        )
        df_tmp_zone = df[df["slide"] == slide]["centroid_point"].progress_apply(
            pointPolygonTest, args=([contours_scaled])
        )
        df_list.append(df_tmp)
        df_list_zone.append(df_tmp_zone)

    ser = pd.concat(df_list)
    df["tumor"] = ser
    ser = pd.concat(df_list_zone)
    df["Zone"] = ser

    # df = pd.read_csv("/data/DeepLearning/mehdi/csv/chkp.csv")
    # FILTERING
    file_name = "models/xgb_lyphocyte.pkl"

    clf_rf = pickle.load(open(file_name, "rb"))

    df_ft = df[LYMPHOCYTE]
    prediction = clf_rf.predict(df_ft)
    prediction_proba = clf_rf.predict_proba(df_ft)
    df["prediction_lymph"] = prediction
    df["prediction_lymph"] = prediction
    df["proba_lymph"] = prediction_proba[:, 1]

    # file_name = "xgb_nucleol.pkl"
    # # pickle.dump(clf_rf, open(file_name, "wb"))
    # clf_rf = pickle.load(open(file_name, "rb"))
    # df_ft = df[selected_ft]
    # prediction = clf_rf.predict(df_ft)
    # prediction_proba = clf_rf.predict_proba(df_ft)
    # df["prediction_nucleol"] = prediction
    # df["proba_nucleol"] = prediction_proba[:, 1]

    file_name = "models/xgb_trash.pkl"
    # pickle.dump(clf_rf, open(file_name, "wb"))
    clf_rf = pickle.load(open(file_name, "rb"))
    df_ft = df[TRASH]
    prediction = clf_rf.predict(df_ft)
    prediction_proba = clf_rf.predict_proba(df_ft)
    df["prediction_trash"] = prediction
    df["proba_trash"] = prediction_proba[:, 1]

    df[["coordinate_x", "coordinate_y"]] = pd.DataFrame(
        df.centroid.tolist(), index=df.index
    )

    for slide in df["slide"].unique():

        mask = df["slide"] == slide
        df_temp = df[mask]
        if (len(df_temp)) > 5:
            neigh = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
            neigh.fit(df_temp[["coordinate_x", "coordinate_y"]])
            distances, indices = neigh.kneighbors(
                df_temp[["coordinate_x", "coordinate_y"]]
            )

            df.loc[mask, "dist_mean"] = distances[:, 1:].mean(axis=1)
            df.loc[mask, "dist_max"] = distances[:, 1:].max(axis=1)
            df.loc[mask, "dist_min"] = distances[:, 1:].min(axis=1)

    df = df[df.prediction_trash == 0]
    df = df[df.tumor == 1]

    columns = GEOMETRIES + COLORS + TEXTURES + ["dist_mean", "slide"]

    mask_lymph = df.prediction_lymph == 1  # | ((df.area < 350) & (df.area > 120))
    mask_tum = df.prediction_lymph == 0  # & ((df.area > 350))

    df.loc[mask_tum, "lymphocyte"] = 0
    df.loc[mask_lymph, "lymphocyte"] = 1
    mask_cell_tum = df.lymphocyte == 0  # | (df.area > 650)

    mean = (
        df.loc[mask_cell_tum, columns].groupby("slide").mean().reset_index()
    ).add_suffix("_mean")

    std = (
        df.loc[mask_cell_tum, columns].groupby("slide").std().reset_index()
    ).add_suffix("_std")

    ##LYMPH VS CELL TUMORAL ALL TUM

    lymph_v_tum = (
        df[["slide", "lymphocyte"]]
        .groupby("slide")
        .agg(ratio)
        .reset_index()
        .rename(columns={"lymphocyte": "lymph_v_tum"})
    )
    # AREA MEAN STD
    mask_cell_tum = df.lymphocyte == 0
    mask_zone_0 = df.Zone == 0
    mask_zone_1 = df.Zone == 1

    # REPARTITION TUMORAL GENERAL
    mask_cell_tum_0 = mask_cell_tum & mask_zone_0
    mask_cell_tum_1 = mask_cell_tum & mask_zone_1

    repartition_tum = (
        (
            df.loc[mask_cell_tum_1, ["slide", "area"]].groupby("slide").count()
            / df.loc[mask_cell_tum_0, ["slide", "area"]].groupby("slide").count()
        )
        .reset_index()
        .rename(columns={"area": "repartition_tum"})
    )

    # REPARTITION TUMORAL PETIT
    mask_cell_tum_0 = (
        mask_cell_tum & mask_zone_0 & ((df.area > 350)) & ((df.area < 600))
    )
    mask_cell_tum_1 = (
        mask_cell_tum & mask_zone_1 & ((df.area > 350)) & ((df.area < 600))
    )

    repartition_tum_petit_0 = (
        (df.loc[mask_cell_tum_0, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_tum_petit_0"})
    )

    repartition_tum_petit_1 = (
        (df.loc[mask_cell_tum_1, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_tum_petit_1"})
    )

    # REPARTITION TUMORAL Moyen
    mask_cell_tum_0 = (
        mask_cell_tum & mask_zone_0 & ((df.area > 600)) & ((df.area < 1000))
    )
    mask_cell_tum_1 = (
        mask_cell_tum & mask_zone_1 & ((df.area > 600)) & ((df.area < 1000))
    )

    repartition_tum_moyen_0 = (
        (df.loc[mask_cell_tum_0, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_tum_moyen_0"})
    )

    repartition_tum_moyen_1 = (
        (df.loc[mask_cell_tum_1, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_tum_moyen_1"})
    )

    # REPARTITION TUMORAL Grand
    mask_cell_tum_0 = mask_cell_tum & mask_zone_0 & ((df.area > 1000))
    mask_cell_tum_1 = mask_cell_tum & mask_zone_1 & ((df.area > 1000))

    repartition_tum_big_1 = (
        (df.loc[mask_cell_tum_1, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_tum_big_1"})
    )

    repartition_tum_big_0 = (
        (df.loc[mask_cell_tum_0, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_tum_big_0"})
    )

    mask_cell_tum_0 = (df.lymphocyte == 1) & mask_zone_0
    mask_cell_tum_1 = (df.lymphocyte == 1) & mask_zone_1

    repartition_lymph_0 = (
        (df.loc[mask_cell_tum_0, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_lymph_0"})
    )
    repartition_lymph_1 = (
        (df.loc[mask_cell_tum_1, ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "repartition_lymph_1"})
    )
    number_lymph = (
        (df.loc[(df.lymphocyte == 1), ["slide", "area"]].groupby("slide").count())
        .reset_index()
        .rename(columns={"area": "number_lymph"})
    )

    number_full = (
        df.loc[mask_cell_tum, ["area", "slide"]]
        .groupby("slide")
        .count()
        .reset_index()
        .rename(columns={"area": "number_full"})
    )

    mask_cell_tum_small = mask_cell_tum & ((df.area > 350)) & ((df.area < 600))
    mask_cell_tum_mid = mask_cell_tum & ((df.area > 600)) & ((df.area < 1000))
    mask_cell_tum_bug = mask_cell_tum & ((df.area > 1000))

    number_small = (
        df.loc[mask_cell_tum_small, ["area", "slide"]]
        .groupby("slide")
        .count()
        .reset_index()
        .rename(columns={"area": "number_small"})
    )

    number_mid = (
        df.loc[mask_cell_tum_mid, ["area", "slide"]]
        .groupby("slide")
        .count()
        .reset_index()
        .rename(columns={"area": "number_mid"})
    )

    number_big = (
        df.loc[mask_cell_tum_bug, ["area", "slide"]]
        .groupby("slide")
        .count()
        .reset_index()
        .rename(columns={"area": "number_big"})
    )

    all_df = [
        number_small,
        number_mid,
        number_big,
        number_full,
        number_lymph,
        lymph_v_tum,
        repartition_tum,
        repartition_tum_petit_0,
        repartition_tum_petit_1,
        repartition_tum_moyen_0,
        repartition_tum_moyen_1,
        repartition_tum_big_0,
        repartition_tum_big_1,
        repartition_lymph_0,
        repartition_lymph_1,
        mean,
        std,
    ]

    dff = pd.concat(all_df, axis=1)

    drop = [
        ("slide", ""),
    ]

    dff = dff.loc[:, ~dff.columns.duplicated()].copy()

    print(dff.head())

    dff.to_csv(out_file)
