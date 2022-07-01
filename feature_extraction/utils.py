"""This module aims to define utils function for the project."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.svm import NuSVR

from models import models


def load_model(cfg):
    """This function aims to load the right model regarding the configuration file

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """
    return models(cfg=cfg)


def launch_grid_search(cfg, preprocessed_data):  # pylint: disable=too-many-locals
    """Launch a grid search on different models

    Args:
        cfg (dict): Configuration file
        preprocessed_data (dict): data
    """
    # Train
    x_train = preprocessed_data["x_train"]
    y_train = preprocessed_data["y_train"]
    # Valid
    x_valid = preprocessed_data["x_valid"]
    y_valid = preprocessed_data["y_valid"]

    if cfg["MODELS"]["ML"]["TYPE"] == "RandomForest":
        rfr = RandomForestRegressor(bootstrap=False, n_jobs=-1)

        param_grid = {
            "min_samples_split": np.arange(4, 9, 2),
            "max_depth": np.arange(18, 28, 2),
            "max_features": np.arange(30, min(x_train.shape[1], 100), 10),
            "n_estimators": np.arange(70, 120, 10),
        }

        rfr_cv = GridSearchCV(rfr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        rfr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in rfr_cv.best_params_.items():
            params[key] = int(value)

        return rfr_cv.best_estimator_, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "ExtraTrees":
        etr = ExtraTreesRegressor(bootstrap=False, n_jobs=-1)

        param_grid = {
            "min_samples_split": np.arange(4, 9, 2),
            "max_depth": np.arange(18, 28, 2),
            "max_features": np.arange(30, min(x_train.shape[1], 100), 10),
            "n_estimators": np.arange(70, 120, 10),
        }

        etr_cv = GridSearchCV(etr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        etr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in etr_cv.best_params_.items():
            params[key] = int(value)

        return etr_cv.best_estimator_, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "GradientBoosting":
        gbr = GradientBoostingRegressor()

        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 1, 0.5],
            "min_samples_leaf": [4, 5, 6],
            "subsample": [0.6, 0.7, 0.8],
            "min_samples_split": np.arange(4, 8, 2),
            "max_depth": np.arange(18, 28, 2),
            "max_features": np.arange(30, min(x_train.shape[1], 100), 10),
            "n_estimators": np.arange(70, 120, 10),
        }

        gbr_cv = GridSearchCV(gbr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        gbr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in gbr_cv.best_params_.items():
            params[key] = int(value)

        return gbr_cv.best_estimator_, params

    elif cfg["MODELS"]["ML"]["TYPE"] == "NuSVR":
        nusvr = NuSVR()

        param_grid = {
            "C": [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
            "gamma": [0.008, 0.009, 0.01, 0.02, 0.03, "auto"],
            "kernel": ["poly", "rbf"],
            "nu": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }

        nusvr_cv = GridSearchCV(
            nusvr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2
        )
        nusvr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in nusvr_cv.best_params_.items():
            params[key] = int(value)

        return nusvr_cv.best_estimator_, params

    model = RandomForestRegressor(
        bootstrap=False,
        max_depth=22,
        max_features=50,
        min_samples_split=4,
        n_estimators=80,
        n_jobs=-1,
    )
    params = {
        "max_depth": 22,
        "max_features": 50,
        "min_samples_split": 4,
        "n_estimators": 80,
        "bootstrap": False,
        "n_jobs": 1,
    }
    return model, params


def retrieve_id(cfg):
    """Retrieve the ID column of the test file

    Args:
        cfg (dict): configuration file

    Returns:
        numpy.array: IDs of the samples
    """
    data_files = os.listdir(os.path.join(cfg["DATA_DIR"], "test/"))

    for datafile in data_files:
        if "input" in datafile:
            test_data = pd.read_csv(
                os.path.join("../data/test", datafile), delimiter=",", decimal="."
            )

    return test_data["_ID"].to_numpy()


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist

    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file

    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def coarseness(image, kmax=5):
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    if w == 0 or h == 0:
        return np.nan
    kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))
    average_gray = np.zeros([kmax, w, h])
    horizon = np.zeros([kmax, w, h])
    vertical = np.zeros([kmax, w, h])
    Sbest = np.zeros([w, h])

    for k in range(kmax):
        window = np.power(2, k)
        for wi in range(w)[window : (w - window)]:
            for hi in range(h)[window : (h - window)]:
                average_gray[k][wi][hi] = np.sum(
                    image[wi - window : wi + window, hi - window : hi + window]
                )
        for wi in range(w)[window : (w - window - 1)]:
            for hi in range(h)[window : (h - window - 1)]:
                horizon[k][wi][hi] = (
                    average_gray[k][wi + window][hi] - average_gray[k][wi - window][hi]
                )
                vertical[k][wi][hi] = (
                    average_gray[k][wi][hi + window] - average_gray[k][wi][hi - window]
                )
        horizon[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))
        vertical[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))

    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:, wi, hi])
            h_max_index = np.argmax(horizon[:, wi, hi])
            v_max = np.max(vertical[:, wi, hi])
            v_max_index = np.argmax(vertical[:, wi, hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2, index)

    fcrs = np.mean(Sbest)
    return fcrs


def contrast(image):
    image = np.array(image)
    image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    m4 = np.mean(np.power(image - np.mean(image), 4))
    v = np.var(image)
    std = np.power(v, 0.5)
    alfa4 = m4 / np.power(v, 2)
    fcon = std / np.power(alfa4, 0.25)
    return fcon


def directionality(image):
    image = np.array(image, dtype="int64")
    h = image.shape[0]
    w = image.shape[1]
    if w == 0 or h == 0:
        return np.nan
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    deltaH = np.zeros([h, w])
    deltaV = np.zeros([h, w])
    theta = np.zeros([h, w])

    # calc for deltaH
    for hi in range(h)[1 : h - 1]:
        for wi in range(w)[1 : w - 1]:
            deltaH[hi][wi] = np.sum(
                np.multiply(image[hi - 1 : hi + 2, wi - 1 : wi + 2], convH)
            )
    for wi in range(w)[1 : w - 1]:
        deltaH[0][wi] = image[0][wi + 1] - image[0][wi]
        deltaH[h - 1][wi] = image[h - 1][wi + 1] - image[h - 1][wi]
    for hi in range(h):
        deltaH[hi][0] = image[hi][1] - image[hi][0]
        deltaH[hi][w - 1] = image[hi][w - 1] - image[hi][w - 2]

    # calc for deltaV
    for hi in range(h)[1 : h - 1]:
        for wi in range(w)[1 : w - 1]:
            deltaV[hi][wi] = np.sum(
                np.multiply(image[hi - 1 : hi + 2, wi - 1 : wi + 2], convV)
            )
    for wi in range(w):
        deltaV[0][wi] = image[1][wi] - image[0][wi]
        deltaV[h - 1][wi] = image[h - 1][wi] - image[h - 2][wi]
    for hi in range(h)[1 : h - 1]:
        deltaV[hi][0] = image[hi + 1][0] - image[hi][0]
        deltaV[hi][w - 1] = image[hi + 1][w - 1] - image[hi][w - 1]

    deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
    deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

    # calc the theta
    for hi in range(h):
        for wi in range(w):
            if deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0:
                theta[hi][wi] = 0
            elif deltaH[hi][wi] == 0:
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
    theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

    n = 16
    t = 12
    cnt = 0
    hd = np.zeros(n)
    dlen = deltaG_vec.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if (
                (deltaG_vec[k] >= t)
                and (theta_vec[k] >= (2 * ni - 1) * np.pi / (2 * n))
                and (theta_vec[k] < (2 * ni + 1) * np.pi / (2 * n))
            ):
                hd[ni] += 1
    hd = hd / np.mean(hd)
    hd_max_index = np.argmax(hd)
    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]
    return fdir


def linelikeness(image, sita, dist):
    pass


def regularity(image, filter):
    pass


def roughness(fcrs, fcon):
    return fcrs + fcon
def iou(shape,cvx_shape):
    return ((shape.intersection(cvx_shape)).area/((shape.union(cvx_shape).area)))