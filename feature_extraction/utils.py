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

 