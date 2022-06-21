# pylint: disable=invalid-name
"""Create Machine Learning Regressor models."""
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR


def models(cfg):
    """Return a ML Regression model

    Args:
        cfg (dict): configuration file

    Returns:
        sklearn.Model: ML model
    """
    if cfg["MODELS"]["ML"]["TYPE"] == "RandomForest":
        model = RandomForestRegressor(**cfg["MODELS"]["ML"]["RandomForest"])
    elif cfg["MODELS"]["ML"]["TYPE"] == "ExtraTrees":
        model = ExtraTreesRegressor(**cfg["MODELS"]["ML"]["ExtraTrees"])
    elif cfg["MODELS"]["ML"]["TYPE"] == "Knn":
        model = KNeighborsRegressor(**cfg["MODELS"]["ML"]["Knn"])
    elif cfg["MODELS"]["ML"]["TYPE"] == "NuSVR":
        model = NuSVR(**cfg["MODELS"]["ML"]["NuSVR"])
    elif cfg["MODELS"]["ML"]["TYPE"] == "GradientBoosting":
        model = GradientBoostingRegressor(**cfg["MODELS"]["ML"]["GradientBoosting"])

    return model
