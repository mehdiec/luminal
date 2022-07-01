import argparse
import os
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader
from preprocessing import apply_preprocessing
from dataset_utils import basic_random_split, load_test_data



def preprocess(cfg):  # pylint: disable=too-many-locals
    """Main function to call to load and process data

    Args:
        cfg (dict): configuration file

    Returns:
        tuple[DataLoader, DataLoader]: train and validation DataLoader
        DataLoader: test DataLoader
    """

    # Set path
    path_to_train = cfg["DATA_DIR"]
    # path_to_test = os.path.join(cfg["DATA_DIR"], "test/")

    # Load the dataset for the training/validation sets
    data = basic_random_split(
        path_to_train=path_to_train, valid_ratio=cfg["DATASET"]["VALID_RATIO"]
    )
    # selected_col = ['pt_ax', 'compactness', 'Angular Second Moment', 'Contrast',
    #    'Correlation', 'Inverse Difference Moment', 'Sum Average',
    #    'Sum Variance', 'Entropy', 'Difference Variance', 'Difference Entropy',
    #    'Information Measure of Correlation 1',
    #    'Information Measure of Correlation 2', 'hu_0']
    # data["x_train"] = data["x_train"][selected_col]
    # data["x_valid"] = data["x_valid"][selected_col]

    # data["columns"] = data["train"][0]["features"].keys()

    if not cfg["DATASET"]["PREPROCESSING"]["NORMALIZE"]["ACTIVE"]:
        return data

    preprocessed_data = apply_preprocessing(
        cfg=cfg["DATASET"]["PREPROCESSING"], data=data
    )

    # Load the test set
    # test_data = load_test_data(path_to_test=path_to_test)
    # preprocessed_test_data = apply_preprocessing(
    #     cfg=cfg["DATASET"]["PREPROCESSING"], data=test_data, test=True
    # )

    return preprocessed_data


if __name__ == "__main__":
    cfg = {
        "DATA_DIR": "/data/DeepLearning/mehdi/features",
        "DATASET": {
            "VALID_RATIO": 0.1,
            "PREPROCESSING": {
                "NORMALIZE": {"ACTIVE": False, "TYPE": "StandardScaler"},
                "PCA": {"ACTIVE": False, "N_COMPONENTS": 0.95},
            },
        },
    }
    a = preprocess(cfg)
