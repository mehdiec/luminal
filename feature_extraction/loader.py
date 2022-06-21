
import argparse
import os
import torch
import yaml

from torch.utils.data import DataLoader
from preprocessing import apply_preprocessing
from dataset_utils import basic_random_split, RegressionDataset, load_test_data


def preprocess(cfg):  # pylint: disable=too-many-locals
    """Main function to call to load and process data

    Args:
        cfg (dict): configuration file

    Returns:
        tuple[DataLoader, DataLoader]: train and validation DataLoader
        DataLoader: test DataLoader
    """

    # Set path
    path_to_train = os.path.join(cfg["DATA_DIR"], "train/")
    # path_to_test = os.path.join(cfg["DATA_DIR"], "test/")

    # Load the dataset for the training/validation sets
    data = basic_random_split(
        path_to_train=path_to_train, valid_ratio=cfg["DATASET"]["VALID_RATIO"]
    )
    preprocessed_data = apply_preprocessing(
        cfg=cfg["DATASET"]["PREPROCESSING"], data=data
    )

    # Load the test set
    # test_data = load_test_data(path_to_test=path_to_test)
    # preprocessed_test_data = apply_preprocessing(
    #     cfg=cfg["DATASET"]["PREPROCESSING"], data=test_data, test=True
    # )
 
    return preprocessed_data, 0

    