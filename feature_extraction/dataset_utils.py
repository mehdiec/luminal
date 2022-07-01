"""This file contains all functions related to the dataset."""
# pylint: disable=import-error
import json
import os
from regex import I
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

CLASS_TO_ID = {
    "luminal A": 0,
    "luminal B": 1,
}
har_ft = [
    "Angular Second Moment",
    "Contrast",
    "Correlation",
    "Sum of Squares: Variance",
    "Inverse Difference Moment",
    "Sum Average",
    "Sum Variance",
    "Sum Entropy",
    "Entropy",
    "Difference Variance",
    "Difference Entropy",
    "Information Measure of Correlation 1",
    "Information Measure of Correlation 2",
]
hu_ft = [f"hu_{i}" for i in range(7)]
# selected_cols ={"color":color_ft,"texture":texture_bis,"glcm":texture_glcm,"geom":geom_ft}
transfs = {"": 0, "bright": 1, "contrast": 2, "stain": 3}


def load_files(path_to_data):
    """Load data input files.

    Args:
        path_to_data (str): path of the data root directory.

    Returns:
        dict(pandas.core.frame.DataFrame): Dictionary of Dataframe containing data from each file.
    """
    data = []  # {"features": [], "target": [], "transformation": [], "slide": []}
    data_files = os.listdir(path_to_data)

    for datafile in tqdm(data_files):
        with open(os.path.join(path_to_data, datafile)) as f:

            result = json.load(f)
        result_dict = {}

        df = pd.DataFrame.from_dict(result["features"])
        tmp = df.to_json(orient="records")
        parsed = json.loads(tmp)
        # lllllllllll = json.dumps(parsed, indent=4)
        # print(parsed[0])

        result_dict["features"] = parsed
        result_dict["target"] = CLASS_TO_ID.get(result["label"])
        result_dict["transformation"] = datafile.split("_")[-1].split(".")[
            0
        ]  # transfs.get(            datafile.split("_")[-1].split(".")[0]    )
        result_dict["slide"] = datafile.split("_")[0]
        result_dict["info"] = result.get("info")

        data.append(result_dict)

    return data


def load_test_data(path_to_test):
    """This function load test data

    Args:
        path_to_test (str): path of the data root directory.

    Returns:
        dict: Dictionary containing every data to create a Dataset.
    """

    # Load the different files
    test_data = load_files(path_to_data=path_to_test)

    # Drop useless
    test_data["input"] = test_data["input"]  # .drop(columns=["_ID"])

    # Create a target
    test_data["target"] = np.ones((len(test_data["input"])))

    feature_and_target = {
        "x_test": test_data["input"].to_numpy(),
        "y_test": np.ones((len(test_data["input"]))).ravel(),
    }

    return feature_and_target


def basic_random_split(path_to_train, valid_ratio=0.2):
    """This function split file according to a ratio to create
    training and validation.

    Args:
        path_to_train (str): path of the data root directory.
        valid_ratio (float): ratio of data for validation dataset.

    Returns:
        dict: Dictionary containing every data to create a Dataset.
    """
    # Load the different files
    training_data = load_files(path_to_data=path_to_train)
    # return training_data

    # Prepare features and targets
    features_and_targets = (
        training_data  # remove_useless_features(training_data=training_data)
    )

    features_and_targets = create_x_and_y(
        input_data=features_and_targets, valid_ratio=valid_ratio
    )

    return features_and_targets


def remove_useless_features(data):
    """Create features and targets

    Args:
        training_data (list): List of Dataframe containing data from each file.

    Returns:
        dict : Dictionary containing features and target for each file.
    """
    # color_ft = [
    #     "h_mean",
    #     "h_std",
    #     "e_mean",
    #     "e_std",
    #     "hue_mean",
    #     "hue_std",
    # ]
    # data = data.drop(columns=color_ft)  # + hu_ft + har_ft)

    return data


def get_index(data, idx):

    return [index for (index, d) in enumerate(data) if d["slide"] == idx]


def create_x_and_y(
    input_data, valid_ratio, transformation=""
):  # pylint: disable=too-many-locals
    """Generate train, valid and test for each file and for each target.

    Args:
        input_data (dict): Features and targets for one file.
        valid_ratio (float): Test and validation ratio.

    Returns:
        dict: train, valid and test inputs and targets.
    """
    feature_and_target = {}
    # all_slide = []
    # for slide in input_data:
    #     if slide["slide"] not in all_slide:
    #         all_slide.append(slide["slide"])

    # if valid_ratio > 0:
    #     valid_idxs = random.sample(
    #         range(1, len(all_slide)), int(valid_ratio * len(all_slide))
    #     )
    #     valid_slide = []
    #     for v_idx in valid_idxs:
    #         slide_name = all_slide[v_idx]
    #         print(slide_name, get_index(input_data, slide_name))
    #         valid_slide += get_index(input_data, slide_name)
    #     valid_slide = np.array(valid_slide)
    #     print(valid_slide)
    #     train = np.array(input_data)[~valid_slide]
    #     valid = np.array(input_data)[valid_slide]
    # else:
    #     train = input_data
    #     valid = input_data

    # for t in train:
    #     t.pop("slide", None)
    # for v in valid:
    #     v.pop("slide", None)

    feature_and_target = {
        "train": input_data,
        "valid": input_data,
    }
    # print(len(valid))

    return feature_and_target
