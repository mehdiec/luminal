"""This module aims to load models for inference and try it on test data."""
# pylint: disable=import-error, no-name-in-module, unused-import
import argparse
import csv
import pickle
import torch
import tqdm
import yaml

import numpy as np
import pandas as pd

import data.loader as loader
from tools.utils import load_model, retrieve_id


def inference_ml(cfg):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
    """

    # Load test data
    _, preprocessed_test_data = loader.main(cfg=cfg)

    # Test
    x_test = preprocessed_test_data["x_test"]

    # Get sample IDs
    idx = retrieve_id(cfg=cfg)

    # Load model
    model = pickle.load(open(cfg["TEST"]["PATH_TO_MODEL"], "rb"))

    # Make predictions
    y_pred_test = model.predict(x_test)
    output = np.concatenate((idx.reshape(-1, 1), y_pred_test.reshape(-1, 1)), axis=1)

    output_df = pd.DataFrame(output, columns=["_ID", "Y"])
    output_df = output_df.astype({"_ID": "int32"})
    output_df.to_csv("output.csv", index=False)


def inference_nn(cfg):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
    """

    # Load test data
    _, _, test_dataloader = loader.main(cfg=cfg)

    # Get sample IDs
    idx = retrieve_id(cfg=cfg)

    # Define device for computational efficiency
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Load model for inference
    input_size = test_dataloader.dataset[0][0].shape[0]

    model = load_model(cfg, input_size)
    model = model.to(device)

    model.load_state_dict(torch.load(cfg["TEST"]["PATH_TO_MODEL"]))
    model.eval()

    y_pred_test = None

    for inputs, _ in tqdm.tqdm(test_dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Concat the result in order to compute f1-score
        if y_pred_test is None:
            y_pred_test = outputs
        else:
            y_pred_test = torch.cat((y_pred_test, outputs))

    y_pred_test = y_pred_test.cpu().int().numpy()
    output = np.concatenate((idx.reshape(-1, 1), y_pred_test.reshape(-1, 1)), axis=1)

    output_df = pd.DataFrame(output, columns=["_ID", "Y"])
    output_df = output_df.astype({"_ID": "int32"})
    output_df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    # Init the parser;
    inference_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add path to the config file to the command line arguments;
    inference_parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file.",
    )
    args = inference_parser.parse_args()

    # Load config file
    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    # Run inference
    if config_file["MODELS"]["NN"]:
        inference_nn(cfg=config_file)

    else:
        inference_ml(cfg=config_file)
