"""This module aims to launch a training procedure."""
# pylint: disable=import-error, no-name-in-module
import os
import argparse
from shutil import copyfile
import json
import pickle
import yaml
import numpy as np

from sklearn.metrics import mean_squared_error
from utils import launch_grid_search, load_model,generate_unique_logpath

from loader import preprocess

 



def main(cfg, path_to_config):  # pylint: disable=too-many-locals
    """Main pipeline to train a ML model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """
    # Load data
    preprocessed_data, _ = preprocess(cfg=cfg)  # TODO

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["MODELS"]["ML"]["TYPE"].lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    if cfg["MODELS"]["ML"]["GRID_SEARCH"]:
        model, params = launch_grid_search(cfg, preprocessed_data)

        with open(os.path.join(save_dir, "best_params.json"), "w") as outfile:
            json.dump(params, outfile, indent=2)

    else:
        model = load_model(cfg=cfg)  # TODO

    model.fit(X=preprocessed_data["x_train"], y=preprocessed_data["y_train"])
    pickle.dump(model, open(os.path.join(save_dir, "model.pck"), "wb"))

    y_pred = model.predict(preprocessed_data["x_valid"])

    print("Valid MSE : ", mean_squared_error(preprocessed_data["y_valid"], y_pred))
    print(
        "Train MSE : ",
        mean_squared_error(
            preprocessed_data["y_train"], model.predict(preprocessed_data["x_train"])
        ),
    )

  

if __name__ == "__main__":
    # Init the parser
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

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

 
    main(cfg=config_file, path_to_config=args.path_to_config)

 