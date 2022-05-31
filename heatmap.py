import argparse
import json
import os.path

from utils import heatmap

DICT_SVS = {
    "0": "21I000004-1-03-1_135435.svs",
    "1": "21I000005-1-16-1_135140.svs",
    "2": "21I000006-1-09-1_134911.svs",
    "3": "21I000007-1-14-1_134640.svs",
    "4": "21I000008-1-02-1_134423.svs",
    "5": "21I000009-1-03-1_145327.svs",
    "6": "21I000245-1-14-1_134026.svs",
    "7": "21I000249-1-08-1_161359.svs",
    "8": "21I000263-1-06-1_133451.svs",
    "9": "21I000268-1-09-1_152552.svs",
}
DICT_T = {
    "0": "luminal_b",
    "1": "luminal_a",
    "2": "luminal_b",
    "3": "luminal_a",
    "4": "luminal_b",
    "5": "luminal_a",
    "6": "luminal_a",
    "7": "luminal_a",
    "8": "luminal_b",
    "9": "luminal_b",
}


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


# Add path to the config file to the command line arguments
# /data/DeepLearning/mehdi/log/luminal/resnet_293/luminal/f3c5a77006b74e168a89605663d65022/checkpoints/epoch=15-val_loss_ce=0.000.ckpt

parser.add_argument(
    "--model_name",
    type=str,
    default="resnet_305",
    help="size of the patches",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=27,
    help=" ",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=5,
    help=" ",
)
parser.add_argument(
    "--size",
    type=int,
    default=2,
    help=" ",
)
parser.add_argument(
    "--resize_ratio",
    type=int,
    default=32,
    help=" ",
)
parser.add_argument(
    "--blend_alpha",
    type=int,
    default=0.3,
    help=" ",
)
args = parser.parse_args()

epoch = args.epoch
model = args.model_name

size = args.size
resize_ratio = args.resize_ratio
blend_alpha = args.blend_alpha

if __name__ == "__main__":
    data_root = f"/data/DeepLearning/mehdi/log/luminal/{model}/"
    result_file = f"/data/DeepLearning/mehdi/log/luminal/{model}/result__{epoch}.json"  # "/home/mehdi/code/luminal/result.json"  # f"/data/DeepLearning/mehdi/log/luminal/{model}/result__{epoch}.json"

    with open(result_file, "r") as fp:
        result = json.load(fp)
    logdir = data_root + "/heatmap"

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # create heatmap for all the slides
    for i, slide_name in DICT_SVS.items():
        heatmap(result_file, size, resize_ratio, blend_alpha, logdir, slide_name)
