import argparse
import cv2
import os.path
import torch

from src.models import ResNet
from utils import gradcam
 
 
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# Add path to the config file to the command line arguments
parser.add_argument(
    "--path_model",
    type=str,
    default="/data/DeepLearning/mehdi/log/luminal/resnet_305/luminal/7756124cf9c94e04b37eec618585d6c1/checkpoints/epoch=27-val_loss_ce=0.000.ckpt",
    help="full path of the model to load",
)

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
    default=3,
    help=" ",
)
args = parser.parse_args()


if __name__ == "__main__":
    name = args.path_model

    data_roots = [
        f"/data/DeepLearning/mehdi/log/luminal/{args.model_name}/{args.epoch}/{i}"
        for i in range(10)
    ]
    num_classes = args.num_classes

    # loading the model chosen
    lesnet = torch.load(name)
    resnet = {k.replace("model.", ""): v for k, v in lesnet["state_dict"].items()}
    resnet_ = {
        k.replace("model.", "resnet."): v for k, v in lesnet["state_dict"].items()
    }
    sttdict = {**resnet, **resnet_}
    sttdict = {k.replace("resnet.fc.1", "resnet.fc"): v for k, v in sttdict.items()}
    model = ResNet(num_classes)

    # Load the models for inference
    print(sttdict.keys())
    model.load_state_dict(sttdict, strict=True)
    _ = model.eval()
 
    for data_root in data_roots[:]:
        print(data_root)
        for j, name in enumerate(os.listdir(data_root)):
            print(name)
      
            full_path = os.path.join(data_root, name)
            image = cv2.imread(full_path)
            gradcam(image, model, data_root, num_classes=3)

            # Define a transform to convert PIL
          