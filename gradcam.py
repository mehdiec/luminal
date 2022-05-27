import argparse
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
import os.path

from src.models import ResNet
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

    target_layers = [model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    # data_ = next(iter(dataloader))
    for data_root in data_roots[:]:
        print(data_root)
        for j, name in enumerate(os.listdir(data_root)):
            print(name)
            logdir = data_root + "/map"
            if not os.path.exists(logdir):
                os.mkdir(logdir)
            full_path = os.path.join(data_root, name)
            image = cv2.imread(full_path)

            # Define a transform to convert PIL
            # image to a Torch tensor
            transform = transforms.Compose([ToTensor()])

            # transform = transforms.PILToTensor()
            # Convert the PIL image to Torch tensor
            if not isinstance(image, type(np.array([]))):
                continue
            img_tensor = ToTensor()(image=image)
            img = img_tensor["image"].unsqueeze(0)

            input_tensor = (
                img  # data_["image"] # Create an input tensor image for your model..
            )

            # We have to specify the target we want to generate
            # the Class Activation Maps for.
            # If targets is None, the highest scoring category
            # will be used for every image in the batch.
            # Here we use ClassifierOutputTarget, but you can define your own custom targets
            # That are, for example, combinations of categories, or specific outputs in a non standard model.

            for x in range(num_classes):
                targets = [ClassifierOutputTarget(x)]

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                rgb_img = to_pil_image(input_tensor.squeeze() / 255)
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                img1 = visualization
                img2 = np.array(to_pil_image(input_tensor.squeeze()))
                dst = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)

                fig, ax = plt.subplots(figsize=(20, 10))

                ax.matshow(dst)
                plt.title(f"map_{j}_label_{x}")
                plt.imsave(f"{logdir}/{name}_map_{x}_.png", dst)
