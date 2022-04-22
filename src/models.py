from typing import Any, Optional, Tuple
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import timm

from nptyping import NDArray


def build_model(model_name, img_size, num_classes):
    model = None

    if model_name == "resnet":
        resnet = timm.create_model("resnet18", pretrained=True)
        infeat = resnet.fc.in_features
        resnet.fc = nn.Linear(infeat, num_classes)
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = resnet

    return model
