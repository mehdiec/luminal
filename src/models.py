from torch import nn
import torch
import timm

# from transformers import ViTForImageClassification

id2label = {
    0: "luminal A",
    1: "luminal B",
}

label2id = {
    "luminal A": 0,
    "luminal B": 1,
}


def dropout_linear_relu(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop), nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]


def conv_relu_maxp(in_channels, out_channels, ks):
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=True,
        ),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    ]


class VanillaCNN(nn.Module):
    def __init__(self, num_classes):
        super(VanillaCNN, self).__init__()

        # By default, Linear layers and Conv layers use Kaiming He initialization

        self.features = nn.Sequential(
            *conv_relu_maxp(3, 16, 5),
            *conv_relu_maxp(16, 32, 5),
            *conv_relu_maxp(32, 64, 5),
        )
        probe_tensor = torch.zeros((1, 3, 1024, 1024))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
            *dropout_linear_relu(out_features.shape[0], 128, 0.5),
            *dropout_linear_relu(128, 256, 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  #  OR  x = x.view(-1, self.num_features)
        y = self.classifier(x)
        return y


def build_model(model_name, num_classes, freeze=False, pretrained=True):
    model = None

    if model_name == "vanilla":
        model = VanillaCNN(num_classes)

    elif model_name == "resnet":
        resnet = timm.create_model("resnet18", pretrained=pretrained)
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        infeat = resnet.fc.in_features
        resnet.fc = nn.Linear(infeat, num_classes)
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = resnet

    # elif model_name == "vit":
    #     model = ViTForImageClassification.from_pretrained(
    #         "google/vit-base-patch16-224-in21k",
    #         num_labels=2,
    #         id2label=id2label,
    #         label2id=label2id,
    #     )

    return model
