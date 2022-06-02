import torch
import timm

from torch import nn

# from coatnet import CoAtNet

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
    def __init__(self, num_classes, shape):
        super(VanillaCNN, self).__init__()

        # By default, Linear layers and Conv layers use Kaiming He initialization

        self.features = nn.Sequential(
            *conv_relu_maxp(3, 16, 5),
            *conv_relu_maxp(16, 32, 5),
            *conv_relu_maxp(32, 64, 5),
        )
        probe_tensor = torch.zeros((1, 3, shape, shape))
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


class ViTBase16(nn.Module):
    def __init__(self, num_classes, pretrained=False, load=None):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_384", pretrained=True)
        if load:
            self.model.load_state_dict(torch.load(load))

        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet, self).__init__()

        # define the resnet152
        self.resnet = timm.create_model("resnet18", pretrained=pretrained)
        infeat = 512  # self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(infeat, num_classes)  # make the change

        # isolate the feature blocks

        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # average pooling layer
        self.global_pool = self.resnet.global_pool

        # classifier
        self.fc = nn.Linear(infeat, num_classes)
        # nn.Sequential(
        #     nn.Dropout2d(0.5), nn.Linear(infeat, num_classes)
        # )  # nn.Linear(infeat, num_classes)#=nn.Linear(infeat, num_classes)# nn.Sequential(nn.Dropout2d(0.5), nn.Linear(infeat, num_classes))# nn.Linear(infeat, num_classes)

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.forward_conv(x)

    def forward_conv(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):

        # extract the features
        x = self.forward_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.global_pool(x)

        x = self.fc(x)

        return x


def build_model(
    model_name, num_classes, freeze=False, pretrained=True, dropout=0, shape=512
):
    model = None

    if model_name == "vanilla":
        model = VanillaCNN(num_classes, shape)

    elif model_name == "resnet":
        resnet = timm.create_model("resnet18", pretrained=pretrained)
        infeat = resnet.fc.in_features
        if dropout > 0:
            resnet.fc = nn.Sequential(
                nn.Dropout2d(dropout), nn.Linear(infeat, num_classes)
            )
        else:
            resnet.fc = nn.Linear(infeat, num_classes)  # make the change

        # isolate the feature blocks

        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = resnet
    elif model_name == "resnet50_":
        resnet = timm.create_model("resnet50", pretrained=pretrained)
        infeat = resnet.fc.in_features
        if dropout > 0:
            resnet.fc = nn.Sequential(
                nn.Dropout2d(dropout), nn.Linear(infeat, num_classes)
            )
        else:
            resnet.fc = nn.Linear(infeat, num_classes)  # make the change

        # isolate the feature blocks

        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = resnet
    elif model_name == "vit":
        model = ViTBase16(num_classes=1)
    elif model_name == "efficientnet":
        model = timm.create_model("efficientnet_b5", pretrained=True)
        infeat = model.classifier.in_features
        if dropout > 0:
            model.classifier = nn.Sequential(
                nn.Dropout2d(dropout), nn.Linear(infeat, num_classes)
            )
        else:
            model.classifier = nn.Linear(infeat, num_classes)  # make the change

        # isolate the feature blocks

        if freeze:
            for param in model.parameters():
                param.requires_grad = False

    elif model_name == "mobilenet":
        resnet = timm.create_model("mobilenetv2_140", pretrained=pretrained)
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        infeat = resnet.classifier.in_features
        resnet.classifier = nn.Linear(infeat, num_classes)
        # resnet = ResNet(1, pretrained)
        model = resnet
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # model = ViT(
        #     image_size=1024,
        #     patch_size=64,
        #     num_classes=1,
        #     dim=1,
        #     depth=12,
        #     heads=12,
        #     mlp_dim=3072,
        #     dropout=0.0,
        #     emb_dropout=0.1,
        # )
    # elif model_name == "coat":

    #     model = CoAtNet(in_ch=3, image_size=1024)

    # elif model_name == "vit":
    #     model = ViTForImageClassification.from_pretrained(
    #         "google/vit-base-patch16-224-in21k",
    #         num_labels=2,
    #         id2label=id2label,
    #         label2id=label2id,
    #     )

    return model
