from albumentations import Normalize, Lambda, Resize
import torch

from torch.utils.data import DataLoader

from src.transforms import ToTensor
from src import data_loader
from tqdm import tqdm

# def compute_mean_std(loader):
#     # Compute the mean over minibatches
#     mean_img = None
#     for imgs, _ in loader:
#         if mean_img is None:
#             mean_img = torch.zeros_like(imgs)
#         mean_img += imgs.sum()
#     mean_img /= len(loader.dataset)

#     # Compute the std over minibatches
#     std_img = torch.zeros_like(mean_img)
#     for imgs, _ in loader:
#         std_img += ((imgs - mean_img) ** 2).sum()
#     std_img /= len(loader.dataset)
#     std_img = torch.sqrt(std_img)

#     # Set the variance of pixels with no variance to 1
#     # Because there is no variance
#     # these pixels will anyway have no impact on the final decision
#     std_img[std_img == 0] = 1

#     return mean_img, std_img


def compute_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    # cnt = 0
    for data in tqdm(loader):

        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
        # if cnt == 20:
        #     break
        # cnt +=1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    print(mean, std)

    return mean, std


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        data = self.base_dataset[index]

        image = self.transform(image=data["image"])

        return image["image"]

    def __len__(self):
        return len(self.base_dataset)


class CenterReduce:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


def load_patches(
    slide_file,
    noted,
    level,
    batch_size,
    num_workers,
    transforms=[],
    normalize=False,
):
    transforms_val = [
        # Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        # Normalize(mean=[0.8459, 0.7529, 0.8145], std=[0.1182, 0.1480, 0.1139]),
        # Resize(384, 384),
        # Normalize(mean=[0.8441, 0.7498, 0.8135], std=[0.1188, 0.1488, 0.1141]),
        ToTensor(),
    ]

    train_ds = data_loader.ClassificationDataset(slide_file, noted=noted, level=level)
    if normalize:

        normalizing_dataset = DatasetTransformer(train_ds, ToTensor())
        normalizing_loader = torch.utils.data.DataLoader(
            dataset=normalizing_dataset, batch_size=batch_size, num_workers=num_workers
        )

        # Compute mean and variance from the training set
        mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)
        print(mean_train_tensor, std_train_tensor)

        normalization_function = CenterReduce(mean_train_tensor, std_train_tensor)

        # Apply the transformation to our dataset
        transforms.append(Lambda(lambda x: normalization_function(x)))
        transforms_val.append(Lambda(lambda x: normalization_function(x)))

        # load the dataset
    train_ds = data_loader.ClassificationDataset(
        slide_file, transforms=transforms, noted=noted, level=level
    )
    val_ds = data_loader.ClassificationDataset(
        slide_file,
        split="valid",
        transforms=transforms_val,
        noted=noted,
        level=level,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_dl, val_dl
