from albumentations import Lambda
import torch

from torch.utils.data import DataLoader

from src.transforms import ToTensor
from src import data_loader


def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img) ** 2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1

    return mean_img, std_img


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

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
        ToTensor(),
    ]

    train_ds = data_loader.ClassificationDataset(slide_file, noted=noted, level=level)
    if normalize:

        normalizing_dataset = DatasetTransformer(train_ds, transforms.ToTensor())
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
