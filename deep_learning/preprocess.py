from regex import B
import torch

from albumentations import Normalize, Lambda, Resize, CenterCrop
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm


from deep_learning import data_loader
from deep_learning.transforms import ToTensor, StainAugmentor
from deep_learning.utils import progress_bar
from sklearn.model_selection import KFold


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
    slide_file: str,
    noted: bool,
    level: int,
    batch_size: int,
    num_workers: int,
    patch_size: int = 1024,
    transforms: list = [],
    normalize: bool = False,
    num_classes: int = 3,
    balance: bool = False,
    fold_id: int = 0,
):
    """_summary_

    Args:
        slide_file (str): _description_
        noted (bool): _description_
        level (int): _description_
        batch_size (int): _description_
        num_workers (int): _description_
        patch_size (int, optional): _description_. Defaults to 1024.
        transforms (list, optional): _description_. Defaults to [].
        normalize (bool, optional): _description_. Defaults to False.
        num_classes (int, optional): _description_. Defaults to 3.
        balance (bool, optional): _description_. Defaults to False.
        fold_id (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    if patch_size == 256:
        transforms_val = [
            Resize(256, 256),
            ToTensor(),
        ]
    else:
        transforms_val = [
            ToTensor(),
        ]

    if normalize:

        normalizing_dataset = DatasetTransformer(train_ds, ToTensor())
        normalizing_loader = torch.utils.data.DataLoader(
            dataset=normalizing_dataset, batch_size=batch_size, num_workers=num_workers
        )

        mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)
        print(mean_train_tensor, std_train_tensor)

        normalization_function = CenterReduce(mean_train_tensor, std_train_tensor)

        transforms.append(Lambda(lambda x: normalization_function(x)))
        transforms_val.append(Lambda(lambda x: normalization_function(x)))

        # load the dataset

    val_dict = {
        0: [0, 2, 14, 21, 27, 28, 32, 45, 49, 52, 53, 57, 64],
        1: [1, 3, 13, 15, 18, 23, 25, 34, 43, 47, 54, 55, 63, 58],
        2: [5, 6, 7, 8, 9, 20, 22, 30, 36, 38, 41, 46, 62],
        3: [4, 10, 17, 26, 31, 33, 35, 40, 48, 50, 56, 61, 60],
        4: [11, 12, 16, 19, 24, 29, 37, 39, 42, 44, 51, 65, 59],
    }
    # val_dict = {
    #     0: [2, 5, 7, 11, 21, 32, 37, 41, 45, 47, 51, 52, 54, 61],
    #     1: [0, 3, 6, 18, 26, 28, 30, 31, 49, 56, 58, 59, 65],
    #     2: [1, 4, 8, 16, 19, 29, 33, 39, 42, 46, 60, 62, 63],
    #     3: [9, 10, 12, 13, 15, 20, 23, 24, 25, 27, 50, 57, 64],
    #     4: [14, 17, 22, 34, 35, 36, 38, 40, 43, 44, 48, 53, 55],
    # }
    fold = val_dict[fold_id]
    train_ds = data_loader.ClassificationDataset(
        slide_file,
        transforms=transforms,
        noted=noted,
        level=level,
        patch_size=patch_size,
        num_classes=num_classes,
        fold=fold,
    )
    val_ds = data_loader.ClassificationDataset(
        slide_file,
        split="valid",
        transforms=transforms_val,
        noted=noted,
        level=level,
        patch_size=patch_size,
        num_classes=num_classes,
        fold=fold,
    )
    if balance:
        total = (
            train_ds.labels.count(0)
            + train_ds.labels.count(1)
            + train_ds.labels.count(2)
        )
        num2 = train_ds.labels.count(2)
        if train_ds.labels.count(2) == 0:
            num2 = 1

        class_weights = [
            (total / train_ds.labels.count(0)),
            total / train_ds.labels.count(1),
            total / num2,
        ]
        print(class_weights)
        # class_weights = [100, 100, 1]

        sample_weights = [0] * len(train_ds)

        for idx, data in enumerate(train_ds):
            class_weight = class_weights[data["target"]]
            sample_weights[idx] = class_weight
            progress_bar(
                idx,
                len(train_ds),
                msg=f"luminal A: {train_ds.labels.count(0)},luminal B: {train_ds.labels.count(1)},trash: {train_ds.labels.count(2)}",
            )

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        )
    else:
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
