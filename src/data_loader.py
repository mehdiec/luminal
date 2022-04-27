from os import PathLike
from typing import Any, Dict, Iterator, List, Sequence, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler
from pathaia.util.types import Slide, Patch
from pathaia.util.basic import ifnone
from albumentations import Compose, BasicTransform
import csv
from nptyping import NDArray
from math import ceil
from pathlib import Path

# from transforms import ToTensor

# slide_file = "/data/DeepLearning/mehdi/csv/luminal_data_split.csv"

MAPPING = {
    "luminal A": 0,
    "luminal B": 1,
}


class ClassificationDataset(Dataset):
    def __init__(
        self,
        slide_file: str,
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
        min_size: int = 10,
        split: str = "train",
        noted: bool = False,
    ):
        super().__init__()
        self.slides = []
        self.masks = []
        self.patches = []
        self.slide_idxs = []
        self.labels = []
        self.split = []
        slide_idx = 0
        self.noted = noted
        with open(slide_file, "r") as out_file:
            reader = csv.DictReader(out_file)

            for row in reader:
                if row["split"] == split:
                    slide_path = row["id"]

                    outfolder = Path("/data/DeepLearning/mehdi/csv/")
                    if self.noted:
                        outfolder = Path("/data/DeepLearning/mehdi/csv_annot/")
                    csv_file = Path(slide_path.split(sep="/")[-1][:-4])
                    patches_path = (
                        outfolder / "patch_csvs" / csv_file.with_suffix(".csv")
                    )
                    self.slides.append(Slide(slide_path, backend=slide_backend))

                    with open(
                        patches_path, "r"
                    ) as patch_file:  # maybe change since i have got the full paths
                        reader = csv.DictReader(patch_file)
                        for patch in reader:
                            self.patches.append(Patch.from_csv_row(patch))

                            self.slide_idxs.append(slide_idx)
                            self.labels.append(MAPPING[row["ab"]])
                    slide_idx += 1
                    # delete
                    # if slide_idx == 1:
                    #     break

        self.transforms = Compose(ifnone(transforms, []))
        self.min_size = min_size
        # self.clean()

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]
        slide_idx = self.slide_idxs[idx]
        slide = self.slides[slide_idx]
        target = self.labels[idx]  # to define

        slide_region = np.asarray(
            slide.read_region(patch.position, patch.level, patch.size).convert("RGB"),
            dtype=np.float32,
        )

        if self.transforms:
            transformed = self.transforms(image=slide_region)

        image_with_slide_idx = {
            "image": transformed["image"],
            "idx": slide_idx,
            "target": target,
        }

        return image_with_slide_idx  # .transpose(2, 0, 1)

    # def clean(self):
    #     print(len(self[0]))
    #     patches = []
    #     slide_idxs = []
    #     idxs = []
    #     for i in range(len(self)):
    #         print(i)
    #         if self[i] is not None:
    #             patches.append(self.patches[i])
    #             slide_idxs.append(self.slide_idxs[i])
    #             idxs.append(i)

    #     self.patches = patches
    #     self.slide_idxs = slide_idxs
