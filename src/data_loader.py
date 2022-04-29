import sys
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
from torchvision.transforms.functional import to_pil_image

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
        outfolder: Path = Path("/data/DeepLearning/mehdi"),
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
        split: str = "train",
        noted: bool = False,
    ):
        """_summary_

        Args:
            slide_file (str): file name with its full path containing the data split
            outfolder (Path, optional): folder containing all the data used in the loader. Defaults to Path("/data/DeepLearning/mehdi").
            transforms (Optional[Sequence[BasicTransform]], optional): image transformation used for data augmentation. Defaults to None.
            slide_backend (str, optional): how we chose to open our slides with. Defaults to "cucim".
            split (str, optional): the split to load. Defaults to "train".
            noted (bool, optional): chose whether to load the slide noted or not. Defaults to False.
        """

        super().__init__()
        # info retrieved from the csv
        self.slides = []
        self.masks = []
        self.patches = []
        self.labels = []
        self.split = []
        # info added to identify each patch to e=its coresponding slide
        self.slide_idxs = []

        slide_idx = 0
        self.noted = noted

        with open(slide_file, "r") as out_file:
            reader = csv.DictReader(out_file)

            if self.noted:
                outfolder = outfolder / "csv_annot"
            else:
                outfolder = outfolder / "csv"
            for (
                row
            ) in (
                reader
            ):  # we read each row of our csv to get the right slide for the right split
                # print(outfolder)
                if row["split"] == split:
                    slide_path = row["id"]
                    # print(outfolder)
                    self.slides.append(Slide(slide_path, backend=slide_backend))

                    # we have access to the csv containing all the patches for a given slide
                    csv_file = Path(slide_path.split(sep="/")[-1][:-4])
                    patches_path = (
                        outfolder / "patch_csvs" / csv_file.with_suffix(".csv")
                    )

                    with open(patches_path, "r") as patch_file:
                        reader = csv.DictReader(patch_file)
                        for patch in reader:
                            self.patches.append(Patch.from_csv_row(patch))
                            self.slide_idxs.append(slide_idx)
                            self.labels.append(MAPPING[row["ab"]])
                    slide_idx += 1

                    # delete
                    if slide_idx == 1:
                        break
        self.transforms = Compose(ifnone(transforms, []))

        # self.clean()

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]
        slide_idx = self.slide_idxs[idx]
        slide = self.slides[slide_idx]
        target = self.labels[idx]

        slide_region = np.asarray(
            slide.read_region(patch.position, patch.level, patch.size).convert("RGB"),
            # dtype=np.float32,
        )
        # image = to_pil_image(slide_region)
        # image.save("oof.png")
        # sys.exit(1)

        if self.transforms:
            transformed = self.transforms(image=slide_region)

        image_with_slide_idx = {
            "image": transformed["image"],
            "idx": slide_idx,
            "target": target,
        }

        return image_with_slide_idx  # .transpose(2, 0, 1)
