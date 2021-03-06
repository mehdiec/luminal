import csv
import numpy as np
import torch

from albumentations import Compose, BasicTransform
from math import ceil
from nptyping import NDArray
from pathaia.patches import filter_thumbnail
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.util.types import Slide, Patch
from pathaia.util.basic import ifnone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Optional, Tuple, Union
from torch.utils.data import Dataset, RandomSampler
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
        level: int = 0,
        patch_size: int = 1024,
        num_classes: int = 2,
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
        self.labels_slide = []

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
                cnt_2 = 0
                cnt = 0
                # print(outfolder)
                if row["split"] == split:
                    slide_path = row["id"]
                    # print(outfolder)
                    self.slides.append(Slide(slide_path, backend=slide_backend))

                    # we have access to the csv containing all the patches for a given slide
                    csv_file = Path(slide_path.split(sep="/")[-1][:-4])
                    patches_path = (
                        outfolder
                        / "patch_csvs"
                        / str(level)
                        / str(patch_size)
                        / csv_file.with_suffix(".csv")
                    )

                    # patches_path = (
                    #     outfolder / "patch_csvs" / csv_file.with_suffix(".csv")
                    # )

                    with open(patches_path, "r") as patch_file:
                        reader = csv.DictReader(patch_file)
                        for i, patch in enumerate(reader):
                            if num_classes == 3:

                                self.patches.append(Patch.from_csv_row(patch))
                                self.slide_idxs.append(slide_idx)
                                self.labels.append(int(patch.get("label")))
                                self.labels_slide.append(MAPPING[row["ab"]])
                            else:
                                if int(patch.get("label")) != 2:
                                    self.patches.append(Patch.from_csv_row(patch))
                                    self.slide_idxs.append(slide_idx)
                                    self.labels.append(int(patch.get("label")))
                                    self.labels_slide.append(MAPPING[row["ab"]])
                    slide_idx += 1

                    # # delete
                    # if slide_idx == 3:
                    #     break

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
        target_slide = self.labels_slide[idx]
        target = self.labels[idx]

        slide_region = (
            np.asarray(
                slide.read_region(patch.position, patch.level, patch.size).convert(
                    "RGB"
                ),
                dtype=np.float32,
            )
            / 255.0
        )

        # image = to_pil_image(slide_region)

        if self.transforms:
            transformed = self.transforms(image=slide_region)

        image_with_slide_idx = {
            "image": transformed["image"],
            # .transpose(2, 0)
            # .transpose(0, 1),  # .transpose(2, 0, 1),
            "idx": slide_idx,
            "target": target,
            "pos_x": patch.position.x,
            "pos_y": patch.position.y,
            "target_slide": target_slide,
        }

        return image_with_slide_idx  #


class SingleSlideClassificationDataset(Dataset):
    def __init__(
        self,
        slide_file: str,
        outfolder: Path = Path("/data/DeepLearning/mehdi"),
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
        split: str = "train",
        noted: bool = False,
        level: int = 0,
        slide_nb: int = 0,
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
        self.labels_slide = []

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
                        outfolder
                        / "patch_csvs"
                        / str(level)
                        / csv_file.with_suffix(".csv")
                    )
                    # patches_path = (
                    #     outfolder / "patch_csvs" / csv_file.with_suffix(".csv")
                    # )

                    with open(patches_path, "r") as patch_file:
                        reader = csv.DictReader(patch_file)
                        for patch in reader:
                            if slide_idx != slide_nb:
                                break
                            self.patches.append(Patch.from_csv_row(patch))
                            self.slide_idxs.append(slide_idx)
                            self.labels.append(MAPPING[row["ab"]])
                            self.labels_slide.append(MAPPING[row["ab"]])
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
        target_slide = self.labels_slide[idx]

        slide_region = (
            np.asarray(
                slide.read_region(patch.position, patch.level, patch.size).convert(
                    "RGB"
                ),
                dtype=np.float32,
            )
            / 255.0
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
            "pos_x": patch.position.x,
            "pos_y": patch.position.y,
            "target_slide": target_slide,
        }

        return image_with_slide_idx  # .transpose(2, 0, 1)


class FullSlideClassificationDataset(Dataset):
    def __init__(
        self,
        slide_file: str,
        outfolder: Path = Path("/data/DeepLearning/mehdi"),
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
        split: str = "train",
        noted: bool = False,
        level: int = 0,
        patch_size: int = 1024,
        num_classes: int = 2,
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
        self.labels_slide = []

        slide_idx = 0
        self.noted = noted
        self.transforms = Compose(ifnone(transforms, []))

        with open(slide_file, "r") as out_file:
            reader = csv.DictReader(out_file)

            for (i, row) in enumerate(
                reader
            ):  # we read each row of our csv to get the right slide for the right split
                # print(outfolder)
                if row["split"] == split:
                    slide_path = row["id"]
                    self.slides.append(Slide(slide_path, backend=slide_backend))
                    self.patches.append(1)
                    self.labels.append(MAPPING[row["ab"]])
                    self.split.append(1)
                    # info added to identify each patch to e=its coresponding slide
                    self.slide_idxs.append(i)
                    self.labels_slide.append(1)

        # self.clean()

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]
        slide_idx = self.slide_idxs[idx]
        slide = self.slides[idx]
        target = self.labels[idx]
        target_slide = self.labels_slide[idx]

        slide_region = slide.get_thumbnail((1024, 1024))
        slide_region = np.array(slide_region.resize((1024, 1024)))

        # image = to_pil_image(slide_region)
        # image.save("oof.png")
        # sys.exit(1)

        if self.transforms:
            transformed = self.transforms(image=slide_region)

        image_with_slide_idx = {
            "image": transformed["image"],
            "idx": slide_idx,
            "target": target,
            "pos_x": 0,
            "pos_y": 0,
            "target_slide": target_slide,
        }

        return image_with_slide_idx  # .transpose(2, 0, 1)


class SingleSlideInference(Dataset):
    def __init__(
        self,
        slide_file: str,
        slide_backend: str = "cucim",
        level: int = 0,
        patch_size: int = 1024,
        transforms: Optional[Sequence[BasicTransform]] = None,
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
        self.patches = []
        self.transforms = Compose(ifnone(transforms, []))
        slide_idx = 0
        slide_path = "/media/AprioricsSlides/" + slide_file
        self.slide = Slide(slide_path, backend=slide_backend)
        for patch in slide_rois_no_image(
            self.slide,
            level,
            psize=patch_size,
            interval=0,
            slide_filters=[filter_thumbnail],
            thumb_size=2000,
        ):
            self.patches.append(patch)

        # self.clean()

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]

        slide_region = (
            np.asarray(
                self.slide.read_region(patch.position, patch.level, patch.size).convert(
                    "RGB"
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        if self.transforms:
            transformed = self.transforms(image=slide_region)
        # image = to_pil_image(slide_region)

        image_with_slide_idx = {
            "image": transformed["image"],
            # .transpose(2, 0)
            # .transpose(0, 1),  # .transpose(2, 0, 1),
            "pos_x": patch.position.x,
            "pos_y": patch.position.y,
        }

        return image_with_slide_idx  #
