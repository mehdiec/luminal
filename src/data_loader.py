from os import PathLike
from typing import Any, Dict, Iterator, List, Sequence, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler
from pathaia.util.types import Slide, Patch
from pathaia.util.basic import ifnone
from albumentations import Compose, BasicTransform
from apriorics.masks import mask_to_bbox
from apriorics.transforms import StainAugmentor
import csv
from nptyping import NDArray
from math import ceil


def split_data_k_fold(
    data: Sequence[Any],
    k: int = 5,
    test_size: Union[int, float] = 0.1,
    seed: Optional[int] = None,
) -> Dict[str, NDArray[Any]]:
    r"""
    Split input data sequence into k folds and a test fold.

    Args:
        data: input data sequence.
        k: number of folds.
        test_size: if int, number of items to keep for test; if float, portion of data
            to keep for test.
        seed: seed to use for RNG.

    Returns:
        A dictionary that maps fold numbers (or "test") to the corresponding data
        sequence.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    data = np.array(data)
    if isinstance(test_size, float):
        test_size = ceil(test_size * n)
    test_idxs = rng.randint(n, size=test_size)
    test_data = data[test_idxs]

    data = np.delete(data, test_idxs)
    folds = rng.randint(k, size=len(data))

    splits = {str(i): data[folds == i] for i in range(k)}
    splits["test"] = test_data

    return splits


class ClassificationDataset(Dataset):
    r"""
    PyTorch dataset for slide segmentation tasks.

    Args:
        slide_paths: list of slides' filepaths.
        mask_paths: list of masks' filepaths. Masks are supposed to be tiled pyramidal
            images.
        patches_paths: list of patch csvs' filepaths. Files must be formatted according
            to `PathAIA API <https://github.com/MicroMedIAn/PathAIA>`_.
        stain_matrices_paths: path to stain matrices .npy files. Each file must contain
            a (2, 3) matrice to use for stain separation. If not sppecified while
            `stain_augmentor` is, stain matrices will be computed at runtime (can cause
            a bottleneckd uring training).
        stain_augmentor: :class:`~apriorics.transforms.StainAugmentor` object to use for
            stain augmentation.
        transforms: list of `albumentation <https://albumentations.ai/>`_ transforms to
            use on images (and on masks when relevant).
        slide_backend: whether to use `OpenSlide <https://openslide.org/>`_ or
            `cuCIM <https://github.com/rapidsai/cucim>`_ to load slides.
    """

    def __init__(
        self,
        slide_paths: Sequence[PathLike],
        mask_paths: Sequence[PathLike],
        patches_paths: Sequence[PathLike],
        stain_matrices_paths: Optional[Sequence[PathLike]] = None,
        stain_augmentor: Optional[StainAugmentor] = None,
        transforms: Optional[Sequence[BasicTransform]] = None,
        slide_backend: str = "cucim",
        min_size: int = 10,
    ):
        super().__init__()
        self.slides = []
        self.masks = []
        self.patches = []
        self.slide_idxs = []
        self.n_pos = []

        for slide_idx, (patches_path, slide_path, mask_path) in enumerate(
            zip(patches_paths, slide_paths, mask_paths)
        ):
            self.slides.append(Slide(slide_path, backend=slide_backend))
            self.masks.append(Slide(mask_path, backend=slide_backend))
            with open(patches_path, "r") as patch_file:
                reader = csv.DictReader(patch_file)
                for patch in reader:
                    self.patches.append(Patch.from_csv_row(patch))
                    # self.n_pos.append(patch["n_pos"])
                    self.slide_idxs.append(slide_idx)

        self.n_pos = np.array(self.n_pos, dtype=np.uint64)

        if stain_matrices_paths is None:
            self.stain_matrices = None
        else:
            self.stain_matrices = [np.load(path) for path in stain_matrices_paths]
        self.stain_augmentor = stain_augmentor
        self.transforms = Compose(ifnone(transforms, []))
        self.min_size = min_size
        self.clean()

    def __len__(self):
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        patch = self.patches[idx]
        slide_idx = self.slide_idxs[idx]
        slide = self.slides[slide_idx]
        mask = self.masks[slide_idx]

        slide_region = np.asarray(
            slide.read_region(patch.position, patch.level, patch.size).convert("RGB")
        )
        mask_region = np.asarray(
            mask.read_region(patch.position, patch.level, patch.size).convert("1"),
            dtype=np.uint8,
        )

        if self.stain_augmentor is not None:
            if self.stain_matrices is not None:
                stain_matrix = self.stain_matrices[slide_idx]
            else:
                stain_matrix = None
            slide_region = self.stain_augmentor(image=(slide_region, stain_matrix))[
                "image"
            ]

        if self.transforms:
            transformed = self.transforms(image=slide_region)

        target = label  # to define
        return transformed["image"], target

    def clean(self):
        patches = []
        slide_idxs = []
        idxs = []
        for i in range(len(self)):
            if self[i] is not None:
                patches.append(self.patches[i])
                slide_idxs.append(self.slide_idxs[i])
                idxs.append(i)

        self.patches = patches
        self.slide_idxs = slide_idxs
        self.n_pos = self.n_pos[idxs]


class BalancedRandomSampler(RandomSampler):
    def __init__(
        self,
        data_source: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = False,
        generator: Optional[torch.Generator] = None,
        p_pos: float = 0.5,
    ):
        self.p_pos = p_pos
        super().__init__(
            data_source,
            replacement=replacement,
            num_samples=num_samples,
            generator=generator,
        )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        num_samples = super().num_samples
        if self.replacement:
            return num_samples
        else:
            n_pos = (self.data_source.n_pos > 0).sum()
            max_avail = int(
                self.p_pos * n_pos + (1 - self.p_pos) * (len(self.data_source) - n_pos)
            )
            return min(num_samples, max_avail)

    def get_idxs(self) -> List[int]:
        mask = self.data_source.n_pos == 0
        avail = [mask.nonzero()[0].tolist(), (~mask).nonzero()[0].tolist()]
        avail = [cl_patches for cl_patches in avail if cl_patches]
        idxs = []
        for _ in range(self.num_samples):
            x = torch.rand(2, generator=self.generator)
            cl = min(int(x[0] < self.p_pos), len(avail) - 1)
            cl_patches = avail[cl]
            idx = int(x[1] * len(cl_patches))
            if self.replacement:
                patch = cl_patches[idx]
            else:
                patch = cl_patches.pop(idx)
                if len(cl_patches) == 0:
                    avail.pop(cl)
            idxs.append(patch)
        return idxs

    def __iter__(self) -> Iterator[int]:
        return iter(self.get_idxs())

    def __len__(self) -> int:
        return self.num_samples
