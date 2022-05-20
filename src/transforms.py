from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
import numpy as np
from pathaia.util.basic import ifnone
from typing import Callable, Optional, Any, List, Sequence, Tuple, Dict
from nptyping import NDArray
from numbers import Number
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from torchvision.transforms.functional import to_tensor
import torch
from pathaia.util.types import NDImage, NDGrayImage, NDByteImage
from staintools.miscellaneous.optical_density_conversion import convert_RGB_to_OD
import spams


def get_concentrations(
    img: NDByteImage, stain_matrix: NDArray[(2, 3), float], regularizer: float = 0.01
) -> NDArray[(Any, Any, 2), float]:
    OD = convert_RGB_to_OD(img).reshape((-1, 3))
    HE = spams.lasso(
        X=OD.T,
        D=stain_matrix.T,
        mode=2,
        lambda1=regularizer,
        pos=True,
        numThreads=1,
    )
    return HE.toarray().T


class ToTensor(DualTransform):
    def __init__(
        self, transpose_mask: bool = False, always_apply: bool = True, p: float = 1
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> Dict[str, Callable[[NDImage], torch.Tensor]]:
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img: NDImage, **params) -> torch.Tensor:
        return to_tensor(img)

    def apply_to_mask(self, mask: NDImage, **params) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("transpose_mask",)


# class ToHSV(DualTransform):
#     def __init__(
#         self, transpose_mask: bool = False, always_apply: bool = True, p: float = 1
#     ):
#         super().__init__(always_apply=always_apply, p=p)
#         self.transpose_mask = transpose_mask

#     @property
#     def targets(self) -> Dict[str, Callable[[NDImage], torch.Tensor]]:
#         return {"image": self.apply, "mask": self.apply_to_mask}

#     def apply(self, img: NDImage, **params) -> torch.Tensor:
#         return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     def apply_to_mask(self, mask: NDImage, **params) -> torch.Tensor:
#         if self.transpose_mask and mask.ndim == 3:
#             mask = mask.transpose(2, 0, 1)
#         return torch.from_numpy(mask)

#     def get_transform_init_args_names(self) -> Tuple[str]:
#         return ("transpose_mask",)


class StainAugmentor(ImageOnlyTransform):
    def __init__(
        self,
        alpha_range: float = 0.3,
        beta_range: float = 0.2,
        alpha_stain_range: float = 0.3,
        beta_stain_range: float = 0.2,
        he_ratio: float = 0.3,
        always_apply: bool = True,
        p: float = 1,
    ):
        super(StainAugmentor, self).__init__(always_apply, p)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.alpha_stain_range = alpha_stain_range
        self.beta_stain_range = beta_stain_range
        self.he_ratio = he_ratio

    def get_params(self):
        return {
            "alpha": np.random.uniform(
                1 - self.alpha_range, 1 + self.alpha_range, size=2
            ),
            "beta": np.random.uniform(-self.beta_range, self.beta_range, size=2),
            "alpha_stain": np.stack(
                (
                    np.random.uniform(
                        1 - self.alpha_stain_range * self.he_ratio,
                        1 + self.alpha_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        1 - self.alpha_stain_range,
                        1 + self.alpha_stain_range,
                        size=3,
                    ),
                ),
            ),
            "beta_stain": np.stack(
                (
                    np.random.uniform(
                        -self.beta_stain_range * self.he_ratio,
                        self.beta_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        -self.beta_stain_range, self.beta_stain_range, size=3
                    ),
                ),
            ),
        }

    def initialize(
        self,
        alpha: Optional[NDArray[(Any, ...), float]],
        beta: Optional[NDArray[(Any, ...), float]],
        shape: Tuple[int, ...] = 2,
    ) -> Tuple[NDArray[(Any, ...)], NDArray[(Any, ...)]]:
        alpha = ifnone(alpha, np.ones(shape))
        beta = ifnone(beta, np.zeros(shape))
        return alpha, beta

    def apply(
        self,
        image_and_stain: Tuple[
            NDArray[(Any, Any, 3), Number], Optional[NDArray[(2, 3), float]]
        ],
        alpha: Optional[NDArray[(2,), float]] = None,
        beta: Optional[NDArray[(2,), float]] = None,
        alpha_stain: Optional[NDArray[(2, 3), float]] = None,
        beta_stain: Optional[NDArray[(2, 3), float]] = None,
        **params
    ) -> NDArray[(Any, Any, 3), Number]:
        image = image_and_stain
        stain_matrix = None
        alpha, beta = self.initialize(alpha, beta, shape=2)
        alpha_stain, beta_stain = self.initialize(alpha_stain, beta_stain, shape=(2, 3))
        if not image.dtype == np.uint8:
            image = (image * 255).astype(np.uint8)
        if stain_matrix is None:
            stain_matrix = VahadaneStainExtractor.get_stain_matrix(image)

        HE = get_concentrations(image, stain_matrix)
        stain_matrix = stain_matrix * alpha_stain + beta_stain
        stain_matrix = np.clip(stain_matrix, 0, 1)
        HE = np.where(HE > 0.2, HE * alpha[None] + beta[None], HE)
        out = np.exp(-np.dot(HE, stain_matrix)).reshape(image.shape)
        out = np.clip(out, 0, 1)
        return out.astype(np.float32)
