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
