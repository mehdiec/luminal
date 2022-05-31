import torch.nn as nn

from typing import Tuple, Sequence


def get_loss(name: str) -> nn.Module:
    r"""
    Get loss object from its name. If name is not recognized, raises `ValueError`.

    Args:
        name: can either be "bce" for :class:`~torch.nn.BCEWithLogitsLoss`, "focal" for
            :class:`FocalLoss`, "dice" for :class:`DiceLoss` or
            "sum_loss1_coef1_loss2_coef2_..._lossN_coefN" for :class:`SumLosses`.

    Returns:
        Loss function as a `torch.nn.Module` object.
    """
    split_name = name.split("_")
    if split_name[0] == "bce":
        return nn.BCEWithLogitsLoss()
    if split_name[0] == "ce":
        return nn.CrossEntropyLoss()  # weight=torch.tensor([3.0, 2.0, 1.0])
    elif split_name[0] == "sum":
        names = split_name[1::2]
        coefs = map(float, split_name[2::2])
        losses_with_coefs = [(get_loss(n), c) for n, c in zip(names, coefs)]
        return SumLosses(*losses_with_coefs)
    else:
        raise ValueError(f"{name} not recognized as a loss function")


def get_loss_name(loss: nn.Module) -> str:
    r"""
    Get a loss object's name. If object is not recognized, will return the class name.

    Args:
        loss: loss module.

    Returns:
        Either "bce" for :class:`~torch.nn.BCEWithLogitsLoss`, "focal" for
            :class:`FocalLoss`, "dice" for :class:`DiceLoss`,
            "sum_loss1_coef1_loss2_coef2_..._lossN_coefN" for :class:`SumLosses` or
            the class name if none of the above.
    """
    if isinstance(loss, nn.BCEWithLogitsLoss):
        return "bce"
    elif isinstance(loss, nn.CrossEntropyLoss):
        return "ce"
    elif isinstance(loss, SumLosses):
        name = "sum"
        for loss_func, coef in loss.losses_with_coef:
            name += f"_{get_loss_name(loss_func)}_{coef}"
        return name
    else:
        return loss.__class__.__name__


class SumLosses(nn.Module):
    r"""
    Wrapper used to compute a weighted sum of different losses.

    Args:
        *losses_with_coef: tuples containg a loss function and a float corresponding to
            the coefficient to use for the weighted sum of the losses.
    """

    def __init__(self, *losses_with_coef: Sequence[Tuple[nn.Module, float]]):
        super().__init__()
        self.losses_with_coef = losses_with_coef

    def forward(self, input, target):
        loss = 0
        for loss_func, coef in self.losses_with_coef:
            loss += coef * loss_func(input, target)
        return loss
