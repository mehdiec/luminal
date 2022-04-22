from typing import Tuple, Sequence
import torch
import torch.nn as nn
from apriorics.metrics import _flatten, _reduce, dice_score


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
    elif split_name[0] == "focal":
        return FocalLoss()
    elif split_name[0] == "dice":
        return DiceLoss()
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
    elif isinstance(loss, FocalLoss):
        return "focal"
    elif isinstance(loss, DiceLoss):
        return "dice"
    elif isinstance(loss, SumLosses):
        name = "sum"
        for loss_func, coef in loss.losses_with_coef:
            name += f"_{get_loss_name(loss_func)}_{coef}"
        return name
    else:
        return loss.__class__.__name__


def dice_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    Compute dice loss (given by 1-dice_score) for given prediction and target.

    Args:
        input: predicted input tensor of shape (N, ...).
        target: target ground truth tensor of shape (N, ...).
        smooth: smooth value for dice score.
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".

    Returns:
        Computed dice loss, optionally reduced using specified reduction method.
    """
    input = torch.sigmoid(input)
    dice = dice_score(input, target, smooth=smooth, reduction=reduction)
    return 1 - dice


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    beta: float = 0.5,
    gamma: float = 2.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    r"""
    Compute focal loss (given by
    :math:`FL(y, t) = -\beta t(1-y)^\gamma \log{y} - (1-\beta)(1-t)(y^\gamma \log{y})`)
    for given prediction and target. Sigmoid is applied to input before computation.

    Args:
        input: predicted input tensor of shape (N, ...).
        target: target ground truth tensor of shape (N, ...).
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".
        beta: coefficient ratio for positive class. Negative class has coefficient
            `1-beta`.
        gamma: focusing parameter for focal loss. When `gamma=0`, focal loss is
            equivalent to binary cros-sentropy. When `gamma` increases, highly
            misclassified predictions will have higher weight.
        eps: term added for numerical stability.

    Returns:
        Computed focal loss, optionally reduced using specified reduction method.
    """
    target = _flatten(target).to(dtype=input.dtype)
    input = _flatten(input)
    input = torch.sigmoid(input).clamp(eps, 1 - eps)
    focal = -(
        beta * target * (1 - input).pow(gamma) * input.log()
        + (1 - beta) * (1 - target) * input.pow(gamma) * (1 - input).log()
    ).mean(-1)
    return _reduce(focal, reduction=reduction)


class FocalLoss(nn.Module):
    r"""
    `torch.nn.Module` for focal loss (given by
    :math:`FL(y, t) = -\beta t(1-y)^\gamma \log{y} - (1-\beta)(1-t)(y^\gamma \log{y})`)
    computation.

    Args:
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".
        beta: coefficient ratio for positive class. Negative class has coefficient
            `1-beta`.
        gamma: focusing parameter for focal loss. When `gamma=0`, focal loss is
            equivalent to binary cros-sentropy. When `gamma` increases, highly
            misclassified predictions will have higher weight.
    """

    def __init__(self, beta: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        loss = focal_loss(
            input, target, beta=self.beta, gamma=self.gamma, reduction=self.reduction
        )
        return loss


class DiceLoss(nn.Module):
    r"""
    `torch.nn.Module` for dice loss (given by 1-dice_score) computation.

    Args:
        smooth: smooth value for dice score.
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".
    """

    def __init__(self, smooth=1, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        loss = dice_loss(input, target, smooth=self.smooth, reduction=self.reduction)
        return loss


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
