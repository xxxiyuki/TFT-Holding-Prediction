"""Quantile (pinball) loss for probabilistic regression."""

from __future__ import annotations

import torch
from torch import nn


class QuantileLoss(nn.Module):
    """Average pinball loss across a list of quantiles.

    ``y_pred`` has shape ``(batch, n_quantiles)`` and ``y_true`` has
    shape ``(batch,)``.
    """

    def __init__(self, quantiles: list[float]):
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles must be a non-empty list")
        for q in quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(f"quantile must be in (0, 1), got {q}")
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() != 2 or y_pred.size(1) != self.quantiles.numel():
            raise ValueError(
                f"y_pred shape {tuple(y_pred.shape)} incompatible with "
                f"{self.quantiles.numel()} quantiles"
            )
        y_true = y_true.view(-1, 1)
        errors = y_true - y_pred
        q = self.quantiles.view(1, -1).to(errors.device)
        losses = torch.maximum(q * errors, (q - 1.0) * errors)
        return losses.mean()
