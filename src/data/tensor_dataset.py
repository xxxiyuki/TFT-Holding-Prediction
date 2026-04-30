"""Tensor dataset wrapper used by the train/eval loops."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class HoldingsTensorDataset(Dataset):
    """Wraps the numpy arrays produced by :func:`build_tensors`."""

    def __init__(
        self,
        encoder_numeric: np.ndarray,
        static_numeric: np.ndarray,
        fund_ids: np.ndarray,
        stock_ids: np.ndarray,
        target: np.ndarray,
    ):
        self.encoder_numeric = torch.from_numpy(encoder_numeric.astype("float32"))
        self.static_numeric = torch.from_numpy(static_numeric.astype("float32"))
        self.fund_ids = torch.from_numpy(fund_ids.astype("int64"))
        self.stock_ids = torch.from_numpy(stock_ids.astype("int64"))
        self.target = torch.from_numpy(target.astype("float32"))

    def __len__(self) -> int:
        return self.target.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "fund_ids": self.fund_ids[idx],
            "stock_ids": self.stock_ids[idx],
            "static_numeric": self.static_numeric[idx],
            "encoder_numeric": self.encoder_numeric[idx],
            "target": self.target[idx],
        }
