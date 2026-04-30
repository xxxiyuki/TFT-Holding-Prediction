"""Standardization utilities (fit on train, apply to all splits)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler2D:
    """Per-feature mean/std scaler for 2-D arrays."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler2D":
        mean = np.nanmean(x, axis=0).astype("float32")
        std = np.nanstd(x, axis=0).astype("float32")
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")


@dataclass
class StandardScaler3D:
    """Per-feature mean/std scaler for ``(N, T, F)`` arrays.

    Statistics are pooled across the time axis to keep a single set of
    mean/std per feature -- consistent with how the lag columns were
    originally produced from one underlying series.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler3D":
        flat = x.reshape(-1, x.shape[-1])
        mean = np.nanmean(flat, axis=0).astype("float32")
        std = np.nanstd(flat, axis=0).astype("float32")
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")
