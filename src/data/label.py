"""Label utilities: continuous regression target and 3-class derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd


BUY, HOLD, SELL = 2, 1, 0


def continuous_target(df: pd.DataFrame, source_column: str) -> np.ndarray:
    """Return the raw continuous target (relative holding change)."""
    if source_column not in df.columns:
        raise KeyError(f"Target source column missing: {source_column}")
    return df[source_column].astype("float32").to_numpy()


def discretize(
    values: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
) -> np.ndarray:
    """Map continuous holding-change values to {SELL=0, HOLD=1, BUY=2}.

    A value strictly above ``buy_threshold`` is BUY, strictly below
    ``sell_threshold`` is SELL, otherwise HOLD. NaNs become HOLD.
    """
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must be greater than sell_threshold")

    out = np.full(values.shape, HOLD, dtype=np.int64)
    finite = np.isfinite(values)
    out[finite & (values > buy_threshold)] = BUY
    out[finite & (values < sell_threshold)] = SELL
    return out
