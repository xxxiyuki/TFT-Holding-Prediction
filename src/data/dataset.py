"""Build seq-to-seq tensors from the flat holdings CSV.

Each raw row already carries 6 lagged values for ``fund_stock_mv`` and
``fund_stock_sh``. We reshape those lags into a length-6 *encoder*
sequence representing the historical 3-year holding trajectory at
half-yearly frequency, and use the current row as the *decoder* step
(predict the next half-year's relative holding change).

Static covariates (fund/stock embeddings + slow-moving fund/macro/peer
features) are passed verbatim. Time-varying covariates that are only
available at the *current* date (Barra exposures, returns, macro) are
broadcast across encoder positions to keep the tensor shapes regular;
the variable-selection network learns to weight them appropriately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema


# Encoder time-varying numeric features (per half-year): the two
# holding-trajectory channels reconstructed from the lag columns.
ENCODER_TIME_VARYING = ["fund_stock_mv", "fund_stock_sh"]

# Decoder / static numeric features: everything else that is not a lag,
# not a key, not the target source. Broadcast across the encoder so the
# VSN sees them at every step.
def _static_numeric_columns() -> list[str]:
    drop = set(
        schema.KEY_COLUMNS
        + schema.DROP_COLUMNS
        + schema.HOLDING_LAGS
        + [schema.TARGET_SOURCE]
        + ENCODER_TIME_VARYING
    )
    return [c for c in schema.all_feature_columns() if c not in drop]


STATIC_NUMERIC = _static_numeric_columns()


@dataclass
class BuiltDataset:
    """Container for tensors produced by :func:`build_tensors`."""

    encoder_numeric: np.ndarray   # (N, T=6, F_enc)
    static_numeric: np.ndarray    # (N, F_stat)
    fund_ids: np.ndarray          # (N,) int64
    stock_ids: np.ndarray         # (N,) int64
    target: np.ndarray            # (N,) float32 (continuous)
    period_idx: np.ndarray        # (N,) int64, half-year ordinal of current row
    fund_codes: np.ndarray        # (N,) raw codes (for diagnostics)
    stock_codes: np.ndarray
    static_feature_names: list[str]
    encoder_feature_names: list[str]


def _encode_categorical(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    categories = sorted(series.dropna().unique().tolist())
    mapping = {c: i + 1 for i, c in enumerate(categories)}  # reserve 0 for unknown
    ids = series.map(mapping).fillna(0).astype("int64").to_numpy()
    return ids, mapping


def _build_encoder_sequence(df: pd.DataFrame) -> np.ndarray:
    """Stack ``fund_stock_mv/sh`` lags into ``(N, 6, 2)`` arrays.

    Position 0 is the oldest (lag6); position 5 is the most recent (lag1).
    The current value is *not* included on the encoder side: it is the
    decoder input / leakage source.
    """
    n = len(df)
    out = np.zeros((n, 6, len(ENCODER_TIME_VARYING)), dtype="float32")
    for j, name in enumerate(ENCODER_TIME_VARYING):
        for t, lag in enumerate(range(6, 0, -1)):  # lag6 -> lag1
            out[:, t, j] = df[f"{name}_lag{lag}"].astype("float32").to_numpy()
    return out


def _half_year_period_index(dates: pd.Series) -> np.ndarray:
    """Map each date to a monotonically increasing half-year ordinal."""
    years = dates.dt.year.to_numpy()
    halves = (dates.dt.month.to_numpy() > 6).astype("int64")  # 0 = H1, 1 = H2
    return years * 2 + halves


def build_tensors(df: pd.DataFrame) -> BuiltDataset:
    """Reshape a half-year-filtered holdings frame into tensors."""
    if df.empty:
        raise ValueError("Cannot build tensors from an empty DataFrame")

    df = df.reset_index(drop=True)

    fund_ids, _ = _encode_categorical(df["fund_code"])
    stock_ids, _ = _encode_categorical(df["stock_code"])

    encoder_numeric = _build_encoder_sequence(df)
    static_numeric = (
        df[STATIC_NUMERIC].astype("float32").fillna(0.0).to_numpy()
    )
    target = df[schema.TARGET_SOURCE].astype("float32").fillna(0.0).to_numpy()
    period_idx = _half_year_period_index(df["date"])

    return BuiltDataset(
        encoder_numeric=encoder_numeric,
        static_numeric=static_numeric,
        fund_ids=fund_ids,
        stock_ids=stock_ids,
        target=target,
        period_idx=period_idx,
        fund_codes=df["fund_code"].to_numpy(),
        stock_codes=df["stock_code"].to_numpy(),
        static_feature_names=list(STATIC_NUMERIC),
        encoder_feature_names=list(ENCODER_TIME_VARYING),
    )


def rolling_split(
    period_idx: np.ndarray,
    val_periods: int = 1,
    test_periods: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-based split by half-year period.

    Returns boolean masks for train / val / test. The most recent
    ``test_periods`` half-years go to test, the preceding
    ``val_periods`` go to validation, and the rest is training.
    """
    if val_periods < 0 or test_periods < 0:
        raise ValueError("val_periods and test_periods must be non-negative")
    unique_periods = np.unique(period_idx)
    if len(unique_periods) < val_periods + test_periods + 1:
        raise ValueError(
            f"Not enough distinct half-year periods ({len(unique_periods)}) "
            f"for val={val_periods} + test={test_periods} + train>=1."
        )
    test_set = set(unique_periods[-test_periods:].tolist()) if test_periods else set()
    val_set = (
        set(unique_periods[-(test_periods + val_periods) : -test_periods].tolist())
        if val_periods
        else set()
    )
    test_mask = np.isin(period_idx, list(test_set))
    val_mask = np.isin(period_idx, list(val_set))
    train_mask = ~(test_mask | val_mask)
    return train_mask, val_mask, test_mask
