"""CSV loading utilities for the holdings dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema


def load_holdings(csv_path: str | Path) -> pd.DataFrame:
    """Load a holdings CSV and validate its schema.

    The CSV is expected to match the column layout of
    ``input_template.csv``. A leading unnamed index column is tolerated.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Holdings CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Drop a leading unnamed index column produced by ``to_csv`` defaults.
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    missing = [c for c in schema.all_required_columns() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(schema.KEY_COLUMNS).reset_index(drop=True)
    return df


def filter_half_year(df: pd.DataFrame, endings: list[str]) -> pd.DataFrame:
    """Keep only rows whose date matches one of the ``MM-DD`` endings."""
    keep = df["date"].dt.strftime("%m-%d").isin(endings)
    return df.loc[keep].reset_index(drop=True)
