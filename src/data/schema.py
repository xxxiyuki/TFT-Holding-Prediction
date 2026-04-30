"""Column schema for the holdings dataset.

The raw CSV (``input_template.csv``) is keyed by ``(date, fund_code,
stock_code)`` and has 71 columns. This module groups columns by their
role for downstream feature engineering and model wiring.
"""

from __future__ import annotations

# --- identifiers -----------------------------------------------------------
KEY_COLUMNS = ["date", "fund_code", "stock_code"]
DROP_COLUMNS = ["id"]  # opaque row id, not a feature

# --- categorical embeddings (static for a given group) ---------------------
STATIC_CATEGORICALS = ["fund_code", "stock_code"]

# --- fund-level numeric features (vary by date, same across stocks) --------
FUND_FEATURES = [
    "fund_value_tile",
    "fund_size_tile",
    "fund_m1ret",
    "fund_m2ret",
    "fund_m3ret",
    "fund_m4ret",
    "fund_m5ret",
    "fund_m6ret",
    "fund_aum",
    "Beta",
    "BookToPrice",
    "EarningsYield",
    "Growth",
    "Leverage",
    "Liquidity",
    "Momentum",
    "NonlinearSize",
    "ResidualVolatility",
    "Size",
    "fund_cat_1m",
    "fund_cat_3m",
    "fund_cat_6m",
    "fund_inflow",
    "fund_inflow_shift",
    "消费_当期占比",
    "稳定_当期占比",
    "金融_当期占比",
    "周期_当期占比",
    "成长_当期占比",
    "周期_过去1年平均占比",
    "成长_过去1年平均占比",
    "消费_过去1年平均占比",
    "稳定_过去1年平均占比",
    "金融_过去1年平均占比",
    "周期_历史以来平均占比",
    "成长_历史以来平均占比",
    "消费_历史以来平均占比",
    "稳定_历史以来平均占比",
    "金融_历史以来平均占比",
]

# --- macro features (global per date, known a priori) ----------------------
MACRO_FEATURES = [
    "macro_term_spreads_5Y1Y",
    "macro_term_spreads_10Y3M",
    "macro_default_spreads",
    "macro_short_yields_nominal",
    "macro_long_ yields_nominal",   # note: source column has a space; preserved
    "macro_short_yields_real",
    "macro_long_ yields_real",
]

# --- holding features (current period) -------------------------------------
HOLDING_CURRENT = ["fund_stock_mv", "fund_stock_sh"]

# --- holding lag features (lag1 .. lag6) -----------------------------------
# These are pre-flattened time series and will be reshaped to a length-6
# encoder sequence by the dataset builder.
HOLDING_LAGS = [f"fund_stock_{name}_lag{i}" for name in ("mv", "sh") for i in range(1, 7)]

# --- peer / category aggregates (stock × date level) -----------------------
PEER_FEATURES = [
    "同类持有该股票的总市值(亿元)",
    "同类持有该股票的平均比例",
    "同类增持比例",
    "同类维持持仓比例",
    "同类减持比例",
    "增减持-绝对变化",  # absolute change; kept as feature
]

# --- target source ---------------------------------------------------------
# Used to build the continuous regression target. Dropped from features to
# avoid label leakage.
TARGET_SOURCE = "增减持-相对变化"

# --- aggregated views ------------------------------------------------------
def all_feature_columns() -> list[str]:
    """All numeric feature columns used by the model."""
    return (
        FUND_FEATURES
        + MACRO_FEATURES
        + HOLDING_CURRENT
        + HOLDING_LAGS
        + PEER_FEATURES
    )


def all_required_columns() -> list[str]:
    """All columns the loader must find in the raw CSV."""
    return KEY_COLUMNS + all_feature_columns() + [TARGET_SOURCE]
