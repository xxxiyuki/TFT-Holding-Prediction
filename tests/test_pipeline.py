"""Sanity tests using the bundled ``input_template.csv``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import build_tensors, rolling_split
from src.data.label import BUY, HOLD, SELL, discretize
from src.data.load import filter_half_year, load_holdings
from src.features.preprocess import StandardScaler2D, StandardScaler3D
from src.losses.quantile import QuantileLoss
from src.models.lstm_baseline import LSTMBaseline, LSTMConfig
from src.models.tft import TemporalFusionTransformer, TFTConfig

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "input_template.csv"


def test_load_template_schema():
    df = load_holdings(TEMPLATE)
    assert len(df) == 50
    assert set(["date", "fund_code", "stock_code"]).issubset(df.columns)
    assert df["date"].dt.year.iloc[0] == 2015


def test_half_year_filter_keeps_quarter_end():
    df = load_holdings(TEMPLATE)
    # March 31 is not a half-year ending; default config should drop it.
    out = filter_half_year(df, ["06-30", "12-31"])
    assert out.empty
    # But March 31 should be kept if explicitly requested.
    out2 = filter_half_year(df, ["03-31"])
    assert len(out2) == 50


def test_discretize_thresholds():
    vals = np.array([0.5, 0.05, -0.5, 0.0, np.nan, -0.1, 0.1], dtype="float32")
    out = discretize(vals, buy_threshold=0.10, sell_threshold=-0.10)
    # >0.10 BUY, <-0.10 SELL, otherwise HOLD (incl. NaN and exact thresholds)
    assert out.tolist() == [BUY, HOLD, SELL, HOLD, HOLD, HOLD, HOLD]


def test_discretize_validates_bounds():
    with pytest.raises(ValueError):
        discretize(np.zeros(1, dtype="float32"), 0.0, 0.0)


def test_build_tensors_shapes():
    df = load_holdings(TEMPLATE)
    built = build_tensors(df)
    n = len(df)
    assert built.encoder_numeric.shape == (n, 6, 2)
    assert built.static_numeric.shape[0] == n
    assert built.fund_ids.shape == (n,)
    assert built.target.shape == (n,)
    assert (built.fund_ids > 0).all()  # 0 is reserved for unknown


def test_rolling_split_requires_enough_periods():
    # Only one period in the template -> cannot split.
    df = load_holdings(TEMPLATE)
    built = build_tensors(df)
    with pytest.raises(ValueError):
        rolling_split(built.period_idx, val_periods=1, test_periods=1)


def test_scaler_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3)).astype("float32")
    s = StandardScaler2D.fit(x)
    z = s.transform(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1, atol=1e-5)

    x3 = rng.normal(size=(4, 6, 2)).astype("float32")
    s3 = StandardScaler3D.fit(x3)
    z3 = s3.transform(x3)
    assert z3.shape == x3.shape


def test_quantile_loss_pinball_value():
    pred = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    true = torch.tensor([1.0], dtype=torch.float32)
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    # err = 1, losses = max(q*1, (q-1)*1) = q for positive err
    expected = (0.1 + 0.5 + 0.9) / 3
    assert abs(loss_fn(pred, true).item() - expected) < 1e-6


def test_quantile_loss_rejects_bad_shape():
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    with pytest.raises(ValueError):
        loss_fn(torch.zeros(2, 2), torch.zeros(2))


def _toy_batch(n=4, t=6, f_enc=2, f_stat=5, n_funds=3, n_stocks=4):
    torch.manual_seed(0)
    return dict(
        fund_ids=torch.randint(1, n_funds, (n,)),
        stock_ids=torch.randint(1, n_stocks, (n,)),
        static_numeric=torch.randn(n, f_stat),
        encoder_numeric=torch.randn(n, t, f_enc),
    )


def test_tft_forward_and_backward():
    cfg = TFTConfig(
        num_funds=4, num_stocks=5, num_static_numeric=5, num_encoder_features=2,
        hidden_size=16, embedding_dim=4, attention_heads=2, lstm_layers=1,
    )
    model = TemporalFusionTransformer(cfg)
    batch = _toy_batch()
    out = model(**batch)
    assert out["quantile_pred"].shape == (4, 3)
    assert out["static_var_weights"].shape == (4, 2 + 5)
    assert out["encoder_var_weights"].shape == (4, 6, 2)
    assert out["attention_weights"].shape == (4, 1, 7)

    target = torch.randn(4)
    loss = QuantileLoss(list(cfg.quantiles))(out["quantile_pred"], target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_lstm_baseline_forward_and_backward():
    cfg = LSTMConfig(
        num_funds=4, num_stocks=5, num_static_numeric=5, num_encoder_features=2,
        hidden_size=16, embedding_dim=4,
    )
    model = LSTMBaseline(cfg)
    batch = _toy_batch()
    out = model(**batch)
    assert out["quantile_pred"].shape == (4, 3)
    target = torch.randn(4)
    loss = QuantileLoss(list(cfg.quantiles))(out["quantile_pred"], target)
    loss.backward()


def test_tft_attention_mask_is_causal():
    cfg = TFTConfig(
        num_funds=4, num_stocks=5, num_static_numeric=5, num_encoder_features=2,
        hidden_size=16, embedding_dim=4, attention_heads=2, decoder_length=2,
    )
    model = TemporalFusionTransformer(cfg).eval()
    batch = _toy_batch()
    out = model(**batch)
    # decoder_length=2 -> attention_weights shape (B, 2, T+2=8); the second
    # decoder step must place no weight on the (yet unseen) third decoder
    # position. Encoder positions = first 6.
    attn = out["attention_weights"]
    # First decoder step attends to positions 0..6 (incl. itself), zero on 7
    assert torch.allclose(attn[:, 0, 7], torch.zeros_like(attn[:, 0, 7]))
