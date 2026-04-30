"""Training entry point.

Usage::

    python -m src.train --model tft   --config configs/default.yaml
    python -m src.train --model lstm  --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import build_tensors, rolling_split
from .data.label import continuous_target  # noqa: F401  (kept for API parity)
from .data.load import filter_half_year, load_holdings
from .data.tensor_dataset import HoldingsTensorDataset
from .features.preprocess import StandardScaler2D, StandardScaler3D
from .losses.quantile import QuantileLoss
from .models.lstm_baseline import LSTMBaseline, LSTMConfig
from .models.tft import TFTConfig, TemporalFusionTransformer
from .utils.seed import set_seed


def _build_model(name: str, cfg: dict, n_funds: int, n_stocks: int,
                 n_static: int, n_encoder: int):
    quantiles = tuple(cfg["model"]["quantiles"])
    common = dict(
        num_funds=n_funds,
        num_stocks=n_stocks,
        num_static_numeric=n_static,
        num_encoder_features=n_encoder,
        decoder_length=cfg["data"]["decoder_length"],
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        quantiles=quantiles,
    )
    if name == "tft":
        return TemporalFusionTransformer(TFTConfig(
            hidden_size=cfg["model"]["hidden_size"],
            lstm_layers=cfg["model"]["lstm_layers"],
            attention_heads=cfg["model"]["attention_heads"],
            **common,
        ))
    if name == "lstm":
        return LSTMBaseline(LSTMConfig(
            hidden_size=cfg["model"]["hidden_size"] * 2,
            num_layers=cfg["model"]["lstm_layers"],
            **common,
        ))
    raise ValueError(f"Unknown model: {name}")


def _epoch(model, loader, loss_fn, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total, n = 0.0, 0
    for batch in loader:
        out = model(
            fund_ids=batch["fund_ids"],
            stock_ids=batch["stock_ids"],
            static_numeric=batch["static_numeric"],
            encoder_numeric=batch["encoder_numeric"],
        )
        loss = loss_fn(out["quantile_pred"], batch["target"])
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bs = batch["target"].size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tft", "lstm"], default="tft")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--csv", default=None,
                        help="Override data.csv_path from the config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["training"]["seed"])

    csv_path = args.csv or cfg["data"]["csv_path"]
    df = load_holdings(csv_path)
    df = filter_half_year(df, cfg["data"]["half_year_endings"])
    if df.empty:
        raise SystemExit(
            "No rows after half-year filtering. Check data/raw/holdings.csv "
            "and configs/default.yaml:data.half_year_endings."
        )

    built = build_tensors(df)
    train_mask, val_mask, test_mask = rolling_split(
        built.period_idx,
        val_periods=cfg["splits"]["val_periods"],
        test_periods=cfg["splits"]["test_periods"],
    )

    enc_scaler = StandardScaler3D.fit(built.encoder_numeric[train_mask])
    stat_scaler = StandardScaler2D.fit(built.static_numeric[train_mask])
    encoder_numeric = enc_scaler.transform(built.encoder_numeric)
    static_numeric = stat_scaler.transform(built.static_numeric)

    def make_loader(mask, shuffle):
        ds = HoldingsTensorDataset(
            encoder_numeric=encoder_numeric[mask],
            static_numeric=static_numeric[mask],
            fund_ids=built.fund_ids[mask],
            stock_ids=built.stock_ids[mask],
            target=built.target[mask],
        )
        return DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["training"]["num_workers"],
        )

    train_loader = make_loader(train_mask, shuffle=True)
    val_loader = make_loader(val_mask, shuffle=False) if val_mask.any() else None
    test_loader = make_loader(test_mask, shuffle=False) if test_mask.any() else None

    n_funds = int(built.fund_ids.max()) + 2
    n_stocks = int(built.stock_ids.max()) + 2
    model = _build_model(
        args.model, cfg,
        n_funds=n_funds, n_stocks=n_stocks,
        n_static=built.static_numeric.shape[1],
        n_encoder=built.encoder_numeric.shape[2],
    )
    loss_fn = QuantileLoss(cfg["model"]["quantiles"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    bad = 0
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        train_loss = _epoch(model, train_loader, loss_fn, optimizer)
        msg = f"epoch {epoch:03d} | train_qloss={train_loss:.4f}"
        if val_loader is not None:
            with torch.no_grad():
                val_loss = _epoch(model, val_loader, loss_fn)
            msg += f" | val_qloss={val_loss:.4f}"
            if val_loss < best_val - 1e-6:
                best_val, bad = val_loss, 0
            else:
                bad += 1
        print(msg, flush=True)
        if val_loader is not None and bad >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if test_loader is not None:
        with torch.no_grad():
            test_loss = _epoch(model, test_loader, loss_fn)
        print(f"test_qloss={test_loss:.4f}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
