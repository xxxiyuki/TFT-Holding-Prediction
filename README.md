# TFT Holding Prediction

A from-scratch **Temporal Fusion Transformer (TFT)** framework for predicting
A-share mutual fund holding changes at half-yearly frequency.

## Highlights

- Sequence-to-sequence model with the four canonical TFT components:
  - **Variable Selection Networks** (separate networks for static, encoder
    and decoder inputs) with softmax importance weights.
  - **LSTM encoder/decoder** seeded by static-context vectors produced from
    fund/stock embeddings + slow-moving covariates.
  - **Interpretable multi-head self-attention** (shared value projection,
    averaged across heads) over the fused encoder/decoder sequence.
  - **Quantile (pinball) loss** producing one prediction per quantile (default
    `0.1 / 0.5 / 0.9`).
- LSTM **baseline** sharing the same input contract for a fair comparison.
- 100% PyTorch CPU-only, no `pytorch-lightning` / `pytorch-forecasting`
  dependencies.

## Repository layout

```
configs/default.yaml          # all hyper-parameters and paths
src/
  data/schema.py              # 71-column grouping for the holdings CSV
  data/load.py                # CSV loading + half-year filtering
  data/label.py               # continuous target + 3-class derivation (±10%)
  data/dataset.py             # reshape lag columns into (B, T=6, F) tensors
  data/tensor_dataset.py      # torch Dataset wrapper
  features/preprocess.py      # per-feature standardisation
  models/tft.py               # GRN, VSN, attention, full TFT
  models/lstm_baseline.py     # comparable LSTM baseline
  losses/quantile.py          # pinball loss
  train.py                    # CLI: `--model {tft,lstm}`
  eval.py                     # quantile loss + 3-class report from median
tests/test_pipeline.py        # schema, dataset, loss and model smoke tests
input_template.csv            # 50-row schema example
```

## Setup (CPU-only)

```bash
pip install -r requirements-cpu.txt   # uses the official PyTorch CPU index
```

If your environment cannot reach `download.pytorch.org`, plain
`pip install torch` from PyPI works as well; the code only relies on stable
core PyTorch APIs.

## Data

Drop a CSV with the same column layout as `input_template.csv` at
`data/raw/holdings.csv`, or pass `--csv path/to/file.csv` on the CLI.
The loader validates required columns and parses dates automatically.

## Training

```bash
python -m src.train --model tft  --config configs/default.yaml
python -m src.train --model lstm --config configs/default.yaml
```

The trainer:

1. Filters the CSV to half-year endings (default `06-30` and `12-31`).
2. Reshapes `fund_stock_mv/sh_lag1..6` into a length-6 encoder sequence;
   the current row becomes the 1-step decoder target.
3. Performs a **rolling time-based split**: most recent half-year → test,
   preceding half-year → validation, all earlier periods → train.
4. Fits standardisation on train only and applies it to all splits.
5. Trains with quantile loss + early stopping and prints per-epoch losses.

## Labels

Two views of the same target are supported:

- **Continuous regression**: `增减持-相对变化` directly, optimised with
  the pinball loss.
- **3-class derivation** (post-hoc, in `src/eval.py`):
  `> +10%` → BUY, `< -10%` → SELL, otherwise HOLD. Applied to the median
  quantile prediction to report macro-F1 / accuracy alongside the loss.

Thresholds and quantiles live in `configs/default.yaml`.

## Tests

```bash
pytest -q
```

Covers schema loading on the bundled `input_template.csv`, label
discretisation boundaries, scaler invariants, quantile-loss math, and forward/
backward passes for both the TFT and the LSTM baseline (incl. attention-mask
causality).
