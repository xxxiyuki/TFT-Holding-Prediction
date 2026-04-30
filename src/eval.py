"""Evaluation utilities: quantile loss + 3-class metrics from the median."""

from __future__ import annotations

import numpy as np

from .data.label import discretize


def quantile_loss(y_pred: np.ndarray, y_true: np.ndarray,
                  quantiles: list[float]) -> float:
    """Average pinball loss; ``y_pred`` shape ``(N, n_quantiles)``."""
    y_true = y_true.reshape(-1, 1)
    q = np.asarray(quantiles, dtype="float32").reshape(1, -1)
    err = y_true - y_pred
    return float(np.maximum(q * err, (q - 1.0) * err).mean())


def median_index(quantiles: list[float]) -> int:
    arr = np.asarray(quantiles, dtype="float32")
    return int(np.argmin(np.abs(arr - 0.5)))


def classification_report_3way(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    quantiles: list[float],
    buy_threshold: float,
    sell_threshold: float,
) -> dict:
    """Derive 3-class metrics from the median quantile prediction."""
    med = y_pred[:, median_index(quantiles)]
    pred_cls = discretize(med, buy_threshold, sell_threshold)
    true_cls = discretize(y_true, buy_threshold, sell_threshold)

    classes = (0, 1, 2)
    per_class = {}
    f1s = []
    for c in classes:
        tp = int(((pred_cls == c) & (true_cls == c)).sum())
        fp = int(((pred_cls == c) & (true_cls != c)).sum())
        fn = int(((pred_cls != c) & (true_cls == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1s.append(f1)

    accuracy = float((pred_cls == true_cls).mean())
    macro_f1 = float(np.mean(f1s))
    return {"per_class": per_class, "macro_f1": macro_f1, "accuracy": accuracy}
