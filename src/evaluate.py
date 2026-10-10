"""Evaluation metrics, threshold analysis and figures (SPEC §10).

Primary metric: PR-AUC (average precision). The decision threshold is chosen
on calibrated validation probabilities by minimising expected
misclassification cost (SPEC §10.2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure  # object-oriented API: no GUI backend needed
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from src import config

# Chart style shared by report figures: validated colorblind-safe categorical slots
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 9, "savefig.dpi": 150, "savefig.bbox": "tight",
}

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
DISPLAY_THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)  # SPEC §10.2 table range


def probability_metrics(y_true, proba) -> dict[str, float]:
    """Threshold-free metrics for predicted default probabilities.

    ``mean_predicted`` vs ``observed_rate`` is calibration-in-the-large: a
    well-calibrated model's average PD matches the actual default rate.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_predicted": float(proba.mean()),
        "observed_rate": float(y_true.mean()),
    }


def metrics_at_threshold(y_true, proba, threshold: float, cost_fn: float = config.COST_FALSE_NEGATIVE,
                         cost_fp: float = config.COST_FALSE_POSITIVE) -> dict:
    """Confusion matrix and positive-class metrics when flagging PD >= threshold."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "flagged_rate": float(flagged.mean()),
        "cost": float(cost_fn * fn + cost_fp * fp),
    }


def threshold_table(y_true, proba, thresholds=THRESHOLD_GRID, **costs) -> pd.DataFrame:
    return pd.DataFrame([metrics_at_threshold(y_true, proba, t, **costs) for t in thresholds])


def choose_threshold(y_true, proba, thresholds=THRESHOLD_GRID, **costs) -> tuple[float, pd.DataFrame]:
    """SPEC §10.2: minimise cost_fn * FN + cost_fp * FP. If several thresholds share the
    minimum cost, the middle one is taken (robust to small probability shifts)."""
    table = threshold_table(y_true, proba, thresholds, **costs)
    tied = table.loc[table["cost"] == table["cost"].min(), "threshold"].to_numpy()
    return float(tied[len(tied) // 2]), table


def plot_threshold_analysis(table: pd.DataFrame, chosen: float, path: Path, title_suffix: str = "") -> None:
    """Precision/recall and expected cost across thresholds, in two stacked panels."""
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(7.5, 6))
        top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[3, 2])
        top.plot(table["threshold"], table["precision"], color=BLUE, lw=2, label="Precision")
        top.plot(table["threshold"], table["recall"], color=ORANGE, lw=2, label="Recall")
        for column, color in (("precision", BLUE), ("recall", ORANGE)):
            last = table.iloc[-1] if column == "precision" else table.iloc[len(table) // 3]
            top.annotate(column.capitalize(), (last["threshold"], last[column]), xytext=(4, 4),
                         textcoords="offset points", color=INK, fontsize=8)
        top.set_ylim(0, 1)
        top.set_ylabel("Default class")
        top.legend(frameon=False, loc="center right")
        top.set_title(f"Threshold analysis{title_suffix}")

        bottom.plot(table["threshold"], table["cost"], color=AQUA, lw=2)
        bottom.set_ylabel("Expected cost\n(5 × FN + 1 × FP)")
        bottom.set_xlabel("Decision threshold on calibrated PD")
        for ax in (top, bottom):
            ax.axvline(chosen, color=MUTED, lw=1, ls="--")
        min_cost = table.loc[table["threshold"] == chosen, "cost"].iloc[0]
        bottom.annotate(f"chosen {chosen:.2f}\ncost {min_cost:,.0f}", (chosen, min_cost), xytext=(8, 10),
                        textcoords="offset points", color=INK, fontsize=8)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
