"""Evaluation metrics, threshold analysis and figures (SPEC §10), and the
one-time final test evaluation: ``python -m src.evaluate --final``.

Primary metric: PR-AUC (average precision). The decision threshold is chosen
on calibrated validation probabilities by minimising expected
misclassification cost (SPEC §10.2).

The final evaluation (SPEC §5.1, §10.1, §11) scores the saved artifact on the
test split exactly once. It loads the model; it never fits anything. It
refuses to run if ``test_metrics.json`` already exists, unless ``--force`` is
passed; a forced re-run is recorded in the output and must be logged in
``DECISIONS.md``. Before scoring, it checks that the data file and the test
split's row IDs are the ones the model was built against.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure  # object-oriented API: no GUI backend needed
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config

logger = logging.getLogger(__name__)

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


# --- final test evaluation (SPEC §5.1, §10.1, §11) -------------------------------------------

BOOTSTRAP_METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score, "brier": brier_score_loss}


def bootstrap_ci(y_true, scores: dict[str, np.ndarray], n_resamples: int = config.BOOTSTRAP_RESAMPLES,
                 level: float = config.CONFIDENCE_LEVEL, random_state: int = config.RANDOM_STATE) -> dict:
    """Percentile bootstrap confidence intervals for PR-AUC, ROC-AUC and Brier.

    Rows are resampled with replacement. Every score set in ``scores`` (e.g.
    raw and calibrated PDs) is evaluated on the same resamples, so the
    intervals are paired. Resamples with only one class are skipped and
    counted.
    """
    y_true = np.asarray(y_true)
    scores = {name: np.asarray(p, dtype=float) for name, p in scores.items()}
    rng = np.random.default_rng(random_state)
    draws = {name: {metric: [] for metric in BOOTSTRAP_METRICS} for name in scores}
    skipped = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        y = y_true[idx]
        if y.min() == y.max():
            skipped += 1
            continue
        for name, proba in scores.items():
            for metric, fn in BOOTSTRAP_METRICS.items():
                draws[name][metric].append(fn(y, proba[idx]))

    tail = 100 * (1 - level) / 2
    result = {"method": "percentile bootstrap, rows resampled with replacement (paired across score sets)",
              "resamples": n_resamples, "skipped_single_class": skipped, "level": level,
              "random_state": random_state}
    for name, metrics in draws.items():
        result[name] = {metric: {"lower": float(np.percentile(v, tail)), "upper": float(np.percentile(v, 100 - tail)),
                                 "std": float(np.std(v))} for metric, v in metrics.items()}
    return result


def calibration_table(y_true, proba, n_bins: int = config.CALIBRATION_BINS) -> list[dict]:
    """Reliability-diagram points: mean predicted PD vs observed default rate per equal-count bin."""
    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return [{"mean_predicted": float(p), "observed_rate": float(o)} for p, o in zip(predicted, observed)]


def risk_band_table(y_true, proba) -> list[dict]:
    """Rows, observed default rate and mean PD per risk band (SPEC §3; descriptive only)."""
    from src.explain import risk_band  # local import: explain imports this module

    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(proba)})
    frame["band"] = frame["pd"].map(risk_band)
    rows = []
    for band in ("LOW", "MEDIUM", "HIGH"):
        part = frame[frame["band"] == band]
        rows.append({"band": band, "rows": len(part), "share": len(part) / len(frame),
                     "defaults": int(part["y"].sum()),
                     "observed_rate": float(part["y"].mean()) if len(part) else None,
                     "mean_predicted": float(part["pd"].mean()) if len(part) else None})
    return rows


def plot_calibration_curves(table_raw: list[dict], table_cal: list[dict], path: Path) -> None:
    """Reliability diagram on the test split: raw vs calibrated PDs."""
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(6, 5.5))
        ax = fig.subplots()
        top = max(r[k] for t in (table_raw, table_cal) for r in t for k in ("mean_predicted", "observed_rate"))
        top = min(1.0, 1.08 * top)
        ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--")
        ax.annotate("perfect calibration", (0.62 * top, 0.62 * top), xytext=(6, -12), textcoords="offset points",
                    color=MUTED, fontsize=8)
        for table, color, label in ((table_raw, ORANGE, "Raw Random Forest"), (table_cal, BLUE, "Calibrated (deployed)")):
            ax.plot([r["mean_predicted"] for r in table], [r["observed_rate"] for r in table], color=color, lw=2,
                    marker="o", ms=5, label=label)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("Mean predicted PD (equal-count bins)")
        ax.set_ylabel("Observed default rate")
        ax.legend(frameon=False, loc="upper left")
        ax.set_title("Calibration on the test split")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)


def plot_roc_pr_curves(y_true, proba_raw, proba_cal, threshold: float, path: Path) -> Figure:
    """ROC and precision-recall curves on the test split, with the operating point at the threshold."""
    point = metrics_at_threshold(y_true, proba_cal, threshold)
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(10, 4.5))
        roc_ax, pr_ax = fig.subplots(1, 2)
        for proba, color, label in ((proba_raw, ORANGE, "Raw"), (proba_cal, BLUE, "Calibrated")):
            fpr, tpr, _ = roc_curve(y_true, proba)
            roc_ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {roc_auc_score(y_true, proba):.3f})")
            precision, recall, _ = precision_recall_curve(y_true, proba)
            pr_ax.plot(recall, precision, color=color, lw=2, drawstyle="steps-post",
                       label=f"{label} (AP {average_precision_score(y_true, proba):.3f})")
        roc_ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        fp_rate = point["fp"] / (point["fp"] + point["tn"])
        roc_ax.plot(fp_rate, point["recall"], "o", color=INK, ms=6)
        roc_ax.annotate(f"threshold {threshold:.2f}", (fp_rate, point["recall"]), xytext=(8, -12),
                        textcoords="offset points", color=INK, fontsize=8)
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("True positive rate (recall)")
        roc_ax.set_title("ROC curve (test split)")
        prevalence = float(np.mean(y_true))
        # Labelled in the legend: a text label on the line collides with the curve's end at recall 1
        pr_ax.axhline(prevalence, color=MUTED, lw=1, ls="--", label=f"No-skill baseline (default rate {prevalence:.1%})")
        pr_ax.plot(point["recall"], point["precision"], "o", color=INK, ms=6)
        pr_ax.annotate(f"threshold {threshold:.2f}", (point["recall"], point["precision"]), xytext=(8, 6),
                       textcoords="offset points", color=INK, fontsize=8)
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.set_ylim(0, 1)
        pr_ax.set_title("Precision-recall curve (test split)")
        roc_ax.legend(frameon=False, loc="lower right")
        pr_ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_confusion_matrix(at_threshold: dict, path: Path) -> None:
    """Confusion matrix at the decision threshold, with counts and row percentages."""
    cells = np.array([[at_threshold["tn"], at_threshold["fp"]], [at_threshold["fn"], at_threshold["tp"]]])
    shares = cells / np.maximum(cells.sum(axis=1, keepdims=True), 1)
    with mpl.rc_context({**STYLE, "axes.grid": False}):
        fig = Figure(figsize=(5.2, 4.4))
        ax = fig.subplots()
        cmap = mpl.colors.LinearSegmentedColormap.from_list("light_blue", [SURFACE, BLUE])
        ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(cells):
            ax.text(j, i, f"{count:,}\n{shares[i, j]:.1%} of row", ha="center", va="center", fontsize=10,
                    color=SURFACE if shares[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Not flagged", "Flagged for review"])
        ax.set_yticks([0, 1], ["No default", "Default"])
        ax.set_xlabel(f"Decision (PD ≥ {at_threshold['threshold']:.2f})")
        ax.set_ylabel("Actual outcome")
        ax.set_title("Confusion matrix (test split)")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)


def version_mismatches(recorded: dict) -> dict:
    """Installed library versions that differ from those recorded in the model metadata."""
    import importlib.metadata

    installed = {}
    for name in recorded:
        if name == "python":
            installed[name] = platform.python_version()
            continue
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    return {name: {"recorded": recorded[name], "installed": installed[name]}
            for name in recorded if recorded[name] != installed[name]}


def final_evaluation(artifact, metadata: dict, test: pd.DataFrame) -> tuple[dict, tuple]:
    """Every SPEC §10.1 / §11 test metric for the deployed (calibrated) model and its raw model.
    Returns (results, (y, raw PDs, calibrated PDs))."""
    from src.explain import unwrap_artifact  # local import: explain imports this module

    X, y = test.drop(columns=config.TARGET), test[config.TARGET]
    proba_cal = artifact.predict_proba(X)[:, 1]
    proba_raw = unwrap_artifact(artifact).predict_proba(X)[:, 1]
    results = {
        "raw": probability_metrics(y, proba_raw),
        "calibrated": probability_metrics(y, proba_cal),
        "at_threshold": metrics_at_threshold(y, proba_cal, metadata["decision_threshold"],
                                             metadata["cost_false_negative"], metadata["cost_false_positive"]),
        "bootstrap": bootstrap_ci(y, {"calibrated": proba_cal, "raw": proba_raw}),
        "calibration_curve": {"bins": config.CALIBRATION_BINS, "strategy": "quantile",
                              "raw": calibration_table(y, proba_raw), "calibrated": calibration_table(y, proba_cal)},
        "risk_bands": risk_band_table(y, proba_cal),
    }
    return results, (y, proba_raw, proba_cal)


def run_final_evaluation(split, data_sha256: str, force: bool = False) -> dict:
    """SPEC §5.1: evaluate the saved artifact on the test split, once.

    Only ``split.test`` is used. Refuses to run if the results already exist
    (unless ``force``), or if the data or the test split differ from the ones
    the model was built against. The results file is written last, so an
    interrupted run leaves no results behind.
    """
    import joblib

    from src.data_loader import index_hash

    out = config.TEST_METRICS_PATH
    previous = out.exists()
    if previous and not force:
        raise FileExistsError(f"{out.name} already exists: the test set has been evaluated. Re-running needs "
                              "--force, and the re-run must be recorded in DECISIONS.md")
    for path in (config.PIPELINE_PATH, config.METADATA_PATH, config.SPLIT_SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found; run `python -m src.train` first")

    metadata = json.loads(config.METADATA_PATH.read_text())
    recorded_split = json.loads(config.SPLIT_SUMMARY_PATH.read_text())["splits"]["test"]
    if data_sha256 != metadata["data"]["sha256"]:
        raise ValueError("The data file differs from the one the model was trained on (SHA-256 mismatch)")
    test_hash = index_hash(split.test.index)
    if test_hash != recorded_split["row_id_sha256"] or len(split.test) != recorded_split["rows"]:
        raise ValueError("The test split differs from the recorded split (row-ID hash mismatch)")
    mismatches = version_mismatches(metadata["versions"])
    for name, versions in mismatches.items():
        logger.warning("%s %s installed, artifact built with %s", name, versions["installed"], versions["recorded"])
    if force and previous:
        logger.warning("--force: overwriting %s. Record this re-run in DECISIONS.md.", out.name)

    artifact = joblib.load(config.PIPELINE_PATH)
    results, (y, proba_raw, proba_cal) = final_evaluation(artifact, metadata, split.test)
    plot_calibration_curves(results["calibration_curve"]["raw"], results["calibration_curve"]["calibrated"],
                            config.TEST_CALIBRATION_FIGURE)
    plot_roc_pr_curves(y, proba_raw, proba_cal, metadata["decision_threshold"], config.TEST_ROC_PR_FIGURE)
    plot_confusion_matrix(results["at_threshold"], config.TEST_CONFUSION_FIGURE)

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m src.evaluate --final" + (" --force" if force else ""),
        "forced_rerun": bool(force and previous),
        "artifact": config.PIPELINE_PATH.name,
        "model": metadata["model"]["type"],
        "model_configuration": metadata["model"]["configuration"],
        "calibration_method": metadata["calibration_method"],
        "decision_threshold": metadata["decision_threshold"],
        "cost_false_negative": metadata["cost_false_negative"],
        "cost_false_positive": metadata["cost_false_positive"],
        "data_sha256": data_sha256,
        "test_split": {"rows": len(split.test), "defaults": int(y.sum()), "default_rate": float(y.mean()),
                       "row_id_sha256": test_hash, "matches_recorded_split": True},
        **results,
        "validation_reference": metadata["validation_metrics"],
        "version_mismatches": mismatches,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Written %s and 3 figures", out.name)
    return report


def main(argv: list[str] | None = None) -> None:
    from src.data_loader import file_sha256, load_raw_data, split_data

    parser = argparse.ArgumentParser(description="One-time final evaluation on the test split (SPEC §5.1).")
    parser.add_argument("--final", action="store_true", help="evaluate the saved artifact on the test split, once")
    parser.add_argument("--force", action="store_true",
                        help="re-run even though test results exist (record the re-run in DECISIONS.md)")
    args = parser.parse_args(argv)
    if not args.final:
        parser.error("only --final is available; validation reports are produced by `python -m src.train`")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if config.TEST_METRICS_PATH.exists() and not args.force:  # refuse before even loading the data
        raise SystemExit(f"{config.TEST_METRICS_PATH.name} already exists: the test set has been evaluated. "
                         "Use --force to re-run, and record the re-run in DECISIONS.md.")
    report = run_final_evaluation(split_data(load_raw_data()), file_sha256(config.RAW_DATA_FILE), force=args.force)

    c, ci, t = report["calibrated"], report["bootstrap"]["calibrated"], report["at_threshold"]
    logger.info("Test PR-AUC %.4f [%.4f, %.4f] | ROC-AUC %.4f [%.4f, %.4f] | Brier %.5f [%.5f, %.5f] (raw %.5f)",
                c["pr_auc"], ci["pr_auc"]["lower"], ci["pr_auc"]["upper"], c["roc_auc"], ci["roc_auc"]["lower"],
                ci["roc_auc"]["upper"], c["brier"], ci["brier"]["lower"], ci["brier"]["upper"], report["raw"]["brier"])
    logger.info("Threshold %.2f: precision %.3f, recall %.3f, F1 %.3f, TP %d FP %d FN %d TN %d",
                t["threshold"], t["precision"], t["recall"], t["f1"], t["tp"], t["fp"], t["fn"], t["tn"])


if __name__ == "__main__":
    main()
