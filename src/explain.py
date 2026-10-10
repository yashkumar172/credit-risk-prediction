"""SHAP explanations (SPEC §13): ``python -m src.explain``.

SHAP explains the **underlying uncalibrated model**, not the calibrated PD.
``unwrap_artifact`` takes the train-fitted pipeline back out of the
calibrated wrapper; its preprocessing steps produce the model's input, which
the explainer works on.

Units depend on the model's own output:

* Random Forest: a probability (the trees' averaged class shares), so SHAP
  values are in **probability units**, shown as percentage points. A forest
  has no log-odds output to explain.
* XGBoost and Logistic Regression: **log-odds** (margin / linear predictor).

In both cases base value + sum of SHAP values = the raw model output exactly.
The calibrator is a monotonic transform applied afterwards, so it keeps the
ranking and direction of feature effects. The contributions do not add up to
the calibrated PD.

The global explanation uses a stratified sample of validation rows (SPEC
§13.2) and never touches the test split. Local explanations are computed
one row at a time (SPEC §13.3).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass

import matplotlib as mpl

mpl.use("Agg")  # files only; SHAP's beeswarm draws through pyplot

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src import config  # noqa: E402
from src.evaluate import BLUE, INK, MUTED, ORANGE, STYLE  # noqa: E402

logger = logging.getLogger(__name__)

PROBABILITY, LOG_ODDS = "probability", "log-odds"
MISSING_PREFIX = "missingindicator_"
ADDITIVITY_TOLERANCE = 1e-6

SHAP_NOTES = {
    PROBABILITY: ("SHAP values explain the uncalibrated model's predicted probability, in percentage points. "
                  "The calibrator is applied afterwards, so the contributions do not add up exactly to the "
                  "calibrated PD shown."),
    LOG_ODDS: ("SHAP values explain the uncalibrated model's output in log-odds. The calibrator is applied "
               "afterwards, so the contributions do not add up exactly to the calibrated PD shown."),
}


# --- unwrapping and the explainer ---------------------------------------------------------


def unwrap_artifact(artifact) -> Pipeline:
    """The train-fitted pipeline (clean → features → preprocess → model) inside the calibrated wrapper."""
    if isinstance(artifact, Pipeline):
        return artifact
    if not isinstance(artifact, CalibratedClassifierCV):
        raise TypeError(f"Expected a CalibratedClassifierCV or Pipeline, got {type(artifact).__name__}")
    if len(artifact.calibrated_classifiers_) != 1:
        raise ValueError("Expected one calibrated classifier (a frozen pipeline calibrated on validation data); "
                         f"found {len(artifact.calibrated_classifiers_)}")
    inner = artifact.calibrated_classifiers_[0].estimator
    return inner.estimator if isinstance(inner, FrozenEstimator) else inner


@dataclass
class ModelExplainer:
    """SHAP explainer for the model inside a pipeline, plus the steps that produce its input."""

    pipeline: Pipeline
    explainer: object
    units: str

    @property
    def model(self):
        return self.pipeline[-1]

    @property
    def feature_names(self) -> list[str]:
        return [str(f) for f in self.pipeline[:-1].get_feature_names_out()]

    @property
    def base_value(self) -> float:
        base = np.atleast_1d(self.explainer.expected_value)
        return float(base[-1])  # the default class when SHAP returns one value per class

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Model input for raw rows: the output of every step before the model."""
        Z = self.pipeline[:-1].transform(X)
        return Z if isinstance(Z, pd.DataFrame) else pd.DataFrame(Z, columns=self.feature_names, index=X.index)

    def shap_values(self, Z: pd.DataFrame) -> np.ndarray:
        """SHAP values of the default class, shape (rows, features)."""
        values = np.asarray(self.explainer.shap_values(Z))
        return values[..., 1] if values.ndim == 3 else values

    def raw_output(self, Z: pd.DataFrame) -> np.ndarray:
        """The model output SHAP explains: base value + sum of SHAP values equals this."""
        if self.units == PROBABILITY:
            return self.model.predict_proba(Z)[:, 1]
        if isinstance(self.model, LogisticRegression):
            return self.model.decision_function(Z.to_numpy())  # fitted on the ColumnTransformer's array
        return self.model.predict(Z, output_margin=True)


def make_explainer(artifact, background: pd.DataFrame | None = None) -> ModelExplainer:
    """``TreeExplainer`` for Random Forest / XGBoost, ``LinearExplainer`` for Logistic Regression.

    ``background`` (raw rows) is needed only for Logistic Regression; tree
    explainers use the training statistics stored in the trees.
    """
    pipeline = unwrap_artifact(artifact)
    model = pipeline[-1]
    if isinstance(model, RandomForestClassifier):
        return ModelExplainer(pipeline, shap.TreeExplainer(model), PROBABILITY)
    if isinstance(model, XGBClassifier):
        return ModelExplainer(pipeline, shap.TreeExplainer(model), LOG_ODDS)
    if isinstance(model, LogisticRegression):
        if background is None:
            raise ValueError("Logistic Regression needs background rows for LinearExplainer")
        partial = ModelExplainer(pipeline, None, LOG_ODDS)
        partial.explainer = shap.LinearExplainer(model, partial.transform(background))
        return partial
    raise TypeError(f"No SHAP explainer configured for {type(model).__name__}")


# --- labels and formatting ----------------------------------------------------------------


def base_feature(name: str) -> str:
    return name.removeprefix(MISSING_PREFIX)


def feature_label(name: str) -> str:
    label = config.FEATURE_LABELS.get(base_feature(name), base_feature(name))
    return f"{label} unknown" if name.startswith(MISSING_PREFIX) else label


def format_value(name: str, value) -> str:
    """Input value as shown to the user. Only the value itself, no qualitative wording."""
    if pd.isna(value):
        return "Unknown"
    if name.startswith(MISSING_PREFIX) or name in config.BINARY_FEATURES:
        return "Yes" if value else "No"
    value = float(value)
    if value.is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"


def format_contribution(contribution: float, units: str) -> str:
    return f"{100 * contribution:+.1f} pp" if units == PROBABILITY else f"{contribution:+.2f} log-odds"


def risk_band(pd_value: float) -> str:
    """SPEC §3 risk bands, from the boundaries in config."""
    if pd_value < config.RISK_BAND_LOW_UPPER:
        return "LOW"
    return "MEDIUM" if pd_value < config.RISK_BAND_MEDIUM_UPPER else "HIGH"


# --- local explanation (SPEC §13.3) -------------------------------------------------------


@dataclass
class Factor:
    feature: str
    label: str
    value: float
    display_value: str
    contribution: float


@dataclass
class LocalExplanation:
    pd: float                 # calibrated PD (what the user sees)
    risk_band: str
    raw_output: float         # what SHAP explains: base_value + sum of contributions
    base_value: float
    units: str
    factors: list[Factor]     # every feature, most risk-increasing first
    top_n: int = config.SHAP_TOP_N

    @property
    def increasing(self) -> list[Factor]:
        return [f for f in self.factors if f.contribution > 0][:self.top_n]

    @property
    def decreasing(self) -> list[Factor]:
        return [f for f in reversed(self.factors) if f.contribution < 0][:self.top_n]

    @property
    def note(self) -> str:
        return SHAP_NOTES[self.units]


def explain_row(model_explainer: ModelExplainer, artifact, row: pd.DataFrame,
                top_n: int = config.SHAP_TOP_N) -> LocalExplanation:
    """Calibrated PD, risk band and SHAP factors for one row of raw inputs."""
    if len(row) != 1:
        raise ValueError(f"explain_row takes exactly one row, got {len(row)}")
    pd_value = float(artifact.predict_proba(row)[0, 1])
    Z = model_explainer.transform(row)
    contributions = model_explainer.shap_values(Z)[0]
    # Show values before scaling: the output of the clean and features steps (equal to Z for tree models)
    readable = model_explainer.pipeline[:2].transform(row).iloc[0]

    factors = []
    for name, contribution in zip(model_explainer.feature_names, contributions):
        value = Z.iloc[0][name] if name.startswith(MISSING_PREFIX) else readable[base_feature(name)]
        factors.append(Factor(name, feature_label(name), float(value), format_value(name, value), float(contribution)))
    factors.sort(key=lambda f: f.contribution, reverse=True)
    return LocalExplanation(pd=pd_value, risk_band=risk_band(pd_value),
                            raw_output=float(model_explainer.raw_output(Z)[0]),
                            base_value=model_explainer.base_value, units=model_explainer.units,
                            factors=factors, top_n=top_n)


def format_local_explanation(local: LocalExplanation) -> str:
    """Plain-text explanation in the SPEC §13.3 layout. Built only from SHAP values and inputs."""
    def lines(factors):
        if not factors:
            return ["- none"]
        width = max(len(f"{f.label}: {f.display_value}") for f in factors)
        return [f"- {f'{f.label}: {f.display_value}':<{width}}   ({format_contribution(f.contribution, local.units)})"
                for f in factors]

    return "\n".join([
        f"Probability of Default: {100 * local.pd:.1f}%   (Risk band: {local.risk_band})",
        "",
        "Factors increasing risk:", *lines(local.increasing),
        "",
        "Factors reducing risk:", *lines(local.decreasing),
        "",
        local.note,
    ])


def plot_local_explanation(local: LocalExplanation, max_features: int = 10) -> Figure:
    """Bar chart of the largest SHAP contributions for one prediction."""
    shown = sorted(local.factors, key=lambda f: abs(f.contribution), reverse=True)[:max_features][::-1]
    scale = 100 if local.units == PROBABILITY else 1
    unit = "percentage points of raw PD" if local.units == PROBABILITY else "log-odds"
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(7.5, 0.4 * len(shown) + 1.6))
        ax = fig.subplots()
        values = [scale * f.contribution for f in shown]
        ax.barh(range(len(shown)), values, color=[ORANGE if v > 0 else BLUE for v in values], height=0.65)
        ax.set_yticks(range(len(shown)), [f"{f.label} = {f.display_value}" for f in shown])
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(values):
            ax.annotate(f"{v:+.1f}" if local.units == PROBABILITY else f"{v:+.2f}", (v, i),
                        xytext=(4 if v > 0 else -4, 0), textcoords="offset points",
                        ha="left" if v > 0 else "right", va="center", color=INK, fontsize=8)
        ax.margins(x=0.15)
        ax.set_xlabel(f"SHAP contribution ({unit})\norange raises risk, blue lowers it")
        ax.set_title(f"Calibrated PD {100 * local.pd:.1f}% (risk band {local.risk_band})")
        fig.tight_layout()
    return fig


# --- global explanation (SPEC §13.2) ------------------------------------------------------


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int = config.SHAP_SAMPLE_SIZE,
                      random_state: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    """``n`` rows stratified by the target with a fixed seed (all rows if there are fewer)."""
    if n >= len(X):
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=random_state)
    return X_s, y_s


def global_explanation(model_explainer: ModelExplainer, X: pd.DataFrame, y: pd.Series,
                       n: int = config.SHAP_SAMPLE_SIZE) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """SHAP values on a stratified sample; returns (report, SHAP values, model input of the sample)."""
    X_s, y_s = stratified_sample(X, y, n)
    Z = model_explainer.transform(X_s)
    start = time.perf_counter()
    values = model_explainer.shap_values(Z)
    elapsed = time.perf_counter() - start

    additivity_error = float(np.max(np.abs(model_explainer.base_value + values.sum(axis=1)
                                           - model_explainer.raw_output(Z))))
    if additivity_error > ADDITIVITY_TOLERANCE:
        raise ValueError(f"SHAP values do not add up to the model output (max error {additivity_error:.2e})")

    features = []
    for i, name in enumerate(model_explainer.feature_names):
        known = Z[name].notna().to_numpy()
        varies = known.sum() > 2 and np.ptp(Z[name].to_numpy()[known]) > 0
        direction = float(spearmanr(Z[name].to_numpy()[known], values[known, i]).statistic) if varies else None
        features.append({
            "feature": name,
            "label": feature_label(name),
            "mean_abs_shap": float(np.abs(values[:, i]).mean()),
            "mean_shap": float(values[:, i].mean()),
            "value_shap_spearman": direction,  # > 0: higher values push the PD up
        })
    features.sort(key=lambda f: f["mean_abs_shap"], reverse=True)
    for rank, f in enumerate(features, 1):
        f["rank"] = rank

    report = {
        "explains": "the uncalibrated model inside the calibrated artifact",
        "explainer": type(model_explainer.explainer).__name__,
        "model": type(model_explainer.model).__name__,
        "units": model_explainer.units,
        "note": SHAP_NOTES[model_explainer.units],
        "sample": {
            "source": "validation split",
            "rows": len(X_s),
            "defaults": int(y_s.sum()),
            "default_rate": float(y_s.mean()),
            "method": "stratified by target (train_test_split)",
            "random_state": config.RANDOM_STATE,
            "row_ids_sha256": hashlib.sha256(np.sort(X_s.index.to_numpy()).tobytes()).hexdigest(),
        },
        "base_value": model_explainer.base_value,
        "mean_raw_output": float(model_explainer.raw_output(Z).mean()),
        "max_additivity_error": additivity_error,
        "shap_seconds": round(elapsed, 1),
        "top_features": [f["label"] for f in features[:5]],
        "features": features,
    }
    return report, values, Z


def plot_beeswarm(values: np.ndarray, Z: pd.DataFrame, units: str, path) -> None:
    """SHAP summary (beeswarm) plot with human-readable labels."""
    scale = 100 if units == PROBABILITY else 1
    explanation = shap.Explanation(values=scale * values, data=Z.to_numpy(dtype=float),
                                   feature_names=[feature_label(c) for c in Z.columns])
    with mpl.rc_context(STYLE):
        plt.figure()
        shap.plots.beeswarm(explanation, max_display=len(Z.columns), show=False, plot_size=(8, 6))
        fig = plt.gcf()
        fig.axes[0].set_xlabel("SHAP value (percentage points of raw PD)" if units == PROBABILITY
                               else "SHAP value (log-odds)")
        fig.axes[0].set_title(f"SHAP summary: {len(Z):,} stratified validation rows")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def plot_mean_abs_bar(report: dict, path) -> None:
    """Mean |SHAP| per feature: global importance."""
    scale = 100 if report["units"] == PROBABILITY else 1
    features = report["features"][::-1]
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(7.5, 0.35 * len(features) + 1.4))
        ax = fig.subplots()
        values = [scale * f["mean_abs_shap"] for f in features]
        ax.barh([f["label"] for f in features], values, color=BLUE, height=0.65)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=8)
        ax.margins(x=0.12)
        ax.set_xlabel("Mean |SHAP| (percentage points of raw PD)" if report["units"] == PROBABILITY
                      else "Mean |SHAP| (log-odds)")
        ax.set_title(f"Feature importance: mean |SHAP|, {report['sample']['rows']:,} validation rows")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)


def example_borrower() -> pd.DataFrame:
    """The invented borrower used for the committed local example, as one row of raw inputs."""
    return pd.DataFrame([config.SHAP_EXAMPLE_BORROWER])[config.RAW_FEATURES]


def local_example(model_explainer: ModelExplainer, artifact) -> tuple[dict, Figure]:
    """Local explanation of the invented example borrower: report entry and figure."""
    local = explain_row(model_explainer, artifact, example_borrower())
    entry = {
        "source": "synthetic borrower (config.SHAP_EXAMPLE_BORROWER), not a dataset row",
        "inputs": config.SHAP_EXAMPLE_BORROWER,
        "text": format_local_explanation(local),
    }
    return entry, plot_local_explanation(local)


def run_global_explanation(artifact, split) -> dict:
    """Global SHAP report and figures from the validation split only, plus the local
    example for the invented borrower (no dataset row is written to the reports)."""
    from src.train import split_xy  # local import: train imports this module

    X_val, y_val = split_xy(split.validation)
    background = None  # only LinearExplainer needs one; training rows, as the model saw them
    if isinstance(unwrap_artifact(artifact)[-1], LogisticRegression):
        background = stratified_sample(*split_xy(split.train))[0]
    model_explainer = make_explainer(artifact, background=background)
    logger.info("Global SHAP: %s on %d stratified validation rows...", type(model_explainer.explainer).__name__,
                min(len(X_val), config.SHAP_SAMPLE_SIZE))
    report, values, Z = global_explanation(model_explainer, X_val, y_val)
    plot_beeswarm(values, Z, model_explainer.units, config.SHAP_BEESWARM_FIGURE)
    plot_mean_abs_bar(report, config.SHAP_BAR_FIGURE)

    report["local_example"], fig = local_example(model_explainer, artifact)
    config.SHAP_LOCAL_EXAMPLE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.SHAP_LOCAL_EXAMPLE_FIGURE)

    config.SHAP_GLOBAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SHAP_GLOBAL_PATH.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("SHAP %.0fs; top features: %s", report["shap_seconds"], ", ".join(report["top_features"]))
    logger.info("Written %s and 3 figures", config.SHAP_GLOBAL_PATH.name)
    return report


def main() -> None:
    """Global SHAP for the saved artifact; no training, validation split only."""
    from src.data_loader import load_raw_data, split_data

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not config.PIPELINE_PATH.exists():
        raise FileNotFoundError(f"{config.PIPELINE_PATH.name} not found; run `python -m src.train` first")
    artifact = joblib.load(config.PIPELINE_PATH)
    run_global_explanation(artifact, split_data(load_raw_data()))


if __name__ == "__main__":
    main()
