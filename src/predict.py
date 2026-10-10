"""Prediction for the app (SPEC §14.2, §15.1): load the artifact, validate raw
inputs, and return the calibrated PD, risk band, review flag and local SHAP
explanation.

Nothing here trains a model or reads the dataset: the only inputs are the saved
artifact, its metadata and the values a user enters. Streamlit is not imported,
so everything is testable with plain pytest.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.evaluate import version_mismatches
from src.explain import LocalExplanation, ModelExplainer, explain_row, make_explainer

logger = logging.getLogger(__name__)

MIN_AGE, MAX_AGE = 18, 110  # SPEC §14.2 hard limits
UNRELIABLE = "the prediction may be less reliable"


@dataclass(frozen=True)
class InputSpec:
    """How one raw feature is entered in the app."""

    integer: bool
    default: float
    step: float
    help: str
    min_value: float = 0.0
    max_value: float | None = None
    allow_unknown: bool = False  # SPEC §14.2: income and dependents can be "Unknown" (NaN)


# One entry per raw feature. The app builds its form from the metadata's feature
# list and looks the specs up here; a test checks both stay in sync.
# Defaults describe an invented, typical borrower (not a dataset row).
INPUT_SPECS = {
    "RevolvingUtilizationOfUnsecuredLines": InputSpec(
        integer=False, default=0.30, step=0.05,
        help="Balance on credit cards and personal lines of credit divided by their combined limit "
             "(0.30 = 30%). Values above 1 mean the borrower is over the limit; values above 10 are "
             "treated as unknown."),
    "age": InputSpec(integer=True, default=45, step=1, min_value=MIN_AGE, max_value=MAX_AGE,
                     help="Age of the borrower in years (18-110)."),
    "NumberOfTime30-59DaysPastDueNotWorse": InputSpec(
        integer=True, default=0, step=1, max_value=89,
        help="Times the borrower was 30-59 days late in the last 2 years (but not worse)."),
    "DebtRatio": InputSpec(
        integer=False, default=0.35, step=0.05,
        help="Monthly debt payments, alimony and living costs divided by monthly gross income. "
             "Ignored by the model when income is unknown."),
    "MonthlyIncome": InputSpec(
        integer=False, default=5000, step=100, allow_unknown=True,
        help="Monthly gross income. An income of 0 is treated as unknown, as in training."),
    "NumberOfOpenCreditLinesAndLoans": InputSpec(
        integer=True, default=8, step=1,
        help="Open loans (e.g. car loan, mortgage) and lines of credit (e.g. credit cards)."),
    "NumberOfTimes90DaysLate": InputSpec(
        integer=True, default=0, step=1, max_value=89,
        help="Times the borrower was 90 days or more late."),
    "NumberRealEstateLoansOrLines": InputSpec(
        integer=True, default=1, step=1,
        help="Mortgage and real-estate loans, including home-equity lines of credit."),
    "NumberOfTime60-89DaysPastDueNotWorse": InputSpec(
        integer=True, default=0, step=1, max_value=89,
        help="Times the borrower was 60-89 days late in the last 2 years (but not worse)."),
    "NumberOfDependents": InputSpec(
        integer=True, default=0, step=1, allow_unknown=True,
        help="Dependents in the household, excluding the borrower (spouse, children, etc.)."),
}
# Past-due counts are capped at 89: values of 90 and above are dataset codes (96/98), not counts.


@dataclass
class Prediction:
    pd: float
    risk_band: str
    flagged: bool
    threshold: float
    explanation: LocalExplanation
    warnings: list[str] = field(default_factory=list)


@dataclass
class Predictor:
    """The loaded artifact with everything needed to score and explain one borrower."""

    artifact: object
    metadata: dict
    explainer: ModelExplainer
    version_warnings: list[str]

    @property
    def raw_features(self) -> list[str]:
        return list(self.metadata["raw_input_features"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["decision_threshold"])

    def predict(self, values: dict[str, float]) -> Prediction:
        errors = validate_inputs(values, self.raw_features)
        if errors:
            raise ValueError("; ".join(errors))
        row = to_frame(values, self.raw_features)
        explanation = explain_row(self.explainer, self.artifact, row)
        return Prediction(pd=explanation.pd, risk_band=explanation.risk_band,
                          flagged=bool(explanation.pd >= self.threshold), threshold=self.threshold,
                          explanation=explanation,
                          warnings=input_warnings(values, self.metadata["training_percentiles"]))


def load_predictor(pipeline_path: Path | None = None, metadata_path: Path | None = None) -> Predictor:
    """Load the artifact and metadata; warn if library versions differ from training (SPEC §15.1)."""
    pipeline_path = pipeline_path or config.PIPELINE_PATH
    metadata_path = metadata_path or config.METADATA_PATH
    for path in (pipeline_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found; run `python -m src.train` first")
    metadata = json.loads(metadata_path.read_text())
    version_warnings = [f"{name} {v['installed']} is installed, but the model was built with {v['recorded']}"
                        for name, v in version_mismatches(metadata["versions"]).items()]
    for message in version_warnings:
        logger.warning(message)
    artifact = joblib.load(pipeline_path)
    return Predictor(artifact, metadata, make_explainer(artifact), version_warnings)


def is_unknown(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def validate_inputs(values: dict, features: list[str]) -> list[str]:
    """Hard limits (SPEC §14.2): every feature present, numeric, not negative; age 18-110;
    only income and dependents may be unknown."""
    errors = []
    missing = [f for f in features if f not in values]
    extra = [f for f in values if f not in features]
    if missing:
        errors.append(f"missing inputs: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected inputs: {', '.join(extra)}")
    for name in features:
        if name not in values:
            continue
        label = config.FEATURE_LABELS[name]
        value = values[name]
        if is_unknown(value):
            if not INPUT_SPECS[name].allow_unknown:
                errors.append(f"{label} is required")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number")
            continue
        if not math.isfinite(value):
            errors.append(f"{label} must be a finite number")
        elif value < 0:
            errors.append(f"{label} cannot be negative")
        elif name == "age" and not MIN_AGE <= value <= MAX_AGE:
            errors.append(f"Age must be between {MIN_AGE} and {MAX_AGE}")
    return errors


def input_warnings(values: dict, percentiles: dict) -> list[str]:
    """SPEC §14.2: warn when an input lies outside the training 1st-99th percentile range,
    plus notes on inputs the model treats as unknown."""
    warnings = []
    for name, bounds in percentiles.items():
        value = values.get(name)
        if is_unknown(value) or cleaned_to_unknown(name, float(value)):
            continue  # the model sees "unknown", so the range check does not apply
        if not bounds["p01"] <= float(value) <= bounds["p99"]:
            warnings.append(f"{config.FEATURE_LABELS[name]} ({value:,.6g}) is outside the training data's "
                            f"1st-99th percentile range ({bounds['p01']:,.6g} to {bounds['p99']:,.6g}): "
                            f"{UNRELIABLE}.")
    income = values.get("MonthlyIncome")
    if is_unknown(income) or income == 0:
        warnings.append("Monthly income is unknown (or 0), so the model ignores the debt ratio, as in training.")
    utilization = values.get("RevolvingUtilizationOfUnsecuredLines")
    if not is_unknown(utilization) and float(utilization) > config.MAX_VALID_UTILIZATION:
        warnings.append(f"Revolving credit utilization above {config.MAX_VALID_UTILIZATION:g} is treated as "
                        "unknown, as in training.")
    return warnings


def cleaned_to_unknown(name: str, value: float) -> bool:
    """Inputs the cleaning step turns into missing values (SPEC §7.2; see CleaningTransformer)."""
    if name == "MonthlyIncome":
        return value == 0
    if name == "RevolvingUtilizationOfUnsecuredLines":
        return value > config.MAX_VALID_UTILIZATION
    if name in config.PAST_DUE_COLUMNS:
        return value >= config.PAST_DUE_CODE_MIN
    return name == "age" and value < config.MIN_VALID_AGE


def to_frame(values: dict, features: list[str]) -> pd.DataFrame:
    """One row of raw inputs in the artifact's expected column order; unknown values become NaN."""
    row = {name: np.nan if is_unknown(values[name]) else float(values[name]) for name in features}
    return pd.DataFrame([row], columns=features)
