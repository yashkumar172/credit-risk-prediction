"""Probability calibration (SPEC §11).

Calibrators are fitted on the validation split only, wrapping a pipeline that
was fitted on the training split. ``FrozenEstimator`` keeps the wrapped
pipeline from being refitted (scikit-learn >= 1.6; ``cv="prefit"`` is deprecated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.evaluate import probability_metrics


def out_of_fold_calibrated_proba(fitted_pipeline, X_val: pd.DataFrame, y_val: pd.Series,
                                 method: str = "sigmoid", n_folds: int = config.CV_FOLDS) -> np.ndarray:
    """Calibrated validation probabilities where no row is scored by a calibrator that saw it.

    The validation split is cut into ``n_folds`` stratified folds; a calibrator is
    fitted on all but one fold and predicts the held-out fold. The wrapped
    pipeline itself is frozen (it was fitted on the training split).
    """
    calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method=method)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    return cross_val_predict(calibrator, X_val, y_val, cv=cv, method="predict_proba")[:, 1]


def calibrated_brier_cv(fitted_pipeline, X_val: pd.DataFrame, y_val: pd.Series, method: str = "sigmoid") -> float:
    """Brier score of out-of-fold calibrated validation probabilities (used for the SPEC §12 tie rule)."""
    proba = out_of_fold_calibrated_proba(fitted_pipeline, X_val, y_val, method=method)
    return float(brier_score_loss(y_val, proba))


CALIBRATION_METHODS = ("sigmoid", "isotonic")
# SPEC §11: prefer sigmoid unless isotonic is clearly better. "Clearly" = at least
# 1% lower out-of-fold Brier score (rule fixed before Phase 9 results were seen).
ISOTONIC_MIN_RELATIVE_GAIN = 0.01


def compare_calibration_methods(fitted_pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Raw vs out-of-fold calibrated metrics on the validation split, for each method."""
    comparison = {"raw": probability_metrics(y_val, fitted_pipeline.predict_proba(X_val)[:, 1])}
    for method in CALIBRATION_METHODS:
        proba = out_of_fold_calibrated_proba(fitted_pipeline, X_val, y_val, method=method)
        comparison[method] = probability_metrics(y_val, proba)
    return comparison


def choose_calibration_method(comparison: dict) -> tuple[str, str]:
    """Apply the SPEC §11 preference for sigmoid; returns (method, reason)."""
    sigmoid, isotonic = comparison["sigmoid"]["brier"], comparison["isotonic"]["brier"]
    gain = (sigmoid - isotonic) / sigmoid
    if gain >= ISOTONIC_MIN_RELATIVE_GAIN:
        return "isotonic", (f"isotonic out-of-fold Brier {isotonic:.5f} is {gain:.1%} lower than sigmoid "
                            f"{sigmoid:.5f} (threshold {ISOTONIC_MIN_RELATIVE_GAIN:.0%})")
    return "sigmoid", (f"isotonic is not clearly better: Brier {isotonic:.5f} vs sigmoid {sigmoid:.5f} "
                       f"({gain:+.1%}; needs at least {ISOTONIC_MIN_RELATIVE_GAIN:.0%})")


def fit_calibrator(fitted_pipeline, X_val: pd.DataFrame, y_val: pd.Series, method: str) -> CalibratedClassifierCV:
    """Final calibrator: one calibrator fitted on the whole validation split around the
    frozen, train-fitted pipeline. This is the deployed model."""
    return CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method=method).fit(X_val, y_val)
