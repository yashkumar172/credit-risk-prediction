"""Cleaning rules and engineered features as scikit-learn transformers (SPEC §7).

Both transformers are stateless: ``fit`` learns nothing from the data, it only
records the input columns. They are the first steps of every model pipeline,
so training, evaluation and the app apply identical logic.

    raw input -> CleaningTransformer -> FeatureEngineer -> model-specific preprocessing -> model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src import config


def _require_columns(X: pd.DataFrame, columns: list[str], step: str) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{step} expects a pandas DataFrame, got {type(X).__name__}")
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"{step}: missing required columns {missing}")


class CleaningTransformer(TransformerMixin, BaseEstimator):
    """Apply the stateless cleaning rules and keep only the 10 raw features.

    Rules (each turns an invalid value into NaN; no rows are removed):

    1. Past-due counts >= 90 (the 96/98 codes) -> NaN
    2. ``age`` < 18 -> NaN
    3. ``MonthlyIncome`` == 0 -> NaN (the debt ratio was never computed)
    4. ``DebtRatio`` -> NaN wherever income is missing, after rule 3
       (there it holds a debt amount, not a ratio)
    5. Utilization > 10 -> NaN (amount-like values, not ratios)

    Infinite values are also turned into NaN. Any other columns, such as the
    target or a row ID, are dropped, so they can never reach the model.
    """

    def fit(self, X: pd.DataFrame, y=None) -> CleaningTransformer:
        _require_columns(X, config.RAW_FEATURES, "CleaningTransformer")
        self.feature_names_in_ = np.asarray(config.RAW_FEATURES, dtype=object)
        self.n_features_in_ = len(config.RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        _require_columns(X, config.RAW_FEATURES, "CleaningTransformer")
        out = X[config.RAW_FEATURES].astype(float)  # new frame; the input is never modified
        out = out.replace([np.inf, -np.inf], np.nan)

        past_due = out[config.PAST_DUE_COLUMNS]
        out[config.PAST_DUE_COLUMNS] = past_due.mask(past_due >= config.PAST_DUE_CODE_MIN)
        out["age"] = out["age"].mask(out["age"] < config.MIN_VALID_AGE)
        out["MonthlyIncome"] = out["MonthlyIncome"].mask(out["MonthlyIncome"] == 0)
        out["DebtRatio"] = out["DebtRatio"].mask(out["MonthlyIncome"].isna())
        utilization = out["RevolvingUtilizationOfUnsecuredLines"]
        out["RevolvingUtilizationOfUnsecuredLines"] = utilization.mask(
            utilization > config.MAX_VALID_UTILIZATION
        )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(config.RAW_FEATURES, dtype=object)


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Add the engineered features to the cleaned raw features.

    - ``total_past_due``: sum of the three past-due counts (NaN if any is missing)
    - ``income_per_dependent``: income / (dependents + 1), so zero dependents is safe
    - ``utilization_over_limit``: 1.0 if utilization > 1, 0.0 if not, NaN if unknown
    """

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        _require_columns(X, config.RAW_FEATURES, "FeatureEngineer")
        self.feature_names_in_ = np.asarray(config.RAW_FEATURES, dtype=object)
        self.n_features_in_ = len(config.RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        _require_columns(X, config.RAW_FEATURES, "FeatureEngineer")
        out = X[config.RAW_FEATURES].copy()

        out["total_past_due"] = out[config.PAST_DUE_COLUMNS].sum(axis=1, skipna=False)
        out["income_per_dependent"] = out["MonthlyIncome"] / (out["NumberOfDependents"] + 1)
        utilization = out["RevolvingUtilizationOfUnsecuredLines"]
        out["utilization_over_limit"] = (utilization > config.UTILIZATION_LIMIT).astype(float).mask(
            utilization.isna()
        )
        return out[config.MODEL_FEATURES]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(config.MODEL_FEATURES, dtype=object)


def make_feature_pipeline() -> Pipeline:
    """Cleaning followed by feature engineering: the shared first steps of every model pipeline."""
    return Pipeline([("clean", CleaningTransformer()), ("features", FeatureEngineer())])
