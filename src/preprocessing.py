"""Model-specific preprocessing (SPEC §8) and the full per-model pipeline.

    Logistic Regression: median imputation + missing indicators -> winsorizing (1st/99th
                         percentile) -> log1p for skewed features -> StandardScaler
    Random Forest:       none (native NaN support, scikit-learn >= 1.4)
    XGBoost:             none (native NaN handling)

Everything here is fitted inside the pipeline, so it only ever learns from
the data passed to ``fit``: the training split, or the training folds in CV.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from src import config
from src.feature_engineering import CleaningTransformer, FeatureEngineer


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Winsorize each column to quantiles learned in ``fit`` (NaN values are ignored)."""

    def __init__(self, lower: float = config.WINSOR_LOWER_QUANTILE, upper: float = config.WINSOR_UPPER_QUANTILE):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None) -> QuantileClipper:
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features, dtype=object)


def make_logistic_preprocessor() -> ColumnTransformer:
    """Imputation, missing indicators, winsorizing, log1p and scaling for Logistic Regression."""
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", QuantileClipper()),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    unskewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", QuantileClipper()),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    # Indicators get their own branch: winsorizing a rare 0/1 column at the 99th
    # percentile would clip every 1 to 0.
    missing = MissingIndicator(features="all")

    return ColumnTransformer(
        [
            ("skewed", skewed, config.SKEWED_FEATURES),
            ("unskewed", unskewed, config.UNSKEWED_FEATURES),
            ("binary", binary, config.BINARY_FEATURES),
            ("missing", missing, config.MISSING_INDICATOR_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_preprocessor(model_name: str):
    """Return the preprocessing step for a model: a ColumnTransformer, or 'passthrough'."""
    if model_name == "logistic_regression":
        return make_logistic_preprocessor()
    if model_name in ("random_forest", "xgboost"):
        return "passthrough"
    raise ValueError(f"Unknown model '{model_name}'. Expected one of {config.MODEL_NAMES}")


def make_model_pipeline(model_name: str, estimator) -> Pipeline:
    """Full pipeline from raw input to model: cleaning -> features -> preprocessing -> estimator."""
    return Pipeline([
        ("clean", CleaningTransformer()),
        ("features", FeatureEngineer()),
        ("preprocess", make_preprocessor(model_name)),
        ("model", estimator),
    ])
