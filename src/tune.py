"""Hyperparameter tuning of the selected model (SPEC §9).

RandomizedSearchCV with stratified 5-fold CV on the training split only,
scored by PR-AUC (average precision). The whole pipeline is refitted in every
fold. Validation and test data are never used here.
"""

from __future__ import annotations

import time

import pandas as pd
import sklearn
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src import config
from src.preprocessing import make_model_pipeline

N_ITER = 30

# Parameter names refer to the "model" step of the pipeline.
SEARCH_SPACES = {
    # Phase 8: 500 trees was dropped from the RF space to keep the search within
    # the 15-minute budget (see DECISIONS.md). min_samples_leaf >= 5 avoids
    # near-pure leaves, which give poor probabilities and very large models.
    "random_forest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 10, 15, 20, 30],
        "model__min_samples_leaf": [5, 10, 20, 50, 100],
        "model__max_features": ["sqrt", 0.3, 0.5],
        "model__max_samples": [None, 0.5, 0.8],
    },
    # SPEC §9 search space
    "xgboost": {
        "model__n_estimators": randint(200, 801),
        "model__max_depth": randint(3, 9),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 11),
        "model__reg_alpha": uniform(0, 1),
        "model__reg_lambda": uniform(0.5, 4.5),
        "model__gamma": uniform(0, 2),
    },
}


def _describe_space(space: dict) -> dict:
    """JSON-friendly description of a search space."""
    described = {}
    for name, values in space.items():
        if isinstance(values, list):
            described[name] = values
        else:
            low, high = values.support()
            described[name] = f"{values.dist.name}({low:g}, {high:g})"
    return described


def search_signature(model_name: str, imbalance: str, data_sha256: str, n_iter: int = N_ITER,
                     n_folds: int = config.CV_FOLDS) -> dict:
    """Everything that determines a search's outcome. Cached results are reused only on an exact match."""
    return {
        "model": model_name,
        "imbalance": imbalance,
        "data_sha256": data_sha256,
        "search_space": _describe_space(SEARCH_SPACES[model_name]),
        "n_iter": n_iter,
        "cv_folds": n_folds,
        "random_state": config.RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }


def tune_model(model_name: str, estimator, X: pd.DataFrame, y: pd.Series, n_iter: int = N_ITER,
               n_folds: int = config.CV_FOLDS) -> dict:
    """Randomized search over SEARCH_SPACES[model_name]. ``estimator`` should be single-threaded:
    the search itself runs one fit per CPU thread."""
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for '{model_name}'")
    search = RandomizedSearchCV(
        make_model_pipeline(model_name, estimator),
        SEARCH_SPACES[model_name],
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE),
        refit=False,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        error_score="raise",
    )
    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start

    results = pd.DataFrame(search.cv_results_)
    best = results.loc[search.best_index_]
    trials = [
        {
            "params": {k.removeprefix("model__"): v for k, v in row["params"].items()},
            "cv_pr_auc_mean": round(float(row["mean_test_score"]), 5),
            "cv_pr_auc_std": round(float(row["std_test_score"]), 5),
            "mean_fit_time_s": round(float(row["mean_fit_time"]), 2),
        }
        for _, row in results.sort_values("rank_test_score").iterrows()
    ]
    return {
        "model": model_name,
        "method": f"RandomizedSearchCV, n_iter={n_iter}, stratified {n_folds}-fold CV on the training split",
        "scoring": "average_precision (PR-AUC)",
        "search_space": _describe_space(SEARCH_SPACES[model_name]),
        "best_params": {k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        "best_cv_pr_auc_mean": float(best["mean_test_score"]),
        "best_cv_pr_auc_std": float(best["std_test_score"]),
        "tuning_time_s": round(elapsed, 1),
        "trials": trials,
    }
