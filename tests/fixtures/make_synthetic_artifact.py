"""A small calibrated model and metadata saved like the real artifact, from synthetic data only
(SPEC §19.1: tests must not depend on the trained artifact existing)."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import sklearn

from src import config
from src.calibration import fit_calibrator
from src.data_loader import load_raw_data, split_data
from src.preprocessing import make_model_pipeline
from src.train import make_estimator, split_xy, training_percentiles
from tests.fixtures.make_synthetic_data import write_synthetic_csv


def write_synthetic_artifact(directory: Path, threshold: float = 0.17) -> tuple[Path, Path, dict]:
    """Save ``pipeline.joblib`` and ``metadata.json`` in ``directory``; returns (pipeline, metadata path, metadata)."""
    split = split_data(load_raw_data(write_synthetic_csv(directory / "cs-training.csv", n_rows=1000)))
    X_train, y_train = split_xy(split.train)
    X_val, y_val = split_xy(split.validation)
    pipeline = make_model_pipeline("random_forest", make_estimator("random_forest", n_jobs=1)
                                   .set_params(n_estimators=30)).fit(X_train, y_train)
    artifact = fit_calibrator(pipeline, X_val, y_val, "isotonic")

    metadata = {
        "raw_input_features": config.RAW_FEATURES,
        "decision_threshold": threshold,
        "cost_false_negative": config.COST_FALSE_NEGATIVE,
        "cost_false_positive": config.COST_FALSE_POSITIVE,
        "training_percentiles": training_percentiles(split),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    pipeline_path, metadata_path = directory / "pipeline.joblib", directory / "metadata.json"
    joblib.dump(artifact, pipeline_path)
    metadata_path.write_text(json.dumps(metadata))
    return pipeline_path, metadata_path, metadata
