"""Load and validate the Give Me Some Credit data, and make the train/validation/test split.

Run ``python -m src.data_loader`` to validate the raw file and write the
split summary to ``reports/metrics/split_summary.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)

DOWNLOAD_INSTRUCTIONS = (
    "Download the data manually (Kaggle requires an account):\n"
    "  1. Sign in to Kaggle and accept the rules of the 'Give Me Some Credit'\n"
    "     competition: https://www.kaggle.com/c/GiveMeSomeCredit\n"
    "  2. Download cs-training.csv\n"
    f"  3. Save it as {config.RAW_DATA_FILE.relative_to(config.PROJECT_ROOT)}"
)


class DataSplit(NamedTuple):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError unless the columns are exactly the target plus the raw features."""
    expected = [config.TARGET_SOURCE, *config.RAW_FEATURES]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing or extra:
        raise ValueError(f"Unexpected schema. Missing columns: {missing}; extra columns: {extra}")
    if not df.index.is_unique:
        raise ValueError("Row ID column (first column) contains duplicate values")


def validate_target(y: pd.Series) -> None:
    """Raise ValueError unless the target is binary 0/1 with no missing values."""
    if y.isna().any():
        raise ValueError(f"Target '{y.name}' contains missing values")
    values = set(y.unique())
    if not values <= {0, 1}:
        raise ValueError(f"Target '{y.name}' must be binary 0/1, found values {sorted(values)}")


def load_raw_data(path: Path = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load cs-training.csv, validate it, and return features plus the renamed target.

    The unnamed first column (row ID) becomes the index, so it is kept for
    traceability but can never be used as a feature. The target
    ``SeriousDlqin2yrs`` is renamed to ``default``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.\n{DOWNLOAD_INSTRUCTIONS}")

    df = pd.read_csv(path, index_col=0)
    df.index.name = config.ID_INDEX_NAME
    validate_schema(df)

    df = df.rename(columns={config.TARGET_SOURCE: config.TARGET})
    validate_target(df[config.TARGET])
    df[config.TARGET] = df[config.TARGET].astype(int)

    logger.info("Loaded %s: %d rows, %d features", path.name, len(df), len(config.RAW_FEATURES))
    return df[[*config.RAW_FEATURES, config.TARGET]]


def split_data(df: pd.DataFrame, random_state: int = config.RANDOM_STATE) -> DataSplit:
    """Stratified 70/15/15 train/validation/test split (SPEC §5.1)."""
    train_val, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET],
        random_state=random_state,
    )
    val_share_of_remainder = config.VALIDATION_SIZE / (config.TRAIN_SIZE + config.VALIDATION_SIZE)
    train, validation = train_test_split(
        train_val,
        test_size=val_share_of_remainder,
        stratify=train_val[config.TARGET],
        random_state=random_state,
    )
    return DataSplit(train=train, validation=validation, test=test)


def index_hash(index: pd.Index) -> str:
    """Hash of the sorted row IDs in a split, used to confirm the split is reproduced exactly."""
    ids = ",".join(map(str, sorted(index)))
    return hashlib.sha256(ids.encode()).hexdigest()


def split_summary(split: DataSplit, data_sha256: str) -> dict:
    """Describe a split: sizes, default rates, row-ID hashes and how it was produced."""
    train_features = split.train[config.RAW_FEATURES]
    train_profiles = set(map(tuple, train_features.fillna(-1).itertuples(index=False)))

    summary = {
        "data_sha256": data_sha256,
        "random_state": config.RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "splits": {},
    }
    for name, part in split._asdict().items():
        entry = {
            "rows": len(part),
            "defaults": int(part[config.TARGET].sum()),
            "default_rate": round(float(part[config.TARGET].mean()), 5),
            "row_id_sha256": index_hash(part.index),
        }
        if name != "train":
            # Identical feature profiles also present in train (informational; see DECISIONS.md)
            profiles = part[config.RAW_FEATURES].fillna(-1).itertuples(index=False)
            entry["rows_with_identical_profile_in_train"] = sum(
                tuple(p) in train_profiles for p in profiles
            )
        summary["splits"][name] = entry
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_hash = file_sha256(config.RAW_DATA_FILE) if config.RAW_DATA_FILE.exists() else None
    df = load_raw_data()
    if data_hash != config.EXPECTED_DATA_SHA256:
        logger.warning("Data hash %s differs from the expected %s", data_hash, config.EXPECTED_DATA_SHA256)

    summary = split_summary(split_data(df), data_hash)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.SPLIT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    for name, s in summary["splits"].items():
        logger.info("%-10s rows=%6d defaults=%5d rate=%.4f", name, s["rows"], s["defaults"], s["default_rate"])
    logger.info("Split summary written to %s", config.SPLIT_SUMMARY_PATH.relative_to(config.PROJECT_ROOT))


if __name__ == "__main__":
    main()
