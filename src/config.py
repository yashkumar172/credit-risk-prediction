"""Project-wide configuration: paths, seeds, schema, split sizes and decision settings."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the project root; no machine-specific absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "cs-training.csv"

MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "credit_risk_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
TEST_METRICS_PATH = METRICS_DIR / "test_metrics.json"
SPLIT_SUMMARY_PATH = METRICS_DIR / "split_summary.json"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.json"
TUNING_RESULTS_PATH = METRICS_DIR / "tuning_results.json"
CALIBRATION_RESULTS_PATH = METRICS_DIR / "calibration_results.json"
THRESHOLD_ANALYSIS_PATH = METRICS_DIR / "threshold_analysis.json"

# ---------------------------------------------------------------------------
# Dataset schema (verified against cs-training.csv in Phase 2)
# ---------------------------------------------------------------------------
# SHA-256 of the Kaggle cs-training.csv used for this project. A different
# hash only triggers a warning: the file may have been re-saved or edited.
EXPECTED_DATA_SHA256 = "1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac"

# The file's first column has an empty header and holds a 1..n row ID.
# It is loaded as the DataFrame index and never used as a feature.
ID_INDEX_NAME = "row_id"

TARGET_SOURCE = "SeriousDlqin2yrs"
TARGET = "default"

# The 10 raw input features, in file order
RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# ---------------------------------------------------------------------------
# Cleaning rules (SPEC §7.2; evidence in DECISIONS.md, Phases 2-3)
# ---------------------------------------------------------------------------
PAST_DUE_CODE_MIN = 90        # past-due values >= this are codes (96/98), not counts
MIN_VALID_AGE = 18            # younger ages are data errors
MAX_VALID_UTILIZATION = 10.0  # larger values are amounts, not utilization ratios

# ---------------------------------------------------------------------------
# Engineered features (SPEC §7.3; selection in DECISIONS.md, Phase 3)
# ---------------------------------------------------------------------------
UTILIZATION_LIMIT = 1.0
ENGINEERED_FEATURES = ["total_past_due", "income_per_dependent", "utilization_over_limit"]
MODEL_FEATURES = [*RAW_FEATURES, *ENGINEERED_FEATURES]

# ---------------------------------------------------------------------------
# Model-specific preprocessing (SPEC §8)
# ---------------------------------------------------------------------------
MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")

# Logistic Regression feature groups
BINARY_FEATURES = ["utilization_over_limit"]
UNSKEWED_FEATURES = ["age"]
SKEWED_FEATURES = [f for f in MODEL_FEATURES if f not in BINARY_FEATURES + UNSKEWED_FEATURES]

# One indicator per distinct missingness pattern. The others are identical:
# DebtRatio / income_per_dependent follow MonthlyIncome, the other past-due
# counts and total_past_due follow NumberOfTimes90DaysLate, and
# utilization_over_limit follows utilization.
MISSING_INDICATOR_FEATURES = [
    "MonthlyIncome",
    "NumberOfDependents",
    "NumberOfTimes90DaysLate",
    "RevolvingUtilizationOfUnsecuredLines",
]

WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data splits (SPEC §5.1): stratified 70 / 15 / 15
# ---------------------------------------------------------------------------
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Decision settings (SPEC §3, §10.2) - illustrative, not banking standards
# ---------------------------------------------------------------------------
# Risk bands: PD < LOW_UPPER -> Low; PD < MEDIUM_UPPER -> Medium; else High
RISK_BAND_LOW_UPPER = 0.10
RISK_BAND_MEDIUM_UPPER = 0.30

# Misclassification costs used to choose the decision threshold
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
