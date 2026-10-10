# Credit Risk Prediction & Explainable Default Probability System

> **Revision 2 (2026-10-08).** Changes from revision 1: fixed the dataset
> (Give Me Some Credit), defined what each data split is for, made
> calibration mandatory after any class reweighting, defined how SHAP
> relates to the calibrated probability, set a primary metric and a
> threshold rule, moved to a single pipeline artifact, defined the run
> commands, and pinned the environment. The age fairness comparison is
> optional.

## 1. Project Overview

Build a production-style, end-to-end **Credit Risk Prediction** system
that estimates the probability that a borrower will become seriously
delinquent (90+ days past due) within two years. This project calls
that event a **default**.

The project is an applied-AIML portfolio project. It should show
practical machine-learning engineering, not research novelty.

### Core objective

Given borrower and credit-history information:

1.  Validate and clean the data.
2.  Engineer meaningful financial features.
3.  Train and compare multiple classification models.
4.  Handle class imbalance correctly.
5.  Tune the strongest model.
6.  Produce a calibrated probability of default.
7.  Explain individual predictions using SHAP.
8.  Evaluate the model with appropriate classification and calibration
    metrics.
9.  Provide an interactive Streamlit application.
10. Keep the whole project runnable locally on a CPU-only laptop.

### Important scope

This is an **educational/portfolio decision-support system**, not a real
lending system. Do not claim that the model is suitable for actual
banking or lending decisions.

------------------------------------------------------------------------

# 2. Project Goals

The final project should demonstrate:

-   Supervised binary classification
-   Data cleaning
-   Exploratory data analysis
-   Feature engineering
-   Handling missing values
-   Handling class imbalance
-   Data leakage prevention
-   Baseline modeling
-   Tree-based machine learning
-   XGBoost
-   Hyperparameter tuning
-   Probability calibration
-   Model evaluation
-   Explainable AI with SHAP
-   Reproducible ML pipelines
-   Model persistence
-   A Streamlit application that runs locally (deploying it to
    Streamlit Community Cloud is optional, see §14.7)
-   Clean software/project structure

The project should put **correct methodology and interpretability**
ahead of maximizing a single score.

------------------------------------------------------------------------

# 3. Problem Definition

## Target

The source target column `SeriousDlqin2yrs` is renamed to `default`:

`default = 1` → borrower experienced 90+ days past-due delinquency
within two years

`default = 0` → borrower did not

User-facing text must describe the target accurately ("serious
delinquency within two years"). It must not imply a specific loan
product or a legal definition of default.

The primary output is:

`Probability of Default (PD)`. This is the **calibrated** probability
(§11).

Example:

> Probability of Default: 18.7%

## Risk bands

The application also maps the PD to a risk band. The band boundaries
live in `config.py`. Defaults:

| Band   | PD range      |
|--------|---------------|
| Low    | PD < 10%      |
| Medium | 10% ≤ PD < 30% |
| High   | PD ≥ 30%      |

These defaults are illustrative, chosen around a base rate of about 7%.
They must not be presented as universally valid banking thresholds.

## Bands vs. decision threshold

These are two separate concepts and must be kept separate:

-   **Risk bands** are a descriptive display aid.
-   **The decision threshold** (§10.2) is a single cut-off chosen on the
    validation set by a cost rule. It defines the binary
    "flag for review" decision and is used for the confusion matrix,
    precision, recall and F1.

The app shows both: the band, and whether the PD is at or above the
decision threshold ("Flagged for manual review: Yes/No").

------------------------------------------------------------------------

# 4. Dataset

## 4.1 Selected dataset

Use **Give Me Some Credit** (Kaggle competition, 2011).

-   Source: <https://www.kaggle.com/c/GiveMeSomeCredit>
-   Use **only `cs-training.csv`**. `cs-test.csv` has no labels and
    must not be used.
-   Expected size: about 150,000 rows, 10 features, about 6.7% positive.
    Verify these figures on inspection.

Alternatives considered and rejected:

-   **German Credit**: only 1,000 rows. A 15% test set would hold about
    45 positives, which is too few for reliable PR-AUC, calibration or
    threshold analysis.
-   **Home Credit Default Risk**: about 2.5 GB across 7+ relational
    tables. It is too heavy for this scope and would make the app's
    input form impractical.

## 4.2 Obtaining the data

Kaggle requires an account and acceptance of the competition rules, so
the data is downloaded **manually**:

1.  Sign in to Kaggle and accept the competition rules.
2.  Download `cs-training.csv`.
3.  Place it at `data/raw/cs-training.csv`. The path is set in
    `config.py`.

`data_loader.py` must fail with a clear message explaining these steps
if the file is missing. Kaggle's rules restrict redistribution, so the
raw data must **not** be committed. This includes test fixtures
(§19.1).

## 4.3 Expected schema

The columns below come from the public documentation. **Verify them
against the actual file before writing code that depends on them.**

| Column | Meaning | Notes |
|---|---|---|
| *(unnamed first column)* | Row ID | Identifier. Drop it; never use it as a feature |
| `SeriousDlqin2yrs` | Target | Rename to `default` |
| `RevolvingUtilizationOfUnsecuredLines` | Credit utilization | Values > 1 exist; some are extreme |
| `age` | Borrower age | Contains invalid values (e.g. 0). Protected attribute, see §21 |
| `NumberOfTime30-59DaysPastDueNotWorse` | Count of 30–59 days late | Contains special codes 96/98 |
| `DebtRatio` | Monthly debt payments / monthly income | Extreme when income is missing |
| `MonthlyIncome` | Monthly income | About 20% missing |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines and loans | |
| `NumberOfTimes90DaysLate` | Count of 90+ days late | Contains special codes 96/98 |
| `NumberRealEstateLoansOrLines` | Real-estate loans | |
| `NumberOfTime60-89DaysPastDueNotWorse` | Count of 60–89 days late | Contains special codes 96/98 |
| `NumberOfDependents` | Dependents | About 2–3% missing |

The dataset has **no** loan amount, credit score, employment length or
loan payment fields. Do not invent features or app inputs that need
them.

## 4.4 Dataset documentation

Before modeling, document in the EDA notebook and the README:

-   Number of rows and features
-   Target distribution
-   Numerical and categorical features. All features in this dataset
    are expected to be numerical.
-   Missing-value percentages
-   Duplicate rows. Report them, and record whether they are kept (as
    genuine identical profiles) or removed, with the reason.
-   Outliers and special codes
-   Identifier columns
-   Potential leakage columns. All features are measured at observation
    time; confirm that none of them is recorded after the outcome.
-   SHA-256 hash of `cs-training.csv`, stored in model metadata

------------------------------------------------------------------------

# 5. Important ML Methodology Rules

These rules are mandatory.

## 5.1 Data splits and their roles

Make one stratified split of `cs-training.csv`, with `random_state`
taken from `config.py`:

| Split | Share | Used for |
|---|---|---|
| **Train** | 70% | Fitting models; stratified 5-fold cross-validation for imbalance strategy and hyperparameter tuning |
| **Validation** | 15% | Model selection, fitting the calibrator, choosing the decision threshold, global SHAP plots |
| **Test** | 15% | Final evaluation, **once**, at the end |

Rules:

-   The final model is the pipeline fitted on **train only**, wrapped
    by a calibrator fitted on **validation only**. Do **not** refit on
    train + validation. This keeps the calibrator's data independent
    of the model's training data.
-   The test set is evaluated once, by `python -m src.evaluate --final`
    (§15.2). That command writes `reports/metrics/test_metrics.json`
    and refuses to run again if the file exists, unless `--force` is
    passed. Any forced re-run must be recorded in `DECISIONS.md`.
-   Split indices, or the seed together with the data hash, are saved
    so the split can be reproduced exactly.

## 5.2 Prevent data leakage

Never fit preprocessing on the complete dataset before splitting.
Anything learned from data must be fitted only on training data (or
training folds inside CV). This includes:

-   Imputation
-   Scaling
-   Outlier capping/winsorizing quantiles
-   Encoding
-   SMOTE
-   Feature engineering that uses dataset-wide statistics

Stateless, rule-based cleaning, such as mapping special codes to
missing, is safe because it learns nothing from data. It is still
implemented inside the pipeline (§7.1) so that inference applies it
the same way.

Use `sklearn.pipeline.Pipeline` and `ColumnTransformer`. Feature
engineering and cleaning must be scikit-learn transformers inside the
pipeline, not separate pre-processing scripts.

This project does **not** perform automated feature selection. If that
changes, selection must be fitted inside CV folds on training data
only, and the leakage audit (§20) must be updated.

## 5.3 Class imbalance

First measure the target imbalance.

Compare these strategies using CV PR-AUC on the training set:

-   No reweighting
-   Class weights (`class_weight="balanced"`) for LR and RF
-   XGBoost `scale_pos_weight`
-   SMOTE (optional; only through an `imblearn` pipeline so it is
    applied to training folds only)

Do not assume that reweighting or SMOTE is better.

**Reweighting and calibration rule (mandatory):** class weights,
`scale_pos_weight` and SMOTE all distort predicted probabilities and
push them upward. Whatever strategy is chosen, the probability shown to
users is **always** the output of the calibrator fitted on the
validation set (§11). Raw scores from a reweighted model must never be
displayed as a PD.

------------------------------------------------------------------------

# 6. Exploratory Data Analysis

`notebooks/01_eda.ipynb` is the only mandatory notebook. It imports
helper functions from `src/` and contains no pipeline logic that the
model depends on. EDA is performed on the **training split only**, so
that analyst decisions are not informed by the test set. Overall
dataset shape and target rate may be reported for the full file.

## Dataset overview

-   Shape
-   Data types
-   Summary statistics
-   Target distribution

## Missing values

-   Missing count and percentage
-   Default rate for missing vs. present `MonthlyIncome`. This decides
    whether a missing-indicator feature is useful.

## Distributions

-   Histograms (log scale where values are heavily skewed)
-   Box plots
-   Distributions split by target

## Relationships

-   Correlation matrix. Note the very high correlation between the
    three past-due counts, which is partly driven by the 96/98 codes.
-   Default rate across binned feature ranges

## Outliers and special codes

Identify and decide on:

-   Past-due counts of 96/98 (special codes, not real counts)
-   `age` values below 18
-   `RevolvingUtilizationOfUnsecuredLines` far above 1
-   Extreme `DebtRatio`, particularly when `MonthlyIncome` is missing

For each one, classify it as genuine, data error, or needing
transformation/capping. Do not delete rows automatically; handle them
through the pipeline rules in §7.1.

## EDA conclusion

Document the key findings and how each one feeds into the cleaning
rules, engineered features and preprocessing.

------------------------------------------------------------------------

# 7. Cleaning & Feature Engineering

Everything in this section lives in `src/feature_engineering.py` as
scikit-learn transformers. They are the first steps of the model
pipeline, so training, evaluation and the app all apply identical
logic.

## 7.1 Order of operations

``` text
raw input (10 source columns)
→ CleaningTransformer      (stateless rules)
→ FeatureEngineer          (ratios/flags on cleaned raw values; NaN propagates)
→ model-specific preprocessing (§8)
→ model
→ calibrator
```

## 7.2 Cleaning rules (stateless; confirm each one during EDA)

-   Past-due counts ≥ 90 (the 96/98 codes) → `NaN`
-   `age` < 18 → `NaN`
-   Drop the identifier column if present
-   Any other rule that EDA justifies, recorded in `DECISIONS.md`

## 7.3 Candidate engineered features

Implement the ones the EDA supports, and document each in the README:

| Feature | Definition |
|---|---|
| `total_past_due` | Sum of the three past-due counts |
| `ever_90_days_late` | `NumberOfTimes90DaysLate > 0` |
| `monthly_income_missing` | 1 if `MonthlyIncome` is missing |
| `income_per_dependent` | `MonthlyIncome / (NumberOfDependents + 1)` |
| `estimated_monthly_debt` | `DebtRatio × MonthlyIncome` (NaN when income is missing) |
| `utilization_over_limit` | `RevolvingUtilizationOfUnsecuredLines > 1` |
| `non_real_estate_lines` | `NumberOfOpenCreditLinesAndLoans − NumberRealEstateLoansOrLines`, clipped at 0 |

Requirements:

-   Reproducible and implemented in code
-   No target leakage
-   Division by zero handled explicitly (`+1` denominators, or `NaN`
    rather than `inf`)
-   No `inf` values after transformation; this is tested
-   Output feature names exposed via `get_feature_names_out` for SHAP

------------------------------------------------------------------------

# 8. Modeling Strategy

All models share the cleaning and feature-engineering steps. Model-
specific preprocessing:

| Model | Preprocessing |
|---|---|
| **Logistic Regression** (interpretable baseline) | Median imputation + missing indicators → winsorizing at 1st/99th percentile (fitted on train) → `log1p` for skewed features → `StandardScaler` |
| **Random Forest** (nonlinear baseline) | Median imputation (scikit-learn ≥ 1.4 RF also accepts NaN natively; either is acceptable, but record the choice) |
| **XGBoost** (main model) | None; native NaN handling |

XGBoost configuration: `tree_method="hist"`, `device="cpu"`,
`n_jobs=-1`. Do not require CUDA or a GPU.

------------------------------------------------------------------------

# 9. Hyperparameter Tuning

Tune the **model selected in §12** (expected to be XGBoost) using
`RandomizedSearchCV`. Optuna may be used only if it stays lightweight.

-   Stratified 5-fold CV on the **training set**
-   Scoring: `average_precision` (PR-AUC)
-   About 30 iterations; target a tuning time of 15 minutes or less on
    a typical laptop CPU
-   Never tune on validation or test data

XGBoost search space:

``` text
n_estimators       200–800
max_depth          3–8
learning_rate      0.01–0.2 (log-uniform)
subsample          0.6–1.0
colsample_bytree   0.6–1.0
min_child_weight   1–10
reg_alpha          0–1
reg_lambda         0.5–5
gamma              0–2
scale_pos_weight   per the §5.3 decision
```

Save to `reports/metrics/tuning_results.json`:

-   Search space
-   Best parameters
-   Best CV PR-AUC, with mean and std across folds
-   Total tuning time

------------------------------------------------------------------------

# 10. Evaluation Metrics

Accuracy must NOT be the primary metric.

## 10.1 Metrics

**Primary metric: PR-AUC (average precision).**

Also report:

-   ROC-AUC
-   Brier score
-   Precision, recall and F1 for the positive (default) class at the
    chosen decision threshold
-   Confusion matrix at the chosen decision threshold
-   Mean predicted PD vs. observed default rate (calibration-in-the-
    large)

**Stability:**

-   CV mean ± std of PR-AUC for each model on the training set
-   Bootstrap 95% confidence intervals (1,000 resamples) for test
    PR-AUC, ROC-AUC and Brier score

## 10.2 Decision threshold

Do not assume 0.5. Choose the threshold on the **validation set using
calibrated probabilities**, by minimizing expected misclassification
cost:

``` text
cost = C_FN × FN + C_FP × FP
```

The defaults in `config.py` are `C_FN = 5` and `C_FP = 1`, meaning a
missed defaulter is assumed to cost five times as much as a false
alarm. These are illustrative assumptions and must be stated as such.

Also produce a threshold-analysis table and plot over 0.05–0.95 showing
precision, recall, FP count, FN count and cost. Record the selected
threshold in the model metadata and in the README.

------------------------------------------------------------------------

# 11. Probability Calibration

Calibration is mandatory, because the app displays a PD and because
§5.3 may introduce reweighting.

-   Fit the calibrator on the **validation set only**, wrapping the
    train-fitted pipeline. With scikit-learn ≥ 1.6, use
    `CalibratedClassifierCV(FrozenEstimator(pipeline), method=...)`;
    `cv="prefit"` is deprecated.
-   Compare `sigmoid` (Platt) and `isotonic`. Because the validation
    set has only about 22,500 rows and about 1,500 positives, prefer
    sigmoid unless isotonic gives a clearly better Brier score.
    Compare them with CV on the validation set, or a small held-out
    portion of it, so the choice is not made on the same data the
    calibrator is fitted on.
-   Report on the **test set**, both raw and calibrated:
    -   Calibration curve (reliability diagram)
    -   Brier score
    -   Mean predicted PD vs. observed default rate

Never fit calibration on training or test data.

------------------------------------------------------------------------

# 12. Model Selection

Select the model on the **validation set**, using untuned default or
lightly configured versions of the three models:

1.  **Primary:** highest validation PR-AUC.
2.  **Tie rule:** models within 0.005 PR-AUC of the best count as
    tied. Among tied models, prefer the lower Brier score after
    calibration, then the simpler and faster model.
3.  Also report ROC-AUC, CV stability (std), and training/inference
    time.

Then tune the selected model (§9), calibrate it (§11), choose the
threshold (§10.2), and evaluate once on test.

Final comparison table, with all values from actual experiments; never
fabricate:

| Model | CV PR-AUC (mean ± std) | Val PR-AUC | Val ROC-AUC | Brier (raw) | Brier (calibrated) | Train time |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Random Forest | | | | | | |
| XGBoost | | | | | | |
| Tuned XGBoost | | | | | | |

Plus a separate **final test table** for the chosen model: PR-AUC,
ROC-AUC, Brier (raw and calibrated), precision, recall, F1, the
confusion matrix at the chosen threshold, and the bootstrap confidence
intervals.

------------------------------------------------------------------------

# 13. Explainable AI

Use **SHAP**.

## 13.1 What SHAP explains (must be stated in the app and README)

SHAP explains the **underlying uncalibrated model** in **log-odds**
units. The calibrator is a monotonic transform applied afterwards, so
it does not change the ranking or direction of feature effects. The
SHAP contributions will not add up exactly to the displayed calibrated
PD, and the UI must say this in one sentence.

-   XGBoost / Random Forest: `shap.TreeExplainer` on the fitted model,
    using the output of the pipeline's preprocessing steps as input.
-   Logistic Regression, if selected: `shap.LinearExplainer`.
-   `src/explain.py` is responsible for extracting the inner model and
    the preprocessing steps from the calibrated wrapper.

## 13.2 Global explanation

Computed on a stratified sample of **2,000 validation rows**, with a
fixed seed. Document the sampling.

-   SHAP summary (beeswarm) plot
-   Mean |SHAP| bar chart
-   Top features list

Saved to `reports/figures/` and shown in the app.

## 13.3 Local explanation

For a single prediction, computed live in the app:

-   Calibrated PD and risk band
-   The top 3 features **increasing** risk and the top 3 **decreasing**
    risk, ranked by SHAP value
-   A SHAP waterfall or bar plot

Each feature is shown with its **human-readable label** (from the
`FEATURE_LABELS` dictionary in `config.py`), the **input value**, and
its **SHAP contribution**:

``` text
Probability of Default: 71.4%   (Risk band: HIGH)

Factors increasing risk:
- Times 90+ days late: 2           (+1.12 log-odds)
- Revolving credit utilization: 0.97   (+0.64)
- Total past-due count: 4          (+0.41)

Factors reducing risk:
- Age: 62                          (−0.38)
- Open credit lines: 9             (−0.12)
```

Explanation text must come only from the SHAP values and the input
values. Do not write qualitative claims like "high income" that the
model output does not provide.

------------------------------------------------------------------------

# 14. Streamlit Application

The app **only loads** precomputed artifacts (`models/` and
`reports/`). It never trains models, never reads the raw dataset, and
never touches the test set.

## 14.1 Home

-   What the project does
-   What the target means (serious delinquency within 2 years)
-   Key limitations

## 14.2 Prediction

The inputs are the **10 raw source features**. They are listed in
`model_metadata.json`, and the form is built from that list rather
than hard-coded.

-   Utilization, age, the three past-due counts, debt ratio, monthly
    income, open credit lines, real-estate loans, dependents
-   `MonthlyIncome` and `NumberOfDependents` each have an **"Unknown"**
    checkbox that sends `NaN`, because the pipeline handles missing
    values
-   Hard limits: no negative values; age 18–110
-   Show a **warning** when an input falls outside the training data's
    1st–99th percentile range (stored in metadata): "the prediction may
    be less reliable"

After clicking **Predict Risk**, display:

``` text
Probability of Default: XX.X%
Risk Band: LOW / MEDIUM / HIGH
Flagged for manual review: Yes / No   (decision threshold: X.XX)
```

plus the local SHAP explanation (§13.3).

## 14.3 Model Performance

Loaded from `reports/metrics/*.json` and `reports/figures/`:

-   Model comparison table
-   ROC and precision-recall curves
-   Confusion matrix at the chosen threshold
-   Calibration curves (raw vs. calibrated)
-   Threshold-analysis plot
-   Final test metrics with confidence intervals

## 14.4 Explainability

Global SHAP plots, with the log-odds note from §13.1.

## 14.5 About / Limitations

-   Dataset limitations: 2011-era data, unknown population, no loan
    details
-   Potential bias and the use of `age` (§21)
-   Distribution shift
-   Why the model must not be used for real lending decisions

## 14.6 Structure

Use either Streamlit multipage (`pages/` or `st.navigation`) or tabs;
whichever is simpler. Use `st.cache_resource` to load the model once.

## 14.7 Deployment

The required target is **local** (`streamlit run app.py`). Deploying to
Streamlit Community Cloud is an **optional** extra. It works because
the model artifacts are committed (§23); if attempted, document the
public URL in the README.

------------------------------------------------------------------------

# 15. Project Architecture

## 15.1 Structure

``` text
credit-risk-prediction/
│
├── data/
│   └── raw/                      # cs-training.csv (git-ignored)
│
├── notebooks/
│   └── 01_eda.ipynb              # mandatory; imports from src/
│                                 # further notebooks optional, no pipeline logic
├── src/
│   ├── __init__.py
│   ├── config.py                 # paths, seeds, split sizes, bands, costs, FEATURE_LABELS
│   ├── data_loader.py            # load, validate schema, rename target, hash file
│   ├── feature_engineering.py    # CleaningTransformer, FeatureEngineer
│   ├── preprocessing.py          # per-model ColumnTransformers
│   ├── train.py                  # entry point: §15.2
│   ├── tune.py
│   ├── calibration.py
│   ├── evaluate.py               # validation reports + one-time --final test evaluation
│   ├── explain.py
│   └── predict.py                # load artifact, predict PD, band, flag, explanation
│
├── models/
│   ├── credit_risk_pipeline.joblib   # ONE artifact: cleaning → FE → preprocessing → model → calibrator
│   └── model_metadata.json
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── tests/
│   ├── fixtures/
│   │   └── make_synthetic_data.py    # generates schema-identical synthetic rows
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_prediction.py
│   └── test_app_schema.py
│
├── app.py
├── requirements.txt
├── README.md
├── DECISIONS.md                  # running log of methodology decisions
├── .gitignore
└── SPEC.md                       # this file
```

The structure may be simplified, but data processing, training,
evaluation, inference and UI must stay separate.

### Single-artifact rule

There is exactly one model artifact, `credit_risk_pipeline.joblib`. It
contains every step from raw input to calibrated probability. There
are no separately saved preprocessors that could drift out of sync
with the model.

`model_metadata.json` contains:

-   Ordered raw input feature names
-   Engineered/transformed feature names
-   Per-feature training 1st/99th percentiles
-   Selected model type and hyperparameters
-   Calibration method
-   Decision threshold, cost settings, and risk-band boundaries
-   Random seed
-   SHA-256 hash of the training data file
-   Training timestamp
-   Python, scikit-learn, xgboost and shap versions

`predict.py` warns if the installed library versions differ from the
ones recorded in metadata.

## 15.2 Commands

``` bash
python -m src.train              # split, CV comparison, tuning, calibration, threshold,
                                 # validation metrics/figures, global SHAP, saves models/
python -m src.evaluate --final   # ONE-TIME test evaluation → reports/metrics/test_metrics.json
pytest                           # runs without the Kaggle data
streamlit run app.py
```

`src.train` uses fixed seeds and must finish in about 30 minutes or
less on a typical laptop CPU, including tuning.

------------------------------------------------------------------------

# 16. Code Quality Requirements

The code should be:

-   Modular
-   Readable
-   PEP 8 compliant where practical
-   Reproducible
-   Properly documented
-   Free of unnecessary duplication

Use:

-   Functions with clear responsibilities
-   Type hints where useful
-   Meaningful variable names
-   Constants in `config.py`
-   Fixed random seeds
-   `logging` instead of `print`

Do not hard-code machine-specific absolute paths. Build paths relative
to the project root (e.g. `Path(__file__).resolve().parents[1]`).

------------------------------------------------------------------------

# 17. Reproducibility

-   **Python 3.12.**
-   `requirements.txt` lists **direct dependencies pinned to exact
    versions** (`==`), taken from the environment where the final
    model was trained. Expected packages: pandas, numpy,
    scikit-learn (≥ 1.6), xgboost (≥ 2.0), shap, matplotlib,
    streamlit, joblib, pytest, and optionally imbalanced-learn and
    ipykernel.
-   Model artifacts are only guaranteed to load with the pinned
    versions (see the version check in §15.1).
-   Record: random seeds, dataset source and hash, cleaning decisions,
    hyperparameters. Methodology decisions go in `DECISIONS.md` as they
    are made.

Setup on Windows PowerShell:

``` powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

Then follow the dataset steps in §4.2 and the commands in §15.2.

------------------------------------------------------------------------

# 18. CPU-Only Requirement

This project must run without a GPU. Do NOT require CUDA, an NVIDIA
GPU, or any deep-learning framework.

-   XGBoost: `tree_method="hist"`, `device="cpu"`
-   Bounded tuning (§9)
-   Global SHAP on a documented 2,000-row sample (§13.2); local SHAP
    for one row at a time

------------------------------------------------------------------------

# 19. Testing

## 19.1 Fixtures

Tests must **not** depend on the Kaggle file or on trained artifacts
existing.

`tests/fixtures/make_synthetic_data.py` (or a pytest fixture)
generates a few hundred **synthetic** rows with the exact source
schema. These rows include missing values, the 96/98 codes, zero
income, zero dependents, and both target classes. No real Kaggle rows
are committed.

Tests that need the real trained artifact are marked to **skip** when
`models/credit_risk_pipeline.joblib` is absent.

## 19.2 Data tests

-   Schema validation detects missing or extra columns
-   Target exists and is binary after renaming
-   Identifier column is dropped

## 19.3 Feature tests

-   Each engineered feature is computed correctly on hand-built rows
-   96/98 codes and age < 18 become `NaN`
-   Division by zero produces no `inf`
-   The target column never appears among the features

## 19.4 Prediction tests

-   Pipeline trained on the synthetic fixture returns probabilities in
    [0, 1]
-   Inputs with `NaN` income or dependents are handled
-   Risk band and review flag are valid and consistent with config
-   (Skip if absent) The real artifact loads and predicts

## 19.5 App-schema test

The app's input fields match `model_metadata.json`'s raw feature list
exactly, in order.

## 19.6 End-to-end test

``` text
synthetic raw input
→ cleaning → feature engineering → preprocessing
→ model → calibrator
→ probability → risk band → review flag → SHAP explanation
```

------------------------------------------------------------------------

# 20. Data Leakage Audit

Before finalizing, answer each question in the README with evidence (a
code reference or explanation):

-   Was the target used in any feature?
-   Were imputation, scaling or capping statistics computed on
    anything other than training data or training folds?
-   Was SMOTE (if used) applied outside training folds?
-   Was any feature selection done? (Expected answer: no.)
-   Was the validation or test set used during hyperparameter tuning?
-   Was the calibrator fitted on validation data only?
-   Was the decision threshold chosen on validation data only?
-   Was EDA-driven decision-making restricted to the training split?
-   Was the test set evaluated exactly once? Mention any `--force`
    re-run.
-   Are any features recorded after the outcome period?

------------------------------------------------------------------------

# 21. Fairness and Responsible ML

Include a short responsible-ML section in the README and the app's
About page covering:

-   Historical bias in financial datasets
-   Demographic bias, including the use of `age`, which is a protected
    attribute in many lending regulations. The model includes `age` by
    default because it is a source feature. Document this decision and
    its implications.
-   Proxy variables
-   Dataset shift (2011 data)
-   The cost of false positives vs. false negatives
-   Interpretability
-   Human oversight (the model output is a "flag for review", not a
    decision)

**Optional, not required for completion:**

-   Train a variant without `age` and compare its metrics
-   Report performance metrics by age band

If either is done, do not claim the model is fair just because group
metrics look similar.

------------------------------------------------------------------------

# 22. README Requirements

1.  **Project title:** Credit Risk Prediction & Explainable Default
    Probability System
2.  **Problem statement:** in simple terms, including the exact target
    definition
3.  **Objectives**
4.  **Dataset:** source and link, manual download steps, rows,
    features, target, class distribution, data hash
5.  **Architecture:** a Mermaid diagram of the pipeline and artifacts
6.  **Methodology:** data → cleaning → feature engineering → baseline →
    ensemble models → tuning → calibration → threshold → evaluation →
    SHAP → application, including the split roles from §5.1
7.  **Results:** the real comparison table and final test table with
    confidence intervals, the chosen threshold, and the cost
    assumptions
8.  **Explainability:** example SHAP output and the log-odds note
9.  **Leakage audit:** the answers from §20
10. **Application:** how to run it, plus screenshots
11. **Limitations and responsible ML:** from §21
12. **Future improvements:** monitoring, external validation, model
    versioning, the optional fairness experiments, etc.

------------------------------------------------------------------------

# 23. GitHub Presentation

Include:

-   A clear README
-   Architecture diagram
-   Results table
-   Streamlit screenshots. These are taken manually by the developer
    and saved to `reports/figures/app_*.png`.
-   Example prediction
-   SHAP visualization
-   Setup instructions
-   Clean directory structure

**Commit:**

-   `models/credit_risk_pipeline.joblib` and `model_metadata.json`, if
    the joblib is under 50 MB, so the app runs right after cloning
-   `reports/`

**Do not commit:**

``` text
.env
data/raw/  (Kaggle data, any real rows)
.venv/ and other virtual environments
__pycache__/, .ipynb_checkpoints/, temporary files
```

------------------------------------------------------------------------

# 24. Development Phases

At the end of each phase, record any decisions in `DECISIONS.md`.

1.  **Setup:** repository, Python 3.12 venv, directory structure,
    `requirements.txt`, `.gitignore`, `SPEC.md`, Git.
2.  **Dataset:** manual download, `data_loader.py` with schema
    validation and hashing, verify the §4.3 schema, split (§5.1) and
    save split metadata.
3.  **EDA (training split):** summary, target, missing values,
    distributions, special codes, correlations, findings.
4.  **Cleaning & feature engineering:** transformers in §7, with
    tests.
5.  **Preprocessing:** per-model ColumnTransformers in §8.
6.  **Baseline:** Logistic Regression; CV and validation metrics.
7.  **Ensembles and imbalance:** Random Forest and XGBoost; the
    imbalance-strategy comparison (§5.3); selection (§12).
8.  **Tuning:** §9; save results.
9.  **Calibration and threshold:** §11 and §10.2; save the single
    artifact and metadata.
10. **Explainability:** global and local SHAP (§13).
11. **Final test evaluation:** `python -m src.evaluate --final`, once.
12. **Application:** §14.
13. **Testing:** the full `pytest` suite passes.
14. **Documentation:** README with real results and the leakage audit.
15. **Final review:** run the acceptance checklist below.

------------------------------------------------------------------------

# 25. Final Acceptance Criteria

-   [ ] Give Me Some Credit dataset documented, including its hash
-   [ ] Target correctly defined and described
-   [ ] Train/validation/test split follows the §5.1 roles
-   [ ] Cleaning rules and engineered features implemented as pipeline
        transformers
-   [ ] Missing values and special codes handled
-   [ ] Class imbalance strategies compared
-   [ ] Logistic Regression, Random Forest and XGBoost implemented
-   [ ] Model selected by the §12 rule
-   [ ] Hyperparameter tuning completed on training CV only
-   [ ] Calibrator fitted on validation only; raw vs. calibrated
        compared
-   [ ] Decision threshold chosen by the cost rule on validation
-   [ ] Test set evaluated exactly once
-   [ ] PR-AUC, ROC-AUC, Brier, precision, recall, F1, confusion
        matrix and bootstrap confidence intervals reported
-   [ ] SHAP global and local explanations implemented, with the
        log-odds note
-   [ ] Single pipeline artifact and metadata saved
-   [ ] Streamlit app works, loads artifacts only, and validates
        inputs
-   [ ] `pytest` passes without the Kaggle data
-   [ ] Leakage audit documented
-   [ ] README and `DECISIONS.md` completed
-   [ ] Repository clean; no raw data committed
-   [ ] Runs on CPU only; `src.train` finishes in about 30 minutes or
        less

The optional items in §14.7 and §21 are not required.

------------------------------------------------------------------------

# 26. Important Instructions for Claude CLI

1.  **Do not generate all files at once.**
2.  Inspect the repository and environment first.
3.  Create the project structure.
4.  Inspect the actual dataset and verify §4.3 before writing code that
    depends on columns.
5.  Implement one phase at a time (§24).
6.  Run tests and small experiments after each major stage.
7.  Never fabricate metrics or results.
8.  Never claim a model improved unless experiments show it.
9.  Never use validation or test data for tuning, and never use test
    data for anything except the single final evaluation.
10. Never introduce data leakage.
11. Prefer simple, robust implementations over unnecessary complexity.
12. Keep the project CPU-only.
13. Keep the app consistent with the single pipeline artifact and its
    metadata.
14. If a decision depends on the data, inspect the data first.
15. If a package or API is unavailable, choose a compatible
    alternative and record it in `DECISIONS.md`.
16. Update the README with actual results after the experiments.
17. Do not add deep-learning components. This is a tabular ML project.
18. Do not turn this into a research project. The goal is a polished,
    technically strong **applied ML system**.

------------------------------------------------------------------------

# 27. Desired Final Outcome

``` text
Raw Data → Data Understanding → Cleaning → Feature Engineering
→ Robust ML Pipeline → Model Comparison → Hyperparameter Optimization
→ Probability Calibration → Threshold Selection → Explainability
→ Final Evaluation → Application
```

The result should be a **clean, reproducible, explainable, CPU-friendly
credit-risk prediction application** that demonstrates applied
machine-learning skills in an AIML portfolio.
