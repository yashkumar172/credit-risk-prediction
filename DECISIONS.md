# Decisions Log

Running log of methodology and engineering decisions (SPEC §17, §24).
Newest entries at the bottom of each phase.

## Phase 1 — Setup (2026-10-08)

- **Python 3.12 instead of 3.11.** Python 3.11 is not installed on the
  development machine; 3.12 is, and every dependency ships wheels for it.
  SPEC §17 updated accordingly.
- **Repository root is the project folder itself** (no extra
  `credit-risk-prediction/` subfolder).
- **`imbalanced-learn` not installed.** SMOTE is optional (SPEC §5.3); the
  package will be added only if that experiment is run.
- **Column names and feature lists are not yet in `config.py`.** They will be
  added in Phase 2 after inspecting `cs-training.csv`.

## Phase 2 — Dataset (2026-10-10)

Inspected `data/raw/cs-training.csv` (SHA-256 `1bd46da4…b24ac`) before writing
any column-dependent code. Full figures are in `notebooks/01_eda.ipynb`, Part 1.

- **Schema matches SPEC §4.3 exactly.** 150,000 rows × 12 columns. Column
  names, target and feature list are now in `config.py`; no SPEC change needed.
- **Row ID.** The unnamed first column is a unique 1…n counter. It is loaded
  as the DataFrame index (`row_id`): kept for traceability and split
  reproducibility, but structurally impossible to use as a feature.
- **Target.** Renamed `SeriousDlqin2yrs` → `default`; 6.684% positive. Loader
  rejects non-binary or missing target values.
- **Expected data hash in `config.py` gives a warning, not an error.** A
  re-saved but otherwise identical file should still load; the hash actually
  used is recorded in the split summary and, later, in model metadata.
- **Duplicates are kept.** 609 duplicate rows (ignoring row ID); 510 of them
  are missing income, and they are mostly thin-file profiles (about one
  credit line, utilization 0 or 1, zero debt ratio, low or missing income),
  where identical values are expected. Each has its own row ID,
  so they are treated as genuinely identical borrowers, not copy errors. Only
  129 test rows (0.6%) share a profile with a training row (recorded in
  `split_summary.json`), so any optimistic effect on test metrics is
  negligible.
- **96/98 codes confirmed.** 269 rows, always coded in all three past-due
  columns together; the largest genuine count is 17. The SPEC §7.2 rule
  (≥ 90 → missing) is confirmed and will be implemented in Phase 4. These
  rows have a distinctive profile (utilization 0.9999999, almost no open
  lines), which stays visible to the models through native NaN handling
  (XGBoost) and missing indicators (Logistic Regression).
- **Age.** One row with age 0 → the SPEC §7.2 rule (< 18 → missing) is
  confirmed. 13 rows aged 101–109 are plausible and kept.
- **Open questions for Phase 3 (training split only):** whether
  `MonthlyIncome == 0` (1,634 rows) should be treated like missing income,
  since its `DebtRatio` median (930) behaves like the missing-income rows
  (1,159) rather than a ratio (0.29); default rates of the 96/98 and
  missing-income groups.
- **For Phase 4:** `estimated_monthly_debt` only makes sense when income is
  present and > 0.
- **Whole-file vs training-split analysis.** Part 1 of the EDA notebook
  reports structural facts for the whole file (allowed by SPEC §4.4/§6). It
  does not relate features to the target. The cleaning rules it confirms were
  fixed in the SPEC beforehand, not chosen from test data. All
  target-related analysis and new decisions happen in Phase 3 on the
  training split only.
- **Split.** Stratified 70/15/15 with seed 42 → 105,000 / 22,500 / 22,500
  rows, 6.68% defaults in each. Reproducibility is recorded as the seed, data
  hash, scikit-learn version and a SHA-256 of each split's sorted row IDs in
  `reports/metrics/split_summary.json` (`python -m src.data_loader`).
- **Testing.** Synthetic fixture generator added (`tests/fixtures/`) with
  the same data problems as the real file; no Kaggle rows are committed.
  `pytest.ini` puts the project root on the import path.
- **`nbformat` / `nbclient` added to `requirements.txt`** so the EDA notebook
  can be re-executed headlessly.

## Phase 3 — EDA on the training split (2026-10-10)

All analysis and decisions in this phase use the **training split only**
(105,000 rows). Evidence is in `notebooks/01_eda.ipynb`, Part 2; figures are
in `reports/figures/eda_*.png`.

- **Correction to Phase 2.** The 96/98-coded rows have utilization
  `0.9999999`, not exactly 1.0. That value is a cap shared by 7,132 training
  rows (18.6% default), so it is not specific to the coded rows. The Phase 2
  notebook text has been corrected.
- **Cleaning rules for Phase 4.** All are stateless; SPEC §7.2 allows rules
  that the EDA justifies.
  1. Past-due ≥ 90 → missing (confirmed). Coded rows default at 56%, so the
     missingness stays visible to the models (native NaN / missing
     indicators).
  2. `age` < 18 → missing (confirmed; 1 training row).
  3. **New: `MonthlyIncome == 0` → missing.** In these rows `DebtRatio` is
     100% whole numbers (median 924), exactly like missing-income rows
     (median 1,160) and unlike rows with income (median 0.29). The ratio was
     never computed, so income is effectively unknown. Default rates: zero
     4.2%, missing 5.6%, present 7.0%.
  4. **New: `DebtRatio` → missing when income is missing** (after rule 3).
     There the value is a debt amount, not a ratio, and shows no consistent
     link to default (4.5–7.8% across quintiles). Keeping it would mix two
     units in one column and distort Logistic Regression's scaling and
     winsorizing.
  5. **New: utilization > 10 → missing.** A ratio above 10 (1,000%) is not
     plausible. Above 100 the values are 99% whole numbers with a 6% default
     rate, so they look like amounts. This affects 165 training rows (0.16%),
     including 16 ambiguous rows between 10 and 100. Utilization 1–10 is
     kept: it is genuinely over the limit and high risk (39%).
- **Engineered features for Phase 4** (SPEC §7.3), screened by univariate
  training-split AUC against their source feature. This is manual feature
  design on training data, not automated feature selection.
  - Build `total_past_due` (0.765 vs 0.679 for the best single count),
    `income_per_dependent` (0.589 vs 0.572 for income), and
    `utilization_over_limit` (captures the 19% → 39% jump at 1.0 that a
    linear model cannot represent from the raw value).
  - Skip `ever_90_days_late` (0.649 vs 0.651, no gain), `monthly_income_missing`
    (duplicates native NaN handling and Logistic Regression's missing
    indicators), `estimated_monthly_debt` (0.509, the weakest), and
    `non_real_estate_lines` (no gain over open lines).
- **Preprocessing (SPEC §8) is confirmed unchanged.** Strong right skew
  supports winsorizing and `log1p` for Logistic Regression. Non-linear,
  U-shaped relationships (open lines, real-estate loans) are expected to
  favour the tree models.
- **Correlations.** The 0.98–0.99 Pearson correlation among the past-due
  counts is caused entirely by the 96/98 codes (0.22–0.31 after cleaning).
  The largest remaining Spearman correlation is 0.47, so no features are
  dropped for multicollinearity.
- **Age** is among the strongest predictors (11.6% default under 30 vs 2.3% at
  70+) and is a protected attribute. It is kept as a feature per SPEC §21, and
  this is noted for the responsible-ML section.

## Phase 4 — Cleaning & feature engineering (2026-10-10)

- **Implemented** `CleaningTransformer` and `FeatureEngineer` in
  `src/feature_engineering.py`, plus `make_feature_pipeline()` as the shared
  first steps of every model pipeline. Thresholds live in `config.py`.
- **Both transformers are stateless.** `fit` only records the expected
  columns, so there is nothing to leak. A test checks that fitting on a subset
  gives the same output as fitting on all the data.
- **Column handling.** `CleaningTransformer` selects the 10 raw features by
  name, so the target, row ID or any extra column is dropped and input column
  order does not matter. Missing columns or non-DataFrame input raise an
  error. The input frame is never modified.
- **Infinite values → NaN** inside the cleaning step, as a defensive measure
  for app inputs (the training data has none).
- **`total_past_due` is NaN if any past-due count is missing.** In practice
  this means the 96/98-coded rows, so the coded group stays distinguishable
  instead of being summed as 0.
- **`utilization_over_limit` is NaN when utilization is unknown** (including
  values removed by the > 10 rule), rather than defaulting to 0.
- **Checked on the real training split:** the NaNs each rule introduces
  match the Phase 3 counts exactly (183 coded rows, 1 age, 1,161 zero
  incomes, 21,898 debt ratios, 165 utilizations). There are no infinite
  values, and the engineered features reproduce their screening AUCs
  (`total_past_due` 0.765, `income_per_dependent` 0.589). Resulting missing
  shares: income / debt ratio / `income_per_dependent` 20.9%, dependents 2.7%,
  past-due counts 0.17%, utilization 0.16%.
- **Note for Phase 12 (app):** when income is entered as "Unknown", rule 4
  makes the model ignore the debt ratio input. The app should say so, or
  disable that field, so users aren't misled.

## Phase 5 — Model-specific preprocessing (2026-10-10)

- **Implemented** `src/preprocessing.py`: `make_preprocessor(model_name)` and
  `make_model_pipeline(model_name, estimator)`, which builds the full pipeline
  (clean → features → preprocess → model). Every learned statistic is fitted
  inside the pipeline, so it only sees the training data, or the training
  folds during CV.
- **Logistic Regression** (SPEC §8, unchanged): median imputation →
  winsorizing at the 1st/99th percentile → `log1p` (skewed features only) →
  `StandardScaler`. `age` is imputed, winsorized and scaled but not
  log-transformed. `utilization_over_limit` is imputed (most frequent value)
  and left unscaled.
- **Missing indicators bypass winsorizing.** In the spec's order they would
  be clipped too, and any indicator set in fewer than 1% of rows would be
  zeroed by the 99th-percentile clip. That includes the past-due indicator
  (0.17% of rows), which marks the 56%-default coded group. They have their
  own `MissingIndicator` branch; a regression test covers this.
- **Four indicators, one per distinct missingness pattern.** On the training
  split the NaN masks of `DebtRatio` and `income_per_dependent` equal
  `MonthlyIncome`'s, the other past-due counts and `total_past_due` equal
  `NumberOfTimes90DaysLate`'s, and `utilization_over_limit`'s equals
  utilization's. Duplicate indicators would only add perfectly collinear
  columns. `age` (1 missing row) gets no indicator. Indicators are produced
  even when a column has no missing values, so the output shape is stable.
- **Winsorizing** uses a small `QuantileClipper` transformer, since
  scikit-learn has none built in. Learned bounds on the training split include
  utilization ≤ 1.062, income 666–22,850, and age 24–87.
- **Random Forest uses native NaN support** (scikit-learn ≥ 1.4) instead of
  median imputation; SPEC §8 allows either. Missingness is informative (coded
  rows 56% default, missing income 5.6%), and median imputation would merge
  the coded rows with "never late". This matches XGBoost and needs no extra
  step.
- **XGBoost:** no preprocessing (native NaN handling), as specified.
- **Checked on the real training split:** 17 LR input columns, all finite;
  scaled columns have mean 0 and std 1; indicator counts are 21,898 / 2,782 /
  183 / 165. Skew is reduced (e.g. debt ratio 33.8 → 1.7, income 132 → −0.6).
  Past-due counts stay skewed (2–4.5) because about 95% of their values are
  0, which no monotonic transform can fix.

## Phase 6 — Logistic Regression baseline (2026-10-10)

- **Implemented** `src/train.py` (entry point `python -m src.train`; it
  currently runs the baseline stage only) and `probability_metrics` in
  `src/evaluate.py`. Results go to `reports/metrics/model_comparison.json`.
- **Protocol** (SPEC §5.1, §10.1): stratified 5-fold CV on the training split
  (shuffle, seed 42), refitting the whole pipeline in each fold; then a fit on
  the full training split and scoring on validation. The test split is not
  used, and a test proves it (the baseline runs with an unusable test split).
- **Estimator:** `LogisticRegression(max_iter=1000)` with the default L2
  penalty and C = 1.0, deliberately untuned because it is the reference point
  (SPEC §9 tunes only the selected model). Convergence warnings are turned
  into errors so an unconverged model can't be reported silently. None
  occurred.
- **Results** (PR-AUC is primary):

  | Variant | CV PR-AUC | Val PR-AUC | Val ROC-AUC | Val Brier | Mean PD (observed 0.0668) |
  |---|---|---|---|---|---|
  | No reweighting | 0.3826 ± 0.0157 | 0.3994 | 0.8646 | 0.0494 | 0.0674 |
  | `class_weight="balanced"` | 0.3801 ± 0.0159 | 0.3972 | 0.8652 | 0.1439 | 0.3354 |

- **Imbalance for LR: no reweighting is preferred.** Balanced weighting does
  not improve ranking (PR-AUC difference 0.0025, far inside the CV std of
  about 0.016). It inflates the mean predicted PD five-fold (0.335 vs an
  observed 0.067), tripling the Brier score. The unweighted model's
  probabilities are already well calibrated on average (0.0674 vs 0.0668).
  This confirms SPEC §5.3's warning; the cross-model decision is made in
  Phase 7.
- **Coefficients make sense:** the largest is the 96/98-coded indicator
  (+2.74), followed by `total_past_due` (+0.64), utilization (+0.62),
  `utilization_over_limit` (+0.33) and age (−0.22). Correlated past-due
  features share weight (the 30–59 count is slightly negative alongside
  `total_past_due`), so individual coefficients shouldn't be read alone.
- **Validation PR-AUC (0.399) sits above the CV mean (0.383)** but inside the
  fold range (0.357–0.406). That is normal split-to-split variation, not a
  sign of leakage.
- **Not yet reported:** precision, recall, F1 and the confusion matrix need
  the decision threshold, which is chosen on calibrated validation
  probabilities (Phase 9, SPEC §10.2). The calibrated Brier score comes with
  model selection in Phase 7.
- **Runtime:** about 15 s for both variants (each CV run in parallel on the
  CPU; final fit about 0.5 s).

## Phase 7 — Ensembles, imbalance comparison and model selection (2026-10-10)

- **Candidates** (SPEC §12: "untuned default or lightly configured"):
  - Logistic Regression as in Phase 6.
  - `RandomForestClassifier(n_estimators=300, min_samples_leaf=10)`.
  - `XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method="hist", device="cpu")`.

  Everything else uses library defaults. Real tuning is Phase 8.
- **Imbalance strategies** (SPEC §5.3): no reweighting vs
  `class_weight="balanced"` (LR, RF) or `scale_pos_weight = neg/pos = 13.96`
  (XGBoost). The ratio is computed from the training split; stratified folds
  keep the same ratio, so using it inside CV leaks nothing about individual
  rows. **SMOTE was not run** (optional in the SPEC; it would add the
  `imbalanced-learn` dependency).
- **Rules, implemented as tested functions in `src/train.py`:**
  - *Imbalance (§5.3):* per model, best CV PR-AUC on train. Within 0.005, no
    reweighting wins, since it is simpler and keeps probabilities
    undistorted.
  - *Selection (§12):* highest validation PR-AUC. Models within 0.005 are
    tied, and ties are broken by calibrated Brier, then simplicity (LR < RF <
    XGBoost), then fit time.
  - *Calibrated Brier for the tie rule:* a sigmoid calibrator fitted with
    5-fold out-of-fold CV inside the validation split, so no row is scored by
    a calibrator that saw it.
- **Results** (test split untouched):

  | Candidate | CV PR-AUC | Val PR-AUC | Val ROC-AUC | Brier raw | Brier cal. | Mean PD | Fit |
  |---|---|---|---|---|---|---|---|
  | LR, none | 0.3826 ± 0.0157 | 0.3994 | 0.8646 | 0.0494 | 0.0495 | 0.0674 | 0.4 s |
  | LR, class_weight | 0.3801 ± 0.0159 | 0.3972 | 0.8652 | 0.1439 | 0.0495 | 0.3354 | 0.5 s |
  | RF, none | 0.3897 ± 0.0135 | **0.4054** | 0.8646 | 0.0491 | 0.0504 | 0.0678 | 12.8 s |
  | RF, class_weight | 0.3857 ± 0.0110 | 0.3992 | 0.8647 | 0.1098 | 0.0496 | 0.2577 | 9.5 s |
  | XGB, none | 0.3869 ± 0.0151 | 0.3994 | 0.8679 | 0.0492 | 0.0509 | 0.0667 | 2.4 s |
  | XGB, scale_pos_weight | 0.3815 ± 0.0127 | 0.4008 | 0.8626 | 0.1297 | 0.0498 | 0.2946 | 2.3 s |

  (Observed validation default rate: 0.0668.)
- **Imbalance decision: no reweighting for every model.** Reweighting never
  improved CV PR-AUC (it was lower by 0.003–0.005 in every case). It always
  inflated the mean PD by 4–5×, which calibration then has to undo.
- **Selection (rule applied as specified): Random Forest, unweighted.** Its
  validation PR-AUC (0.4054) beats LR and XGBoost (both 0.3994) by 0.0060,
  just outside the 0.005 tie tolerance, so the tie-break was not triggered.
  RF also has the best CV PR-AUC, so CV and validation agree on the order.
- **Caveat: the lead is not statistically clear.** This was an ad-hoc check,
  not part of the pipeline. A paired bootstrap of the validation split (2,000
  resamples) gives RF − XGBoost +0.006 [95% CI −0.003, +0.014] and RF − LR
  +0.006 [−0.004, +0.016]. The three models are practically equivalent on
  ranking.
- **Practical cost of the RF choice** (same ad-hoc check):
  - Artifact: 101 MB as saved by joblib, 27 MB compressed. XGBoost is 1 MB.
  - SHAP: about 0.3 s per row, so about 9 minutes for the 2,000-row global
    plots (SPEC §13.2), vs about 1 s for XGBoost.
  - Both still fit SPEC §23 (< 50 MB if compressed) and the 30-minute
    training budget. Tuning could change the size.
- **Finding for Phase 9: scikit-learn's sigmoid calibration hurts the tree
  models.** For models without `decision_function` (RF, XGBoost),
  `CalibratedClassifierCV` fits the sigmoid on the probability scale, not
  log-odds. That made the out-of-fold Brier slightly *worse* than raw (RF
  0.0491 → 0.0504; XGBoost 0.0492 → 0.0509), while isotonic stayed neutral
  (0.0492, 0.0493). It did not affect selection (no tie), but SPEC §11's
  sigmoid-vs-isotonic comparison in Phase 9 must take it into account.
- **Runtime:** `python -m src.train` takes about 2.6 minutes for all six
  candidates.

## Phase 8 — Hyperparameter tuning of Random Forest (2026-10-11)

- **Model:** Random Forest, unweighted, as selected in Phase 7 and confirmed
  by the user.
- **Method** (SPEC §9): `RandomizedSearchCV`, 30 iterations, stratified
  5-fold CV on the training split (the same folds as Phase 7), scored by PR-AUC,
  refitting the whole pipeline in every fold. Validation and test data were
  not used. Implemented in `src/tune.py`, which also holds the SPEC's XGBoost
  space.
- **RF search space** (SPEC §9 only defines one for XGBoost):
  - `n_estimators` {200, 300}
  - `max_depth` {None, 10, 15, 20, 30}
  - `min_samples_leaf` {5, 10, 20, 50, 100}
  - `max_features` {sqrt, 0.3, 0.5}
  - `max_samples` {None, 0.5, 0.8}

  Leaf size 1 was excluded (poor probabilities, very large models). 500 trees
  was dropped before the run after single-threaded timing (40 s for the
  untuned configuration, 120 s for the slowest corner) suggested it would
  exceed the 15-minute target.
- **Rule fixed before the run:** keep the tuned configuration only if its CV
  PR-AUC beats the untuned one on the same folds. That makes the decision
  without using validation data.
- **Result: tuned configuration kept.**
  - Best parameters: `n_estimators=300`, `max_depth=20`,
    `min_samples_leaf=20`, `max_features=0.3`, `max_samples=0.8`.
  - CV PR-AUC 0.3933 ± 0.0139, vs 0.3897 ± 0.0135 untuned (+0.0036).
  - On validation (reported, not used to decide): PR-AUC 0.4054 → 0.4120,
    ROC-AUC 0.8646 → 0.8695, Brier 0.0491 → 0.0488, mean PD 0.0673 (observed
    0.0668).
- **The gain is modest.** It is smaller than the fold-to-fold std (0.014). The
  top 14 of 30 trials lie within 0.001 of each other (0.3924–0.3933), so the
  search found a broad plateau rather than a sharp optimum. Leaf size 5 and
  `max_features=0.5` were consistently the worst settings.
- **The tuned model is smaller:** 15.4 MB compressed (joblib, compress=3) vs
  27 MB untuned, thanks to the depth limit and larger leaves. That is within
  SPEC §23's 50 MB limit. Full fit on the training split: 18.7 s.
- **Sigmoid calibration again worsens Brier** for the tuned RF (raw 0.0488 →
  calibrated 0.0501, out-of-fold), confirming the Phase 7 finding. To handle in
  Phase 9.
- **Runtime exceeds the budget.** Tuning took 1,583 s (26.4 min) against the
  15-minute target, and the full `python -m src.train` took 1,797 s (30.0 min)
  before any later stages exist. Cause: with 12 fits running in parallel on 6
  physical cores, each fit took about 3× as long as when timed
  single-threaded (best configuration: 121 s mean per fit vs a 40 s estimate).
  The plan estimate was wrong. A fix is pending the user's decision.
- **Runtime fix (user decision): cache the tuning search.**
  `python -m src.train` reuses `tuning_results.json` when its signature
  matches exactly: data hash, model, imbalance strategy, search space,
  iterations, CV folds, seed, and scikit-learn/XGBoost versions.
  `--retune` forces a fresh search. Only the search is reused. The
  tuned-vs-untuned comparison, the fit on the training split and the
  validation scores are always recomputed.
- **Signature added to the existing results.** The search above was run just
  before caching existed, with exactly the configuration its signature
  records. The signature was written into the file after safety checks (same
  model, search space and trial count) instead of repeating the 26-minute
  search.
- **Verified:** a cached run reproduces the tuned validation PR-AUC exactly
  (0.41202650…) in 245 s. A from-scratch run (`--retune`) still takes about 30
  minutes on this 6-core laptop before calibration and SHAP are added, so it
  exceeds the SPEC's 30-minute budget. That is accepted and documented here.

## Phase 9 — Calibration, decision threshold and model artifact (2026-10-11)

- **Implemented** `src/calibration.py` (sigmoid vs isotonic comparison, final
  calibrator), threshold selection, table and plot in `src/evaluate.py`, and
  the Phase 9 stage of `src/train.py`, which saves the single artifact and
  its metadata (SPEC §15.1).
- **Interrupted run, resumed with `--from-phase 9`.** The laptop shut down
  after the Phase 9 code was written and before it was run. Rerunning
  `python -m src.train` would have recomputed and rewritten the Phase 6–8
  reports, so a `--from-phase 9` option was added instead. It reads
  `model_comparison.json` and `tuning_results.json` and does not modify them
  (their SHA-256 hashes were checked before and after the run). It checks the
  data hash and the tuning signature, refits the final configuration on the
  training split, and stops unless the refit reproduces the saved validation
  PR-AUC to within 1e-9. It reproduced 0.41202650… exactly. `--retune` cannot
  be combined with it. Seven tests cover it (`tests/test_resume.py`).
- **Calibration** (SPEC §11): fitted on the validation split only, around the
  frozen train-fitted pipeline (`CalibratedClassifierCV(FrozenEstimator(...))`).
  The methods were compared by 5-fold out-of-fold calibration within the
  validation split:

  | Probabilities | Brier | PR-AUC | ROC-AUC | Mean PD (observed 0.0668) |
  |---|---|---|---|---|
  | Raw | 0.04879 | 0.4120 | 0.8695 | 0.0673 |
  | Sigmoid | 0.05014 | 0.4111 | 0.8684 | 0.0668 |
  | Isotonic | 0.04881 | 0.4005 | 0.8673 | 0.0668 |

  **Chosen: isotonic**, by the rule fixed before any Phase 9 results were seen
  (isotonic must beat sigmoid's Brier by at least 1%; it did by 2.7%). As
  Phases 7–8 predicted, scikit-learn's sigmoid (fitted on the probability
  scale for models without `decision_function`) makes Brier worse.
- **Caveats about isotonic (open for review):**
  - It does not improve on the raw forest: Brier 0.04881 vs 0.04879. The raw
    probabilities were already well calibrated (mean 0.0673 vs 0.0668).
  - Isotonic is a step function. The deployed model gives only 58 distinct
    PDs on the validation split, and the ties cost ranking detail: out-of-fold
    PR-AUC 0.4120 → 0.4005, a drop larger than the Phase 7 selection margin
    (0.006).
  - SPEC §11 requires calibration and only compares sigmoid with isotonic, so
    the rule was applied as written. Options to consider before Phase 10:
    keep isotonic, or fit the sigmoid on log-odds (Platt scaling as
    intended), which keeps the ranking. Changing the method after seeing these
    numbers is a validation-set decision and must be recorded as such.
- **Decision threshold** (SPEC §10.2): it minimises 5 × FN + 1 × FP over 0.01–0.99,
  using the deployed (calibrated) model's validation PDs. **Chosen: 0.17**,
  close to the theoretical 1/6 ≈ 0.167 for calibrated probabilities. At 0.17
  on validation: precision 0.326, recall 0.614, F1 0.426, 12.6% flagged,
  TP 923 / FP 1,910 / FN 581 / TN 19,086, cost 4,815. Cost is almost flat
  from 0.14 to 0.17 (4,815–4,817). The costs are illustrative assumptions.
- **The threshold is slightly optimistic.** It is chosen on PDs from a
  calibrator fitted on the same rows. In an ad-hoc check with out-of-fold
  isotonic PDs, the cost at 0.17 is 4,901 (+1.8%). The out-of-fold optimum is
  0.15 (cost 4,848). The difference is small, and 0.17 is kept as the SPEC
  rule specifies.
- **Artifact:** `models/credit_risk_pipeline.joblib`, 15.4 MB (joblib,
  compress=3, within SPEC §23's 50 MB), with `models/model_metadata.json`.
  It takes the 10 raw columns and returns a calibrated PD. The test split was
  not used; it was only created by the deterministic split, never read.
- **Runtime:** `--from-phase 9` takes 18 s, including the refit on the
  training split.

## Phase 10 — SHAP explanations (2026-10-11)

- **Implemented** `src/explain.py`. `unwrap_artifact` takes the frozen,
  train-fitted pipeline out of the calibrated wrapper. `make_explainer` uses
  `TreeExplainer` for Random Forest and XGBoost and `LinearExplainer` for
  Logistic Regression, on the output of the preprocessing steps. The
  module also provides the global report and figures, `explain_row` and
  `format_local_explanation` for local explanations, and `FEATURE_LABELS`
  in `config.py`. Run it with `python -m src.explain` (loads the saved
  artifact, no training). `python -m src.train`, including
  `--from-phase 9`, now ends with the same step (SPEC §15.2).
- **Units deviate from SPEC §13.1 for the Random Forest.** A scikit-learn
  forest outputs an averaged probability, so `TreeExplainer` explains a
  probability, not log-odds. SHAP values are therefore reported in
  percentage points of the uncalibrated PD. On the real artifact they add
  up exactly to that PD (max error 1e-14), including rows with missing
  values (the forest's native NaN routing is handled correctly). Converting
  to log-odds afterwards would break additivity, so it was not done. XGBoost
  and LR keep log-odds, and tests check additivity for all three models.
  SPEC §13.1 now carries an amendment, and the explanation note is worded
  per unit.
- **Calibration and SHAP.** The isotonic calibrator is monotonic
  (non-decreasing), so the ranking and direction of effects carry over. It
  is also a step function: two borrowers with different SHAP profiles can
  get the same displayed PD. The local explanation shows the calibrated PD
  for the decision and the contributions for the uncalibrated output. The
  example borrower has an uncalibrated PD of 34.4% and a calibrated PD of
  29.7%.
- **Global sample** (SPEC §13.2): 2,000 validation rows, stratified by the
  target with `train_test_split(random_state=42)`. That gives 134 defaults
  (6.7%). The SHA-256 of the sorted row IDs is recorded in `shap_global.json`.
  The test split is never read; tests run the stage with an unusable test
  split.
- **Results:** utilization (2.79 pp mean |SHAP|), total past-due count
  (2.67), 90+ days late (1.70), 60–89 days late (0.91) and age (0.79) lead.
  Income-related features are near the bottom (0.26–0.32). Directions (rank
  correlation of value vs SHAP) agree with the EDA: utilization and past-due
  counts increase risk, and age decreases it (ρ = −0.89). Age is the
  protected attribute flagged in Phase 3 and ranks 5th; noted for §21.
- **Display choices:**
  - Input values are shown after cleaning, i.e. what the model actually saw:
    a past-due code of 98 or zero income appears as "Unknown". Engineered
    features appear with their computed values.
  - For LR, values are shown before scaling, and missing-indicator columns
    appear as "<feature> unknown: Yes/No".
  - The text contains only labels, input values and signed contributions,
    with no qualitative wording. A test checks this.
- **Local example** for the README and reports: an invented borrower
  (`SHAP_EXAMPLE_BORROWER` in `config.py`), not a dataset row. See the
  correction below.
- **`risk_band` lives in `src/explain.py`** because the local explanation
  needs it. Phase 12's `predict.py` should import it rather than duplicate
  it.
- **Runtime:** 236 s for the 2,000 rows (about 118 ms per row, single
  process). Local explanations take about 0.1 s per row, which is fine for
  the app. Global SHAP adds about 4 minutes to `python -m src.train`. A
  from-scratch `--retune` run therefore takes about 34 minutes instead of 30,
  on top of the overrun already accepted in Phase 8.
- **`scipy` pinned** in `requirements.txt` (1.18.1, already installed as a
  scikit-learn dependency), because `src/explain.py` imports it directly.
- **Tests:** 28 new tests in `tests/test_explain.py`. They cover
  unwrapping, additivity for RF, XGBoost and LR, labels and value
  formatting, risk-band boundaries, and the SPEC §19.6 end-to-end path from
  synthetic raw input to the SHAP explanation. They also cover the
  qualitative-wording check, the fixed stratified sample, the global report
  and files, and one test on the real artifact that is skipped when it is
  absent. The Phase 9 end-to-end test now redirects the SHAP output paths,
  so it never writes to `reports/`.
- **Not changed:** the model, metadata and all Phase 6–9 reports (SHA-256
  checked before and after).

### Phase 10 correction — synthetic local example (2026-10-11)

- **Problem.** The first Phase 10 commit (never pushed) used a real
  validation row as the local example. Its input values appeared in the README, in `shap_global.json`
  and in `shap_local_example.png`. That publishes a Kaggle record, against
  the competition's data rules and the SPEC's "no real Kaggle rows
  committed" rule (§19.1).
- **Fix.** The example is now an invented borrower,
  `SHAP_EXAMPLE_BORROWER` in `config.py`: a higher-risk profile with
  utilization just over the limit and two 30–59-day delays. It was built by
  hand with round values. A one-off check confirmed that no row of
  `cs-training.csv` matches it on all 10 features; the closest row shares 7,
  mostly zero counts. `example_row` (which picked a real row) was removed,
  and the report entry now records `"source": "synthetic borrower"` and the
  inputs instead of a row ID.
- **Global results unchanged.** Only the `local_example` block of
  `shap_global.json` and `shap_local_example.png` were regenerated, from
  the saved model and without loading the dataset. A script checked that
  every other byte of `shap_global.json` was identical before writing it.
  The beeswarm, the bar chart, the model and the metadata have the same
  SHA-256 as before. No retraining was done.
- **Tests.** Three new tests: the example is a valid raw input, it is not
  present in the synthetic data, and the report entry matches
  `explain_row`. The report test now checks that no row ID is written. 167
  tests in total.
- **Git history.** At the user's request, the Phase 10 commit and this fix
  were squashed into one commit, and the unreachable originals were pruned.
  The repository history therefore never contains the real row. Nothing had
  been pushed.
