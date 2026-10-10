"""Generate synthetic rows with the exact cs-training.csv schema (SPEC §19.1).

No real Kaggle rows are used. The generator deliberately includes the data
problems found in the real file: missing income/dependents, 96/98 past-due
codes, zero income, an invalid age of 0, and utilization above 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def make_synthetic_raw(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame shaped like the raw CSV: unnamed ID column, target, 10 features."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(8.5, 0.6, n_rows).round()
    df = pd.DataFrame(
        {
            "": np.arange(1, n_rows + 1),
            config.TARGET_SOURCE: (rng.random(n_rows) < 0.1).astype(int),
            "RevolvingUtilizationOfUnsecuredLines": rng.beta(0.6, 2.0, n_rows),
            "age": rng.integers(21, 95, n_rows),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.3, n_rows),
            "DebtRatio": rng.gamma(1.5, 0.25, n_rows),
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n_rows),
            "NumberOfTimes90DaysLate": rng.poisson(0.2, n_rows),
            "NumberRealEstateLoansOrLines": rng.poisson(1, n_rows),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.2, n_rows),
            "NumberOfDependents": rng.poisson(0.8, n_rows).astype(float),
        }
    )
    df[config.TARGET_SOURCE] = df[config.TARGET_SOURCE].astype(int)
    df.loc[:9, config.TARGET_SOURCE] = 1  # guarantee both classes

    # Known data problems from the real file
    df.loc[10:39, "MonthlyIncome"] = np.nan
    df.loc[10:39, "DebtRatio"] = rng.integers(100, 3000, 30)  # raw debt when income missing
    df.loc[10:17, "NumberOfDependents"] = np.nan
    df.loc[40:44, "MonthlyIncome"] = 0.0
    df.loc[45:49, config.PAST_DUE_COLUMNS] = 98
    df.loc[50, config.PAST_DUE_COLUMNS] = 96
    df.loc[51, "age"] = 0
    df.loc[52:54, "RevolvingUtilizationOfUnsecuredLines"] = [1.5, 20.0, 5000.0]
    df.loc[55:59, "NumberOfDependents"] = 0.0
    return df


def write_synthetic_csv(path: Path, n_rows: int = 300, seed: int = 0) -> Path:
    """Write synthetic data to ``path`` in the same CSV layout as cs-training.csv."""
    make_synthetic_raw(n_rows, seed).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print(make_synthetic_raw().head())
