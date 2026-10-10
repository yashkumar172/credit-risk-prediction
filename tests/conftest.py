import pandas as pd
import pytest

from tests.fixtures.make_synthetic_data import make_synthetic_raw, write_synthetic_csv


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic_raw()


@pytest.fixture
def synthetic_csv(tmp_path):
    return write_synthetic_csv(tmp_path / "cs-training.csv")
