"""Data loading, schema validation and split tests (SPEC §19.2). No Kaggle data needed."""

import pandas as pd
import pytest

from src import config
from src.data_loader import file_sha256, load_raw_data, split_data, split_summary


def test_load_returns_features_and_renamed_target(synthetic_csv):
    df = load_raw_data(synthetic_csv)
    assert list(df.columns) == [*config.RAW_FEATURES, config.TARGET]
    assert config.TARGET_SOURCE not in df.columns


def test_target_is_binary_after_renaming(synthetic_csv):
    y = load_raw_data(synthetic_csv)[config.TARGET]
    assert set(y.unique()) == {0, 1}
    assert y.dtype.kind == "i"


def test_identifier_is_index_not_feature(synthetic_csv):
    df = load_raw_data(synthetic_csv)
    assert df.index.name == config.ID_INDEX_NAME
    assert not any(c == "" or c.startswith("Unnamed") for c in df.columns)


def test_missing_values_preserved(synthetic_csv, synthetic_raw):
    df = load_raw_data(synthetic_csv)
    assert df["MonthlyIncome"].isna().sum() == synthetic_raw["MonthlyIncome"].isna().sum()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns=["DebtRatio"]), "Missing columns: \\['DebtRatio'\\]"),
        (lambda d: d.assign(extra_col=1), "extra columns: \\['extra_col'\\]"),
        (lambda d: d.drop(columns=[config.TARGET_SOURCE]), "SeriousDlqin2yrs"),
    ],
)
def test_schema_validation_rejects_wrong_columns(tmp_path, synthetic_raw, mutate, message):
    path = tmp_path / "bad.csv"
    mutate(synthetic_raw).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_raw_data(path)


def test_non_binary_target_rejected(tmp_path, synthetic_raw):
    path = tmp_path / "bad.csv"
    synthetic_raw.assign(**{config.TARGET_SOURCE: 2}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_raw_data(path)


def test_missing_target_rejected(tmp_path, synthetic_raw):
    path = tmp_path / "bad.csv"
    bad = synthetic_raw.astype({config.TARGET_SOURCE: float})
    bad.loc[0, config.TARGET_SOURCE] = float("nan")
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_raw_data(path)


def test_missing_file_explains_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle.com/c/GiveMeSomeCredit"):
        load_raw_data(tmp_path / "nope.csv")


def test_split_sizes_stratification_and_no_overlap():
    df = pd.DataFrame(
        {**{c: range(2000) for c in config.RAW_FEATURES}, config.TARGET: [1] * 200 + [0] * 1800}
    )
    split = split_data(df)
    assert [len(s) for s in split] == [1400, 300, 300]
    for part in split:
        assert part[config.TARGET].mean() == pytest.approx(0.10, abs=0.005)
    ids = [set(p.index) for p in split]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == set(df.index)


def test_split_is_deterministic(synthetic_csv):
    df = load_raw_data(synthetic_csv)
    a = split_summary(split_data(df), "hash")
    b = split_summary(split_data(df), "hash")
    assert a == b
    c = split_summary(split_data(df, random_state=7), "hash")
    assert a["splits"]["test"]["row_id_sha256"] != c["splits"]["test"]["row_id_sha256"]


def test_file_sha256_matches_known_value(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
