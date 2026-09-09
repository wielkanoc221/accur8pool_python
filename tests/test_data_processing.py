import numpy as np
import pandas as pd
import pytest

from accur8pool.data_processing.const import TIME, TIMESTAMP
from accur8pool.data_processing.data_transformations import (
    DataFrameTransformerBase,
    DataFrameTransformerV2,
)
from accur8pool.data_processing.prepare_raw_data import (
    REQUIRED_COLUMNS,
    WrongColumnsException,
    check_columns,
    prepare_raw_data_and_save,
    transform_raw_df,
)


def make_raw_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in sorted(REQUIRED_COLUMNS - {"timestamp"})})
    df[TIMESTAMP] = 10.0

    return df


def test_check_columns_raises_on_missing():
    with pytest.raises(WrongColumnsException, match="brakujace kolumny"):
        check_columns(pd.DataFrame({"accx": [1.0]}))


def test_check_columns_passes():
    assert check_columns(make_raw_df(n=5)) is None


def test_transform_raw_df_adds_expected_columns():
    out = transform_raw_df(make_raw_df())

    for col in ("acc_magnitude", "gyr_magnitude", "jerk_accx", "acc_magnitude_jerk", "roll", "pitch", TIME):
        assert col in out.columns


def test_v2_add_time_keeps_row_count_on_non_zero_index():
    """dt[0] = 0 na indeksie nie zaczynajacym sie od zera dopisywalo wiersz zamiast nadpisac."""
    df = pd.DataFrame({TIMESTAMP: [1_000_000, 3_000_000, 6_000_000]}, index=[5, 6, 7])

    out = DataFrameTransformerV2(df).add_time().result()

    assert len(out) == 3
    assert out[TIMESTAMP].iloc[0] == 0.0
    assert out[TIME].tolist() == [0.0, 2.0, 5.0]


def test_normalize_without_columns_skips_non_numeric():
    df = pd.DataFrame({"accx": [0.0, 1.0, 2.0], "name": ["a", "b", "c"], TIME: [1.0, 2.0, 3.0]})

    out = DataFrameTransformerBase(df).normalize().result()

    assert out["accx"].tolist() == [0.0, 0.5, 1.0]
    assert out["name"].tolist() == ["a", "b", "c"]
    assert out[TIME].tolist() == [1.0, 2.0, 3.0]


def test_normalize_raises_on_unknown_column():
    with pytest.raises(KeyError):
        DataFrameTransformerBase(pd.DataFrame({"accx": [1.0]})).normalize(columns=["nope"])


def test_lowpass_rejects_nan():
    df = pd.DataFrame({"accx": [1.0, np.nan] * 50})

    with pytest.raises(ValueError, match="NaN"):
        DataFrameTransformerBase(df).lowpass(columns=["accx"], cutoff=10)


def test_prepare_raw_data_and_save_reports_failures(tmp_path):
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    make_raw_df().to_csv(good, index=False)
    pd.DataFrame({"accx": [1.0]}).to_csv(bad, index=False)

    out_dir = tmp_path / "prepared"
    ok_count = prepare_raw_data_and_save([good, bad], out_dir)

    assert ok_count == 1
    assert (out_dir / "good.csv").exists()
    assert not (out_dir / "bad.csv").exists()
