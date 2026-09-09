import numpy as np
import pandas as pd
import pytest

from accur8pool.files_utils.dataFrameLoader import PandasDataFrameLoader
from accur8pool.files_utils.dataFrameUtils import (
    checkSameColsAmount,
    concat_data_with_label,
    concatDataFramesRows,
)


def test_concat_data_with_label_ignores_misaligned_index():
    """Bez resetu indeksu pd.concat(axis=1) wstawialby ciche NaN-y."""
    data = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 11, 12])
    label = pd.Series([7, 8, 9], name="label", index=[0, 1, 2])

    out = concat_data_with_label(data, label)

    assert out["x"].tolist() == [1, 2, 3]
    assert out["label"].tolist() == [7, 8, 9]
    assert not out.isna().any().any()


def test_concat_data_with_label_row_mismatch():
    with pytest.raises(ValueError, match="rozna ilosc wierszy"):
        concat_data_with_label(pd.DataFrame({"x": [1, 2]}), pd.Series([1], name="label"))


def test_concat_rows_accepts_single_frame():
    """Wczesniej wymagane byly >= 2 ramki, przez co zbior z jednego pliku nie przechodzil."""
    df = pd.DataFrame({"x": [1, 2]})

    assert concatDataFramesRows([df]).equals(df)


def test_concat_rows_rejects_empty():
    with pytest.raises(ValueError):
        concatDataFramesRows([])


def test_check_same_cols_amount():
    assert checkSameColsAmount([pd.DataFrame(np.zeros((2, 3))), pd.DataFrame(np.zeros((5, 3)))])

    with pytest.raises(ValueError):
        checkSameColsAmount([pd.DataFrame(np.zeros((2, 3))), pd.DataFrame(np.zeros((2, 4)))])


def test_concat_datas_with_labels_assigns_file_ids():
    data = pd.DataFrame({"x": [1, 2]})
    label = pd.DataFrame({"label": [0, 1]})

    out = PandasDataFrameLoader.concat_datas_with_labels([(data, label), (data, label)])

    assert out["file_id"].tolist() == [0, 0, 1, 1]


def test_concat_datas_with_labels_works_for_single_file():
    out = PandasDataFrameLoader.concat_datas_with_labels(
        [(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"label": [0, 1]}))]
    )

    assert out["file_id"].tolist() == [0, 0]


def test_loaders_read_written_files(tmp_path):
    data_path = tmp_path / "d.csv"
    labels_path = tmp_path / "l.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(data_path, index=False)
    pd.DataFrame({"label": [0, 1]}).to_csv(labels_path, index=False)

    loaded = PandasDataFrameLoader.load_data_labels_dfs([(data_path, labels_path)])

    assert len(loaded) == 1
    assert loaded[0][0]["x"].tolist() == [1, 2]
    assert loaded[0][1]["label"].tolist() == [0, 1]
