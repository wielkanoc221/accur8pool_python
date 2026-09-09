import numpy as np
import pandas as pd
import pytest

from accur8pool.data_processing.const import DATA_COLUMNS_TO_TRAIN
from accur8pool.data_processing.dataWindowing import (
    CenterLabelingStrategy,
    DataWindowing,
    LabelMajorityStrategy,
)
from accur8pool.files_utils.filesManager import FilesManager
from accur8pool.ML.dataset import (
    build_training_frame,
    split_by_file,
    split_by_index,
    split_features_labels,
)


def write_pair(root, name, rows, label_value=0):
    for subdir, frame in (
        ("prepared", pd.DataFrame({col: np.zeros(rows) for col in DATA_COLUMNS_TO_TRAIN})),
        ("labeled", pd.DataFrame({"label": [label_value] * rows})),
    ):
        (root / subdir).mkdir(parents=True, exist_ok=True)
        frame.to_csv(root / subdir / name, index=False)


@pytest.fixture
def root(tmp_path):
    for index in range(4):
        write_pair(tmp_path, f"s{index}.csv", rows=10, label_value=index % 2)

    return tmp_path


def test_build_training_frame_assigns_file_ids(root):
    frame = build_training_frame(FilesManager(root_dir=root))

    assert len(frame) == 40
    assert frame["file_id"].nunique() == 4


def test_build_training_frame_needs_pairs(tmp_path):
    (tmp_path / "prepared").mkdir()
    (tmp_path / "labeled").mkdir()

    with pytest.raises(ValueError, match="ani jednej pary"):
        build_training_frame(FilesManager(root_dir=tmp_path))


def test_split_by_file_keeps_files_whole(root):
    frame = build_training_frame(FilesManager(root_dir=root))
    train, test, evaluation = split_by_file(frame, 0.5, 0.25)

    ids = [set(part["file_id"]) for part in (train, test, evaluation)]

    assert set.union(*ids) == {0, 1, 2, 3}
    assert not ids[0] & ids[1] and not ids[1] & ids[2] and not ids[0] & ids[2]


def test_split_by_file_needs_enough_files(tmp_path):
    write_pair(tmp_path, "only.csv", rows=10)

    frame = build_training_frame(FilesManager(root_dir=tmp_path))

    with pytest.raises(ValueError, match="co najmniej 3 plikow"):
        split_by_file(frame)


def test_split_by_index_matches_notebook_proportions():
    frame = pd.DataFrame({"x": range(100)})
    train, test, evaluation = split_by_index(frame, 0.7, 0.2)

    assert (len(train), len(test), len(evaluation)) == (70, 20, 10)


@pytest.mark.parametrize("train_frac,test_frac", [(0.9, 0.2), (0.0, 0.5), (0.5, 1.5)])
def test_split_rejects_invalid_fractions(train_frac, test_frac):
    frame = pd.DataFrame({"x": range(10)})

    with pytest.raises(ValueError):
        split_by_index(frame, train_frac, test_frac)


def test_split_features_labels_casts_to_int(root):
    frame = build_training_frame(FilesManager(root_dir=root))
    X, y = split_features_labels(frame)

    assert list(X.columns) == list(DATA_COLUMNS_TO_TRAIN)
    assert y.dtype.kind == "i"


def test_split_features_labels_reports_missing_columns():
    with pytest.raises(ValueError, match="Brakujace kolumny cech"):
        split_features_labels(pd.DataFrame({"label": [0]}))


def test_center_strategy_labels_are_written_at_window_centre():
    """Bez offsetu wynik byl przesuniety o polowe okna wzgledem wejscia."""
    windowing = DataWindowing(window_size=20, window_step=1, window_labeling_strategy=CenterLabelingStrategy())
    y = np.array([0] * 50 + [1] * 50)
    X = np.repeat(y, 2).reshape(-1, 2).astype(float)

    _, y_win = windowing.window(X, y)
    restored = windowing.reverse_window_labels(y_win, len(y))

    assert (restored == y).mean() > 0.95


def test_majority_strategy_keeps_full_window_span():
    windowing = DataWindowing(window_size=4, window_step=4, window_labeling_strategy=LabelMajorityStrategy())

    restored = windowing.reverse_window_labels([1, 0], y_len=8)

    assert restored.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("n_files,expected", [(3, [1, 1, 1]), (4, [2, 1, 1]), (5, [3, 1, 1]), (10, [7, 2, 1])])
def test_split_by_file_never_leaves_a_set_empty(n_files, expected):
    """round() na granicach potrafil zostawic pusty zbior testowy."""
    frame = pd.DataFrame({"file_id": range(n_files)})

    assert [len(part) for part in split_by_file(frame, 0.7, 0.2)] == expected
