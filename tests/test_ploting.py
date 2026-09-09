import numpy as np
import pandas as pd
import pytest

from accur8pool.data_processing.const import DATA_COLUMNS_TO_TRAIN
from accur8pool.scripts.ploting import plot_prepared_data, to_segments


def make_prepared_df(n=20):
    rng = np.random.default_rng(0)

    return pd.DataFrame({col: rng.normal(size=n) for col in DATA_COLUMNS_TO_TRAIN})


def test_to_segments_covers_whole_signal():
    segments = to_segments([0, 0, 1, 1, 1, 2])

    assert [(start, end) for start, end, _ in segments] == [(0, 2), (2, 5), (5, 6)]
    assert [label for _, _, label in segments] == [0, 1, 2]


def test_to_segments_has_no_empty_segment_when_signal_starts_with_change():
    """np.roll porownywal pierwszy element z ostatnim i tworzyl pusty segment (0, 0)."""
    assert all(end > start for start, end, _ in to_segments([1, 0, 0]))


def test_to_segments_on_empty_input():
    assert to_segments([]) == []


def test_plot_without_labels_does_not_raise():
    """Kolumna 'label' byla czytana przed sprawdzeniem czy labels_data_frame is None."""
    fig = plot_prepared_data(make_prepared_df(), labels_data_frame=None)

    assert fig is not None


def test_plot_with_labels_adds_vlines():
    labels = pd.DataFrame({"label": [0] * 10 + [1] * 10})

    fig = plot_prepared_data(make_prepared_df(), labels_data_frame=labels)

    assert len(fig.layout.shapes) == 2


def test_plot_reports_missing_columns():
    with pytest.raises(ValueError, match="Brakujace kolumny"):
        plot_prepared_data(pd.DataFrame({"accx": [1.0]}))


def test_plot_handles_label_outside_palette():
    labels = pd.DataFrame({"label": [99] * 20})

    assert plot_prepared_data(make_prepared_df(), labels_data_frame=labels) is not None
