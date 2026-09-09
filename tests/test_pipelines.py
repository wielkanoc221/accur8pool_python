import numpy as np
import pytest

from accur8pool.data_processing.dataWindowing import DataWindowing
from accur8pool.ML.pipelines import PipelineInit, XGBPipeline, make_pipeline


def test_pipeline_init_defaults_are_scalars():
    """Zwisajace przecinki w dataclass robily z defaultow jednoelementowe tuple."""
    init = PipelineInit(model_params={})

    assert init.eval_metric == "mlogloss"
    assert init.objective == "multi:softprob"
    assert init.num_class == 5


def test_make_pipeline_passes_scalar_params_to_model():
    model_params = make_pipeline(PipelineInit(model_params={"max_depth": 3})).model.get_params()

    assert model_params["objective"] == "multi:softprob"
    assert model_params["eval_metric"] == "mlogloss"
    assert model_params["max_depth"] == 3


def test_save_and_load_model_roundtrip(tmp_path):
    pipeline = make_pipeline(PipelineInit(model_params={}))
    path = tmp_path / "pipeline.pkl"

    pipeline.save_model(path)
    loaded = XGBPipeline.load_model(path)

    assert isinstance(loaded, XGBPipeline)
    assert loaded.windowing.get_params() == pipeline.windowing.get_params()


def test_load_model_rejects_other_objects(tmp_path):
    import pickle

    path = tmp_path / "not_a_pipeline.pkl"
    path.write_bytes(pickle.dumps({"foo": "bar"}))

    with pytest.raises(TypeError):
        XGBPipeline.load_model(path)


def test_make_fit_summary_writes_to_given_path(tmp_path):
    import json

    pipeline = make_pipeline(PipelineInit(model_params={}))
    path = tmp_path / "config.json"

    summary = pipeline._make_fit_summary(path)

    assert json.loads(path.read_text())["window_params"] == summary["window_params"]


def test_smooth_replaces_short_groups():
    y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    assert XGBPipeline.smooth(y_pred, min_group_size=3).tolist() == [0] * 9


def test_sqrt_balanced_weights_favour_minority_class():
    weights = XGBPipeline.sqrt_balanced_weights(np.array([0] * 9 + [1]))

    assert weights[-1] > weights[0]


def test_windowing_roundtrip():
    windowing = DataWindowing(window_size=4, window_step=2)
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.zeros(20)

    X_win, y_win = windowing.window(X, y)

    assert X_win.shape == (9, 4, 2)
    assert y_win.shape == (9,)
    assert windowing.reverse_window_labels(y_win, len(y)).shape == (20,)


def test_window_rejects_non_array_with_clear_error():
    """Sprawdzenie .shape bylo przed isinstance - lista dawala AttributeError."""
    with pytest.raises(ValueError, match="np.ndarray"):
        DataWindowing(window_size=2).window([[1, 2], [3, 4]], np.zeros(2))


def test_window_rejects_length_mismatch():
    with pytest.raises(ValueError, match="różnych rozmiarów"):
        DataWindowing(window_size=2).window(np.zeros((10, 2)), np.zeros(9))


def test_windowing_validates_params():
    with pytest.raises(ValueError):
        DataWindowing(window_size=0)

    with pytest.raises(ValueError):
        DataWindowing(window_step=0)
