"""Testy CLI - pelna sciezka prepare -> train -> predict na danych syntetycznych."""
import json

import numpy as np
import pandas as pd
import pytest

from accur8pool.cli.main import build_parser, main
from accur8pool.data_processing.prepare_raw_data import REQUIRED_COLUMNS

PHASE_LEN = 60
N_ROWS = 1200


def write_dataset(root, n_files=5, seed=1):
    """Sygnal z powtarzajacymi sie fazami 0..4, latwy do nauczenia."""
    rng = np.random.default_rng(seed)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "labeled").mkdir(parents=True, exist_ok=True)

    for index in range(n_files):
        phase = np.repeat(np.tile([0, 1, 2, 3, 4], 4), PHASE_LEN)[:N_ROWS]
        df = pd.DataFrame(
            {col: rng.normal(size=N_ROWS) * 0.3 + phase for col in sorted(REQUIRED_COLUMNS - {"timestamp"})}
        )
        df["timestamp"] = 10.0
        df.to_csv(root / "raw" / f"s{index}.csv", index=False)
        pd.DataFrame({"label": phase}).to_csv(root / "labeled" / f"s{index}.csv", index=False)

    return root


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = write_dataset(tmp_path_factory.mktemp("data"))

    assert main(["prepare", "-i", str(root / "raw"), "-o", str(root / "prepared")]) == 0
    assert main([
        "train", "--root-dir", str(root),
        "-o", str(root / "model.pkl"),
        "--n-estimators", "40",
    ]) == 0

    return root


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_prepare_writes_transformed_files(trained):
    prepared = sorted((trained / "prepared").glob("*.csv"))

    assert len(prepared) == 5
    assert "acc_magnitude" in pd.read_csv(prepared[0]).columns


def test_train_writes_model_and_summary(trained):
    summary = json.loads((trained / "model.json").read_text())

    assert (trained / "model.pkl").exists()
    assert summary["data"] == {
        "files": 5,
        "rows": 5 * N_ROWS,
        "split_by": "file",
        "train_frac": 0.7,
        "test_frac": 0.2,
    }
    assert summary["metrics"]["f1_weighted"] > 0.8
    assert summary["window_params"] == {"window_size": 20, "window_step": 3}


def test_predict_labels_every_row_and_matches_truth(trained, tmp_path):
    out_dir = tmp_path / "pred"

    assert main([
        "predict", "-m", str(trained / "model.pkl"),
        "--input-files", str(trained / "raw" / "s0.csv"),
        "-o", str(out_dir),
    ]) == 0

    predicted = pd.read_csv(out_dir / "s0.csv")["label"].to_numpy()
    truth = pd.read_csv(trained / "labeled" / "s0.csv")["label"].to_numpy()

    assert len(predicted) == len(truth)
    assert (predicted == truth).mean() > 0.8


def test_predict_on_prepared_input_with_data(trained, tmp_path):
    out_dir = tmp_path / "pred_prepared"

    assert main([
        "predict", "-m", str(trained / "model.pkl"),
        "--input-files", str(trained / "prepared" / "s1.csv"),
        "-o", str(out_dir), "--prepared", "--with-data",
    ]) == 0

    out = pd.read_csv(out_dir / "s1.csv")

    assert "acc_magnitude" in out.columns
    assert "label" in out.columns


def test_predict_reports_error_for_too_short_input(trained, tmp_path):
    short = tmp_path / "short.csv"
    pd.read_csv(trained / "prepared" / "s0.csv").head(5).to_csv(short, index=False)

    assert main([
        "predict", "-m", str(trained / "model.pkl"),
        "--input-files", str(short), "-o", str(tmp_path / "out"), "--prepared",
    ]) == 1


def test_train_rejects_conflicting_inputs(trained):
    assert main(["train", "--data-dir", str(trained / "prepared")]) == 2


def test_prepare_rejects_both_input_forms(tmp_path):
    assert main([
        "prepare", "-i", str(tmp_path), "--input-files", "a.csv", "-o", str(tmp_path / "o"),
    ]) == 2


def test_unknown_paths_are_reported_not_raised(tmp_path):
    assert main(["prepare", "-i", str(tmp_path / "nope"), "-o", str(tmp_path / "o")]) == 2
