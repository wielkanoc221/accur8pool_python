"""Interfejs wiersza polecen accur8pool: przygotowanie danych, trening, predykcja."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from accur8pool.data_processing.const import DATA_COLUMNS_TO_TRAIN, LABEL
from accur8pool.data_processing.prepare_raw_data import (
    prepare_raw_data_and_save,
    transform_raw_df,
)
from accur8pool.files_utils.filesManager import FilesManager
from accur8pool.ML.dataset import (
    build_training_frame,
    split_by_file,
    split_by_index,
    split_features_labels,
)
from accur8pool.ML.pipelines import PipelineInit, XGBPipeline, make_pipeline

DEFAULT_MODEL_PATH = 'model.pkl'


# ============================================================
# prepare
# ============================================================

def run_prepare(args: argparse.Namespace) -> int:
    input_paths = _collect_input_paths(args.input_dir, args.input_files)
    ok_count = prepare_raw_data_and_save(input_paths, Path(args.output_dir))

    return 0 if ok_count == len(input_paths) else 1


# ============================================================
# train
# ============================================================

def run_train(args: argparse.Namespace) -> int:
    files_manager = _make_files_manager(args)

    print('Skladanie zbioru treningowego...')
    frame = build_training_frame(files_manager, label_column=args.label_column)
    print(f'  wierszy: {len(frame)}, plikow: {frame["file_id"].nunique()}')

    splitter = split_by_file if args.split_by == 'file' else split_by_index
    train_df, test_df, eval_df = splitter(frame, args.train_frac, args.test_frac)
    print(f'  podzial ({args.split_by}): train={len(train_df)} test={len(test_df)} eval={len(eval_df)}')

    feature_columns = list(DATA_COLUMNS_TO_TRAIN)
    X_train, y_train = split_features_labels(train_df, feature_columns, args.label_column)
    X_test, y_test = split_features_labels(test_df, feature_columns, args.label_column)
    X_eval, y_eval = split_features_labels(eval_df, feature_columns, args.label_column)

    num_class = args.num_class or int(y_train.max()) + 1
    _validate_classes(y_train, num_class)

    pipeline = make_pipeline(PipelineInit(
        model_params={
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'subsample': args.subsample,
        },
        window_size=args.window_size,
        window_step=args.window_step,
        model_n_estimators=args.n_estimators,
        model_early_stopping_rounds=args.early_stopping_rounds,
        num_class=num_class,
    ))
    pipeline.DATA_COLUMNS = feature_columns

    print(f'Trening (okno={args.window_size}, krok={args.window_step}, klas={num_class})...')
    pipeline.fit_prepared(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        summary_path=None,
        verbose=args.verbose,
    )

    metrics = pipeline.evaluate_prepared(X_test, y_test)
    print(f'\nWynik na zbiorze testowym ({metrics["n_windows"]} okien):')
    print(f'  f1_weighted       = {metrics["f1_weighted"]:.4f}')
    print(f'  balanced_accuracy = {metrics["balanced_accuracy"]:.4f}')

    model_path = Path(args.output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_model(model_path)
    print(f'\nModel zapisany: {model_path}')

    summary_path = Path(args.summary) if args.summary else model_path.with_suffix('.json')
    summary = pipeline._make_fit_summary(summary_path)
    summary['metrics'] = metrics
    summary['data'] = {
        'files': frame['file_id'].nunique(),
        'rows': len(frame),
        'split_by': args.split_by,
        'train_frac': args.train_frac,
        'test_frac': args.test_frac,
    }
    summary_path.write_text(json.dumps(summary, indent=4, default=str), encoding='utf-8')
    print(f'Podsumowanie zapisane: {summary_path}')

    return 0


# ============================================================
# predict
# ============================================================

def run_predict(args: argparse.Namespace) -> int:
    pipeline = XGBPipeline.load_model(args.model)
    input_paths = _collect_input_paths(args.input_dir, args.input_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    for index, path in enumerate(input_paths, start=1):
        print(f'{index}/{len(input_paths)} {path}')
        try:
            df = pd.read_csv(path, engine='pyarrow')
            prepared = df if args.prepared else transform_raw_df(df)
            labels = pipeline.predict_prepared(prepared, min_group_size=args.min_group_size)

            out = prepared.copy() if args.with_data else pd.DataFrame(index=prepared.index)
            out[args.label_column] = labels.astype(int)
            out.to_csv(output_dir / path.name, index=False)
            ok_count += 1
        except Exception as exc:
            print(f'  ERROR {type(exc).__name__}: {exc}')

    print(f'przetworzono {ok_count}/{len(input_paths)} plikow -> {output_dir}')

    return 0 if ok_count == len(input_paths) else 1


# ============================================================
# Helpers
# ============================================================

def _collect_input_paths(input_dir: str | None, input_files: list[str] | None) -> list[Path]:
    if input_dir and input_files:
        raise ValueError('Podaj albo --input-dir albo --input-files, nie oba')

    if input_dir:
        paths = FilesManager.get_csv_paths(input_dir)
    elif input_files:
        paths = [Path(p) for p in input_files]
    else:
        raise ValueError('Musisz podac --input-dir albo --input-files')

    if not paths:
        raise ValueError('Nie znaleziono zadnego pliku .csv')

    return paths


def _make_files_manager(args: argparse.Namespace) -> FilesManager:
    if args.root_dir:
        return FilesManager(
            root_dir=args.root_dir,
            data_subdir=args.data_subdir,
            labels_subdir=args.labels_subdir,
        )

    if not (args.data_dir and args.labels_dir):
        raise ValueError('Podaj --root-dir albo --data-dir razem z --labels-dir')

    return FilesManager(
        data_paths=FilesManager.get_csv_paths(args.data_dir),
        labels_paths=FilesManager.get_csv_paths(args.labels_dir),
    )


def _validate_classes(y_train: pd.Series, num_class: int) -> None:
    present = sorted(y_train.unique().tolist())
    expected = list(range(num_class))

    if present != expected:
        raise ValueError(
            f'XGBoost wymaga klas 0..{num_class - 1} w zbiorze treningowym, '
            f'a sa {present}. Zmien podzial danych albo --num-class.'
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='accur8pool',
        description='Detekcja i klasyfikacja faz uderzenia bilardowego z danych IMU',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ---- prepare ----
    prepare = subparsers.add_parser(
        'prepare',
        help='Przygotowanie surowych danych (filtracja, magnitude, jerk, roll, pitch)',
    )
    _add_input_args(prepare)
    prepare.add_argument('-o', '--output-dir', default='prepared', help='Katalog zapisu')
    prepare.set_defaults(func=run_prepare)

    # ---- train ----
    train = subparsers.add_parser('train', help='Trening modelu na danych prepared + labeled')
    train.add_argument('--root-dir', help='Katalog z podkatalogami prepared/ i labeled/')
    train.add_argument('--data-dir', help='Katalog z przygotowanymi danymi (zamiast --root-dir)')
    train.add_argument('--labels-dir', help='Katalog z etykietami (zamiast --root-dir)')
    train.add_argument('--data-subdir', default='prepared', help='Podkatalog danych w --root-dir')
    train.add_argument('--labels-subdir', default='labeled', help='Podkatalog etykiet w --root-dir')
    train.add_argument('-o', '--output', default=DEFAULT_MODEL_PATH, help='Sciezka zapisu modelu')
    train.add_argument('--summary', help='Sciezka pliku json z podsumowaniem (domyslnie obok modelu)')
    train.add_argument('--label-column', default=LABEL, help='Nazwa kolumny etykiet')
    train.add_argument(
        '--split-by',
        choices=('file', 'index'),
        default='file',
        help='file = caly plik w jednym zbiorze (domyslne), index = podzial po numerze wiersza',
    )
    train.add_argument('--train-frac', type=float, default=0.7)
    train.add_argument('--test-frac', type=float, default=0.2)
    train.add_argument('--window-size', type=int, default=20)
    train.add_argument('--window-step', type=int, default=3)
    train.add_argument('--n-estimators', type=int, default=2000)
    train.add_argument('--early-stopping-rounds', type=int, default=20)
    train.add_argument('--max-depth', type=int, default=6)
    train.add_argument('--learning-rate', type=float, default=0.1)
    train.add_argument('--subsample', type=float, default=0.8)
    train.add_argument('--num-class', type=int, help='Domyslnie wykrywane z danych')
    train.add_argument('--verbose', type=int, default=0, help='Co ile rund logowac postep XGBoost')
    train.set_defaults(func=run_train)

    # ---- predict ----
    predict = subparsers.add_parser('predict', help='Etykietowanie danych wyuczonym modelem')
    predict.add_argument('-m', '--model', default=DEFAULT_MODEL_PATH, help='Sciezka modelu .pkl')
    _add_input_args(predict)
    predict.add_argument('-o', '--output-dir', default='predicted', help='Katalog zapisu wynikow')
    predict.add_argument(
        '--prepared',
        action='store_true',
        help='Wejscie jest juz po transformacji (pomija transform_raw_df)',
    )
    predict.add_argument(
        '--with-data',
        action='store_true',
        help='Zapisz kolumny danych obok etykiet (domyslnie sama etykieta)',
    )
    predict.add_argument('--label-column', default=LABEL, help='Nazwa kolumny wyjsciowej')
    predict.add_argument('--min-group-size', type=int, default=6, help='Wygladzanie krotkich segmentow')
    predict.set_defaults(func=run_predict)

    return parser


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-i', '--input-dir', help='Katalog z plikami .csv (rekurencyjnie)')
    parser.add_argument('--input-files', nargs='*', help='Konkretne pliki .csv')


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        return args.func(args)
    except Exception as exc:
        print(f'ERROR {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
