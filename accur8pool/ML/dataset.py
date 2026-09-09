"""Skladanie zbioru treningowego z plikow prepared + labeled."""
from __future__ import annotations

import pandas as pd

from accur8pool.data_processing.const import DATA_COLUMNS_TO_TRAIN, LABEL
from accur8pool.files_utils.dataFrameLoader import PandasDataFrameLoader
from accur8pool.files_utils.filesManager import FilesManager

FILE_ID = 'file_id'


def build_training_frame(
        files_manager: FilesManager,
        label_column: str = LABEL,
        fill_label: float = 0.0,
) -> pd.DataFrame:
    """
    Laczy wszystkie sparowane pliki danych i etykiet w jedna ramke.

    Kazdy plik dostaje wlasne file_id, dzieki czemu podzial na train/test/eval
    moze isc po plikach i nie przeciekac miedzy zbiorami.
    """
    data_label_paths = files_manager.get_data_label_paths()

    if not data_label_paths:
        raise ValueError('Nie znaleziono ani jednej pary plik danych - plik etykiet')

    unlabeled = files_manager.get_unlabeled_data()
    if unlabeled:
        print(f'UWAGA pomijam {len(unlabeled)} plikow bez etykiet, np. {unlabeled[0]}')

    frames = PandasDataFrameLoader.load_data_labels_dfs(data_label_paths)
    concated = PandasDataFrameLoader.concat_datas_with_labels(frames)

    if label_column not in concated.columns:
        raise ValueError(f"Brak kolumny etykiet '{label_column}' w polaczonych danych")

    concated[label_column] = concated[label_column].fillna(fill_label)

    return concated


def split_features_labels(
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        label_column: str = LABEL,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = list(feature_columns or DATA_COLUMNS_TO_TRAIN)

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f'Brakujace kolumny cech: {missing}')

    labels = df[label_column]
    if labels.isna().any():
        raise ValueError(f"Kolumna '{label_column}' zawiera NaN")

    return df[feature_columns].copy(), labels.astype(int)


def split_by_file(
        df: pd.DataFrame,
        train_frac: float = 0.7,
        test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dzieli po file_id - caly plik trafia do jednego zbioru.

    Podzial po numerze wiersza rozcina pojedynczy plik miedzy train i test,
    przez co okna z sasiadujacych probek trafiaja do obu zbiorow.
    """
    _validate_fracs(train_frac, test_frac)

    if FILE_ID not in df.columns:
        raise ValueError(f"Brak kolumny '{FILE_ID}' - uzyj split_by_index")

    file_ids = df[FILE_ID].drop_duplicates().tolist()

    if len(file_ids) < 3:
        raise ValueError(
            f'Podzial po plikach wymaga co najmniej 3 plikow, jest {len(file_ids)}. '
            f'Uzyj --split-by index.'
        )

    n = len(file_ids)
    # kazdy z trzech zbiorow musi dostac co najmniej jeden plik
    train_end = min(max(1, round(n * train_frac)), n - 2)
    test_end = min(max(train_end + 1, round(n * (train_frac + test_frac))), n - 1)

    groups = (file_ids[:train_end], file_ids[train_end:test_end], file_ids[test_end:])

    return tuple(df[df[FILE_ID].isin(group)] for group in groups)


def split_by_index(
        df: pd.DataFrame,
        train_frac: float = 0.7,
        test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Podzial po numerze wiersza - tak jak w notebooku xgboostv2."""
    _validate_fracs(train_frac, test_frac)

    n = len(df)
    train_end = min(max(1, round(n * train_frac)), n - 2)
    test_end = min(max(train_end + 1, round(n * (train_frac + test_frac))), n - 1)

    return df[:train_end], df[train_end:test_end], df[test_end:]


def _validate_fracs(train_frac: float, test_frac: float) -> None:
    if not 0 < train_frac < 1:
        raise ValueError(f'train_frac musi byc w (0, 1), jest {train_frac}')

    if not 0 < test_frac < 1:
        raise ValueError(f'test_frac musi byc w (0, 1), jest {test_frac}')

    if train_frac + test_frac >= 1:
        raise ValueError(
            f'train_frac + test_frac musi byc < 1 (reszta idzie na eval), '
            f'jest {train_frac + test_frac}'
        )
