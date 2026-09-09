from pathlib import Path
from typing import Sequence

import pandas as pd

PathPair = tuple[str | Path, str | Path]


def concat_datas_and_labels_from_paths(label_data_paths: Sequence[PathPair]) -> pd.DataFrame:
    dfs = []
    for data_path, label_path in label_data_paths:
        Xdf = pd.read_csv(data_path, engine='pyarrow')
        ydf = pd.read_csv(label_path, engine='pyarrow')
        dfs.append(concat_data_with_label(Xdf, ydf))

    if not dfs:
        raise ValueError('label_data_paths jest puste')

    return concatDataFramesRows(dfs)


def concat_data_with_label(data: pd.DataFrame, label: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    funkcja laczy kolumny danych wraz z label
    data = pd.DataFrame({'x': [1,2,3], 'y': [1,2,3]})
    label = pd.Series([1,2,3], name='label')
    result = pd.DataFrame({'x': [1,2,3], 'y': [1,2,3], 'label': [1,2,3]})
    :param data:
    :param label:
    :return:
    """
    if data.shape[0] != label.shape[0]:
        raise ValueError(f'Dane i label maja rozna ilosc wierszy  dane: {data.shape[0]}, label: {label.shape[0]}')

    # pd.concat laczy po indeksie - bez resetu rozjechane indeksy daja ciche NaN-y
    return pd.concat(
        [data.reset_index(drop=True), label.reset_index(drop=True)],
        axis=1,
    )


def concatDataFramesRows(dataFrames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    laczy liste DataFrame względem wierszy
    :param dataFrames:
    :return:
    """
    if not dataFrames:
        raise ValueError('lista DataFramow nie moze byc pusta')

    return pd.concat(dataFrames, axis=0, ignore_index=True)


def checkSameColsAmount(dataFrames: Sequence[pd.DataFrame]) -> bool:
    if not dataFrames:
        raise ValueError('lista DataFramow nie moze byc pusta')

    expected_cols = dataFrames[0].shape[1]
    for dataFrame in dataFrames[1:]:
        if dataFrame.shape[1] != expected_cols:
            raise ValueError('Nieprawidlowa liczba kolumn')

    return True


if __name__ == '__main__':
    import numpy as np

    dfs = [pd.DataFrame(np.random.rand(3, 3)), pd.DataFrame(np.random.rand(5, 3))]
    labels = [pd.Series(np.random.rand(3), name='label'), pd.Series(np.random.rand(5), name='label')]

    print(checkSameColsAmount(dfs))
    merged = [concat_data_with_label(df, label) for df, label in zip(dfs, labels)]
    print(concatDataFramesRows(merged).shape)
