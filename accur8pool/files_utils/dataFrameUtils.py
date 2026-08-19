from pathlib import Path

import pandas as pd


def concat_datas_and_labels_from_paths(label_data_paths: list[Path, str], ):
    dfs = []
    for data_path, label_path in label_data_paths:
        Xdf = pd.read_csv(data_path, engine='pyarrow')
        ydf = pd.read_csv(label_path, engine='pyarrow')
        merged = pd.concat([Xdf, ydf], axis=1)
        dfs.append(merged)

    df = pd.concat(dfs, ignore_index=True)
    return df


def concat_data_with_label(data: pd.DataFrame, label: pd.Series | pd.DataFrame):
    """
    funkcja laczy kolumny data1 wraz z label
    data1 = pd.Dataframe({'x'=[1,2,3],'y'=[1,2,3]})
    label = pd.Series([1,2,3],name='label')
    result = pd.Dataframe({'x'=[1,2,3],'y'=[1,2,3],'label'=[1,2,3]})
    :param data:
    :param label:
    :return:
    """
    if data.shape[0] != label.shape[0]:
        raise ValueError(f'Dane i label maja rozna ilosc wierszy  dane: {data.shape[0]}, label: {label.shape[0]}')

    return pd.concat([data, label], axis=1)


def concatDataFramesRows(dataFrames: list[pd.DataFrame]):
    """
    laczy liste DataFrame względem wierszy
    :param dataFrames:
    :return:
    """
    if len(dataFrames) <= 1:
        raise ValueError('lista DataFramow musi miec conajmniej 2 elementy')

    return pd.concat(dataFrames, axis=0, ignore_index=True)


# def checkSameColsAmount(dataFrames: list[pd.DataFrame]):
#     rev_cols = dataFrames[0].shape[1]
#     for dataFrame in dataFrames[1:]:
#         if dataFrame.shape[1] != rev_cols:
#             raise ValueError('Nieprawidlowa liczba kolumn')
#
#     return True


if __name__ == '__main__':
    import numpy as np

    lalels = [pd.Series(np.random.rand(3)), pd.Series(np.random.rand(5))]
    dfs = [pd.DataFrame(np.random.rand(3, 3)), pd.DataFrame(np.random.rand(5, 3))]
    print(checkSameColsAmount(dfs))
    x = concat_data_with_label(dfs, lalels)
    print(concatDataFramesRows(dfs).shape)
