from pathlib import Path
from typing import Iterable, Iterator, TypeAlias

import pandas as pd
import polars as pl

from accur8pool.files_utils.dataFrameUtils import concat_data_with_label, concatDataFramesRows

PathPair: TypeAlias = tuple[str | Path, str | Path]
PathPairs: TypeAlias = list[PathPair]
FilePaths: TypeAlias = Iterable[str | Path]


class PandasDataFrameLoader:


    @staticmethod
    def concat_datas_with_labels(data_labels_list: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
        """
        funkcja tworzy pojedynczy dataFrame laczac data1 z etkietami
        oraz nadajaca kazdemu plikowi identyfikator w celu pozniejszemu
        splitowi. Nadanie identyfikatorow ma zapobiec wycieku danych do danych
        testowych
        :param data_labels_list: lista tupli (dataDataFrame,labelsDataFrame)
        :return: jeden dataframe z danymi oraz etykietami
        """

        if not data_labels_list:
            raise ValueError('data_labels_list jest puste')

        concated_data_labels_list = []
        for df_index, (data, labels) in enumerate(data_labels_list):
            concated_df = concat_data_with_label(data, labels)
            concated_df['file_id'] = df_index
            concated_data_labels_list.append(concated_df)

        return concatDataFramesRows(concated_data_labels_list)

    @staticmethod
    def generate_data_labels_dfs(data_label_paths: PathPairs) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        for data_path, labels_path in data_label_paths:
            data_df = pd.read_csv(data_path, engine="pyarrow")
            labels_df = pd.read_csv(labels_path, engine="pyarrow")

            yield data_df, labels_df

    @staticmethod
    def generate_dataframes(files_paths: FilePaths) -> Iterator[pd.DataFrame]:
        for file_path in files_paths:
            yield pd.read_csv(file_path, engine="pyarrow")

    @classmethod
    def load_data_labels_dfs(cls, data_label_paths: PathPairs) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        return list(cls.generate_data_labels_dfs(data_label_paths))

    @classmethod
    def load_dataframes(cls, files_paths: FilePaths) -> list[pd.DataFrame]:
        return list(cls.generate_dataframes(files_paths))


class PolarsDataFrameLoader:
    @staticmethod
    def generate_data_labels_dfs(data_label_paths: PathPairs) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        for data_path, labels_path in data_label_paths:
            data_df = pl.read_csv(data_path)
            labels_df = pl.read_csv(labels_path)

            yield data_df, labels_df

    @staticmethod
    def generate_dataframes(files_paths: FilePaths) -> Iterator[pl.DataFrame]:
        for file_path in files_paths:
            yield pl.read_csv(file_path)

    @classmethod
    def load_data_labels_dfs(cls, data_label_paths: PathPairs) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
        return list(cls.generate_data_labels_dfs(data_label_paths))

    @classmethod
    def load_dataframes(cls, files_paths: FilePaths) -> list[pl.DataFrame]:
        return list(cls.generate_dataframes(files_paths))


if __name__ == '__main__':
    datas = {'d1': [1, 2, 3], 'd2': [4, 5, 6]}
    labels = {'labels': [0, 0, 1], 'labels_idx': [10, 11, 12]}
    d1 = pd.DataFrame(datas)
    d2 = pd.DataFrame(datas)
    l1 = pd.DataFrame(labels)
    l2 = pd.DataFrame(labels)

    full = [(d1, l1), (d2, l2)]
    x = PandasDataFrameLoader.concat_datas_with_labels(full)
    print(x.shape)
    print(x)
