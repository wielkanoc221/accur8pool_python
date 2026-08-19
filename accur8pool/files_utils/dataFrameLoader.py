from typing import TypeAlias
from pathlib import Path
import pandas as pd
import polars as pl


from accur8pool.files_utils.dataFrameUtils import concat_data_with_label, concatDataFramesRows

PathPair: TypeAlias = tuple[str | Path, str | Path]
PathPairs: TypeAlias = list[PathPair]


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

        df_index = 0
        concated_data_labels_list = []
        for data, labels in data_labels_list:
            concated_df = concat_data_with_label(data, labels)
            concated_df['file_id'] = df_index
            df_index += 1
            concated_data_labels_list.append(concated_df)

        return concatDataFramesRows(concated_data_labels_list)

    @staticmethod
    def generate_data_labels_dfs(data_label_paths: PathPairs):
        if data_label_paths:
            for data_path, labels_path in data_label_paths:
                data_df = pd.read_csv(data_path, engine="pyarrow")
                labels_df = pd.read_csv(labels_path, engine="pyarrow")

                yield data_df, labels_df

    @staticmethod
    def generate_dataframes(files_paths: Path | str):
        for file_path in files_paths:
            yield pd.read_csv(file_path)

    @staticmethod
    def load_data_labels_dfs(data_label_paths: PathPairs) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        data_dfs = []
        labels_dfs = []
        for data_path, labels_path in data_label_paths:
            data_dfs.append(pd.read_csv(data_path, engine="pyarrow"))
            labels_dfs.append(pd.read_csv(labels_path, engine="pyarrow"))

        return list(zip(data_dfs, labels_dfs))

    @staticmethod
    def load_dataframes(files_paths: Path | str):
        dfs = []
        for file_path in files_paths:
            dfs.append(pd.read_csv(file_path))

        return dfs


class PolarsDataFrameLoader:
    @staticmethod
    def generate_data_labels_dfs(data_label_paths: PathPairs):
        if data_label_paths:
            for data_path, labels_path in data_label_paths:
                data_df = pl.read_csv(data_path)
                labels_df = pl.read_csv(labels_path)

                yield data_df, labels_df

    @staticmethod
    def generate_dataframes(files_paths: Path | str):
        files_paths = [Path(file_path) for file_path in files_paths]

        for file_path in files_paths:
            yield pl.read_csv(file_path)


if __name__ == '__main__':
    from accur8pool.files_utils.filesManager import FilesManager
    import time

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
