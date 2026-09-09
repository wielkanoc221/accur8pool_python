import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from accur8pool.data_processing.dataWindowing import CenterLabelingStrategy, DataWindowing, extractWindowFeatures
from accur8pool.data_processing.prepare_raw_data import transform_raw_df


class XGBPipeline:
    DATA_COLUMNS = ['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz', 'magx', 'magy', 'magz',
                    'linaccx', 'linaccy', 'linaccz', 'rotx', 'roty', 'rotz',
                    'acc_magnitude', 'gyr_magnitude', 'jerk_accx', 'jerk_accy',
                    'jerk_accz', 'acc_magnitude_jerk', 'jerk_gyrx', 'jerk_gyry',
                    'jerk_gyrz', 'gyr_magnitude_jerk', 'roll', 'pitch']

    LABEL_COLUMN = 'label'

    def __init__(self, windowing: DataWindowing, model):
        self.windowing: DataWindowing = windowing
        self.model: XGBClassifier = model

    @staticmethod
    def load_model(path: str | Path) -> "XGBPipeline":
        with open(path, "rb") as f:
            pipeline = pickle.load(f)

        if not isinstance(pipeline, XGBPipeline):
            raise TypeError(f"{path} nie zawiera XGBPipeline, tylko {type(pipeline)}")

        return pipeline

    def fit(self, X_data, y_data: pd.Series, summary_path: str | Path = "pipeline_config.json"):
        """Uczy na SUROWYCH danych - transformacja jest robiona w srodku."""
        X_data = transform_raw_df(X_data)[self.DATA_COLUMNS]
        y_data = y_data.fillna(0)
        stop_idx_train = int(len(X_data) * 0.8)

        self.fit_prepared(
            X_train=X_data[:stop_idx_train],
            y_train=y_data[:stop_idx_train],
            X_eval=X_data[stop_idx_train:],
            y_eval=y_data[stop_idx_train:],
            summary_path=summary_path,
        )

    def fit_prepared(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_eval: pd.DataFrame,
            y_eval: pd.Series,
            summary_path: str | Path | None = "pipeline_config.json",
            verbose: bool | int = 2,
    ) -> None:
        """Uczy na danych juz przepuszczonych przez transform_raw_df (katalog prepared)."""
        X_train_windowed, y_train_windowed = self.windowing.window_with_extractions(
            X_train[self.DATA_COLUMNS].to_numpy(), np.asarray(y_train)
        )
        X_eval_windowed, y_eval_windowed = self.windowing.window_with_extractions(
            X_eval[self.DATA_COLUMNS].to_numpy(), np.asarray(y_eval)
        )

        if len(X_train_windowed) == 0:
            raise ValueError(
                f'Zbior treningowy jest krotszy niz okno ({self.windowing.window_size} probek)'
            )

        weights = self.sqrt_balanced_weights(y_train_windowed)

        self.model.fit(
            X_train_windowed,
            y_train_windowed,
            eval_set=[(X_eval_windowed, y_eval_windowed)],
            verbose=verbose,
            sample_weight=weights,
        )

        if summary_path is not None:
            self._make_fit_summary(summary_path)

    def evaluate_prepared(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Metryki liczone na poziomie okien, tak jak uczony jest model."""
        X_windowed, y_windowed = self.windowing.window_with_extractions(
            X[self.DATA_COLUMNS].to_numpy(), np.asarray(y)
        )

        if len(X_windowed) == 0:
            raise ValueError(f'Zbior jest krotszy niz okno ({self.windowing.window_size} probek)')

        y_pred = self.model.predict(X_windowed)

        return {
            'n_windows': int(len(y_windowed)),
            'f1_weighted': float(f1_score(y_true=y_windowed, y_pred=y_pred, average='weighted')),
            'balanced_accuracy': float(balanced_accuracy_score(y_true=y_windowed, y_pred=y_pred)),
            'per_class': classification_report(
                y_true=y_windowed, y_pred=y_pred, output_dict=True, zero_division=0
            ),
        }

    def predict(self, X_data: pd.DataFrame, min_group_size: int = 6):
        """Predykcja na SUROWYCH danych - transformacja jest robiona w srodku."""
        return self.predict_prepared(transform_raw_df(X_data), min_group_size=min_group_size)

    def predict_prepared(self, prepared: pd.DataFrame, min_group_size: int = 6):
        """
        Predykcja na danych juz przygotowanych.

        Zwraca etykiete dla kazdego wiersza wejscia (okna sa rozwijane z powrotem
        na probki), wiec dlugosc wyniku jest rowna len(prepared).
        """
        features = prepared[self.DATA_COLUMNS]

        if len(features) < self.windowing.window_size:
            raise ValueError(
                f'Dane maja {len(features)} probek, a okno wymaga co najmniej '
                f'{self.windowing.window_size}'
            )

        windowed = self.windowing.window(features.to_numpy())
        featured = extractWindowFeatures(windows=windowed)
        y_pred = self.model.predict(featured)
        smoothed = self.smooth(y_pred=y_pred, min_group_size=min_group_size)

        return self.windowing.reverse_window_labels(smoothed, len(features))

    @staticmethod
    def calc_class_weights(y_train_windowed):
        counts = pd.Series(y_train_windowed).value_counts()
        row_count = y_train_windowed.shape[0]
        weights = {c: row_count / counts[c] * len(counts) for c in counts}
        sample_weights = [weights[label] for label in y_train_windowed]
        return sample_weights

    @staticmethod
    def sqrt_balanced_weights(y):
        counts = pd.Series(y).value_counts()

        n_samples = len(y)
        n_classes = len(counts)

        class_weights = {
            cls: np.sqrt(n_samples / (n_classes * count))
            for cls, count in counts.items()
        }

        sample_weights = np.array(
            [class_weights[label] for label in y]
        )

        return sample_weights

    @staticmethod
    def smooth(y_pred, min_group_size):
        y_pred = np.asarray(y_pred).copy()

        diff = np.diff(y_pred)
        switch_idx = np.where(diff != 0)[0] + 1
        switch_idx = np.r_[0, switch_idx, len(y_pred)]

        for start, end in zip(switch_idx[:-1], switch_idx[1:]):
            if start == 0:
                continue

            if end - start < min_group_size:
                y_pred[start:end] = y_pred[start - 1]

        return y_pred

    @staticmethod
    def cal_class_weights_sklearn(y_train):
        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        return weights

    def _make_fit_summary(self, path: str | Path = "pipeline_config.json") -> dict:
        summary = {
            'model_params': self.model.get_params(),
            'window_params': self.windowing.get_params(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, default=str)

        return summary

    def save_model(self, path: str | Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)


@dataclasses.dataclass
class PipelineInit:
    model_params: dict
    window_size: int = 20
    window_step: int = 2
    model_n_estimators: int = 2000
    model_early_stopping_rounds: int = 20
    eval_metric: str = 'mlogloss'
    objective: str = "multi:softprob"
    num_class: int = 5


# {'max_depth': 5, 'subsample': 0.8, 'learning_rate': 0.2}
def make_pipeline(init: PipelineInit):
    window = DataWindowing(window_size=init.window_size, window_step=init.window_step,
                           window_labeling_strategy=CenterLabelingStrategy())
    params = dict(init.model_params)
    model = XGBClassifier(
        n_estimators=init.model_n_estimators,
        early_stopping_rounds=init.model_early_stopping_rounds,
        eval_metric=init.eval_metric,
        objective=init.objective,
        num_class=init.num_class,
        **params,
    )
    pipeline = XGBPipeline(windowing=window, model=model)
    return pipeline

# if __name__ == '__main__':
#     from accur8pool.files_utils.filesManager import FilesManager
#     from accur8pool.files_utils.dataFrameLoader import PandasDataFrameLoader
#
#
#     def smooth(y_pred, min_group_size):
#         y_pred = np.asarray(y_pred).copy()
#
#         diff = np.diff(y_pred)
#         switch_idx = np.where(diff != 0)[0] + 1
#         switch_idx = np.r_[0, switch_idx, len(y_pred)]
#
#         for start, end in zip(switch_idx[:-1], switch_idx[1:]):
#             if start == 0:
#                 continue
#
#             if end - start < min_group_size:
#                 y_pred[start:end] = y_pred[start - 1]
#
#         return y_pred
#
#
#     files = FilesManager(data_paths=r"C:\Users\apietka\PycharmProjects\accur8pool\data\raw_data",
#                          labels_paths=r"C:\Users\apietka\PycharmProjects\accur8pool\data\raw_data")
#
#     data_labels_paths = files.get_data_label_paths()
#     dataFrameLoader = PandasDataFrameLoader()
#     data_df_label_df = dataFrameLoader.load_data_labels_dfs(data_labels_paths)
#     dataFrameLoader.
