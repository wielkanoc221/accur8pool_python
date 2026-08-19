import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from accur8pool.data_processing.dataWindowing import DataWindowing, extractWindowFeatures, CenterLabelingStrategy
from sklearn.utils.class_weight import compute_sample_weight
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

    def load_model(self):
        pass

    def fit(self, X_data, y_data: pd.Series):
        X_data = transform_raw_df(X_data)[self.DATA_COLUMNS]
        y_data = y_data.fillna(0)
        stop_idx_train = int(len(X_data) * 0.8)

        X_train, y_train = X_data[:stop_idx_train], y_data[:stop_idx_train]
        X_eval, y_eval = X_data[stop_idx_train:], y_data[stop_idx_train:]

        X_train_windowed, y_train_windowed = self.windowing.window_with_extractions(X_train.to_numpy(),
                                                                                    y_train.to_numpy())
        X_eval_windowed, y_eval_windowed = self.windowing.window_with_extractions(X_eval.to_numpy(), y_eval.to_numpy())

        weights = self.sqrt_balanced_weights(y_train_windowed)

        self.model.fit(X_train_windowed, y_train_windowed,
                       eval_set=[(X_eval_windowed, y_eval_windowed)], verbose=2, sample_weight=weights)

        self._make_fit_summary()

    def predict(self, X_data: pd.DataFrame):
        prepared = transform_raw_df(X_data)[self.DATA_COLUMNS]
        windowed = self.windowing.window(prepared.to_numpy())
        featured = extractWindowFeatures(windows=windowed)
        y_pred = self.model.predict(featured)
        smoothed = self.smooth(y_pred=y_pred, min_group_size=6)
        reversed_ = self.windowing.reverse_window_labels(smoothed, len(prepared))
        return reversed_

    @staticmethod
    def calc_class_weights(y_train_windowed):
        counts = pd.Series(y_train_windowed).value_counts()
        row_count = y_train_windowed.shape[0]
        weights = {c: row_count / counts[c] * len(counts) for c in counts.keys()}
        sample_weights = [weights[l] for l in y_train_windowed]
        return sample_weights

    @staticmethod
    def calc_weights(y):
        weights = compute_sample_weight(y)
        return weights

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

    def _make_fit_summary(self):
        model_params = self.model.get_params()
        window_params = self.windowing.get_params()

        summary = {'model_params': model_params, 'window_params': window_params}
        with open("pipeline_config.json", "w+", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)


def make_pipeline():
    window = DataWindowing(window_size=20, window_step=2, window_labeling_strategy=CenterLabelingStrategy())
    params = {'max_depth': 5, 'subsample': 0.8, 'learning_rate': 0.2}
    model = XGBClassifier(
        n_estimators=2000,
        early_stopping_rounds=20,
        eval_metric='mlogloss',
        objective="multi:softprob",
        num_class=5,
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