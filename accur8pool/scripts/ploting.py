from __future__ import annotations
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
from pandas import DataFrame
from accur8pool.data_processing.data_transformations import DataFrameTransformerBase
import pandas as pd


def add_labels_to_plot(
        fig: Figure,
        data_df: DataFrame,
        labels_df: DataFrame,
        x_axis: str = "time",
) -> Figure:
    required_label_cols = ["label", "label_id", "start_idx", "end_idx"]

    missing = [col for col in required_label_cols if col not in labels_df.columns]
    if missing:
        raise ValueError(f"labels_df missing columns: {missing}")

    if x_axis not in data_df.columns:
        raise ValueError(f"data_df missing x_axis column: {x_axis}")

    data_df = data_df.reset_index(drop=True)
    labels_df = labels_df.reset_index(drop=True)

    segments = (
        labels_df
        .groupby("label_id", as_index=False)
        .agg(
            label=("label", "first"),
            start_idx=("start_idx", "min"),
            end_idx=("end_idx", "max"),
        )
    )

    for _, row in segments.iterrows():
        label = row["label"]

        # opcjonalnie: pomiń brak labela / label 0
        if pd.isna(label) or label == 0:
            continue

        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])

        if start_idx < 0 or end_idx >= len(data_df):
            raise ValueError(
                f"Label index out of range: start_idx={start_idx}, "
                f"end_idx={end_idx}, data_len={len(data_df)}"
            )

        x0 = data_df.loc[start_idx, x_axis]
        x1 = data_df.loc[end_idx, x_axis]

        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="green",
            opacity=0.2,
            line_width=1,
            annotation_text=str(label),
            annotation_position="top left",
        )

    return fig


def to_segments(labels):
    labels = np.asarray(labels)

    change = np.where(labels != np.roll(labels, 1))[0]
    change = np.r_[0, change, len(labels)]

    segments = []
    for i in range(len(change) - 1):
        start = change[i]
        end = change[i + 1]
        label = labels[start]
        segments.append((start, end, label))

    return segments


def plot_prepared_data(prepared_data_frame: DataFrame, labels_data_frame: DataFrame = None, normalize_data=True,
                       columns=None):
    """
    Funkcja robi wykres w plotly ktory rysuje wykres z przygotowanych danych,
    dodatkowo jako opcja jest podanie odpowiadajacej danym labeli aby nalozylo sie na wyrkes
    :param prepared_data_frame:
    :param labels_data_frame:
    :param normalize_data:
    :return:
    """
    colors = {
        0: "rgba(0,0,0,1)",
        1: "rgba(102, 255, 51,1)",
        2: "rgba(0, 153, 255,1)",
        3: "rgba(255, 0, 0,1)",
        4: "rgba(255, 255, 153,1)",
    }

    data_columns = columns or ['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz', 'magx', 'magy', 'magz',
                               'linaccx', 'linaccy', 'linaccz', 'rotx', 'roty', 'rotz', 'acc_magnitude',
                               'gyr_magnitude',
                               'jerk_accx',
                               'jerk_accy', 'jerk_accz', 'acc_magnitude_jerk', 'jerk_gyrx',
                               'jerk_gyry', 'jerk_gyrz', 'gyr_magnitude_jerk', 'roll', 'pitch']

    if not all(col in prepared_data_frame.columns for col in data_columns):
        raise ValueError('Nieprawidlowe wartosci kolumn')

    if normalize_data:
        prepared_data_frame = DataFrameTransformerBase(prepared_data_frame).normalize(
            columns=list(data_columns)).result()

    fig = px.line(prepared_data_frame, y=data_columns)
    labels = labels_data_frame['label']
    if labels_data_frame is not None:
        segments = to_segments(labels)
        for index, (start, end, label) in enumerate(segments):
            print(index, '/', len(segments))
            fig.add_vline(
                x=start,
                line_color=colors[label],
                line_width=5
            )
    fig.show()

#
# if __name__ == '__main__':
#     d1 = pd.read_csv(r'C:\Users\apietka\PycharmProjects\accur8pool\data\prepared\krzysiek\data20260704_182159.csv')
#     l1 = pd.read_csv(r'C:\Users\apietka\PycharmProjects\accur8pool\data\labeled_v2\data20260704_182159.csv')
#     plot_prepared_data(d1, l1)
