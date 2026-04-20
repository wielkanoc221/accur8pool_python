import csv
import dataclasses
import time
from typing import List

import numpy as np
from pandas import DataFrame

from dataframe_transformations import get_df_from_csv, DataFrameTransformations


@dataclasses.dataclass
class TimeLabel:
    start: float
    stop: float


def get_index_by_time(df: DataFrame, time: float):
    index = (df['time'] - time).abs().idxmin()
    return index


def make_shot_label(df: DataFrame, time_labels: TimeLabel | List[TimeLabel]):
    if not isinstance(time_labels, List):
        time_labels = [time_labels]
    print(df.shape)
    labels = np.zeros(df.shape[0])
    for time_label in time_labels:
        start_index = get_index_by_time(df, time_label.start)
        stop_index = get_index_by_time(df, time_label.stop)
        labels[start_index:stop_index + 1] = 1
        print(labels)
    return labels

def make_label_csv(filename, labels_list):
    with open(filename + '.csv', 'w+') as f:
        f.write('shot'+'\n')
        for label in labels_list:
            f.write(str(label) + ',' + '\n')


if __name__ == '__main__':
    path = r'data/danejaroty.csv'

    df = get_df_from_csv(path)
    data = DataFrameTransformations(df)
    (data.drop_first_row()
     .dt2sec()
     .add_time_row()
     )
    tl = [TimeLabel(1, 2), TimeLabel(5, 7)]
    t1= time.time()
    lista = make_shot_label(data.data, tl)
    make_label_csv('test', lista)
    print(time.time()-t1)