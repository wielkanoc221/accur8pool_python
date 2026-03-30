from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from const import *
from dataframe_transformations import DataFrameTransformations
from utils import _smooth


def load_and_prepare(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    transformed = DataFrameTransformations(df)
    transformed.dt2sec().add_pitch().add_roll().add_jerk([LIN_ACC_X,LIN_ACC_Y,LIN_ACC_Z]).add_time_row(
    )

    df = transformed.data
    df["gyro_magnitude"] = np.sqrt(df[GYR_X] ** 2 + df[GYR_Y] ** 2 + df[GYR_Z] ** 2)
    df["lin_acc_magnitude"] = np.sqrt(df[LIN_ACC_X] ** 2 + df[LIN_ACC_Y] ** 2 + df[LIN_ACC_Z] ** 2)
    df["movement_score"] = (

            + 0.6 * _smooth(df["gyro_magnitude"])
            + 0.4 * _smooth(df["lin_acc_magnitude"])
    )
    return df


if __name__ == '__main__':
    df = load_and_prepare(r'C:\Users\apietka\PycharmProjects\accur8pool\data\data20260329_220029.csv')
    fig = px.line(df, x=df[TIME], y=['gyro_magnitude','movement_score'])
    fig.show()
