from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from const import *
from utils import (
    _normalize,
    _smooth,
    calc_pitch,
    calc_roll,
    calc_magnitude,
    lowpass_filter,
)


class DataFrameTransformer:
    def __init__(self, data: DataFrame, copy: bool = True):
        self.data = data.copy() if copy else data

    def drop_first_row(self) -> "DataFrameTransformer":
        self.data = self.data.iloc[1:].reset_index(drop=True)
        return self

    def dt_ms_to_sec(self, dt_col: str = DT) -> "DataFrameTransformer":
        self.data[dt_col] = self.data[dt_col] / 1000.0
        return self

    def add_time(self, dt_col: str = DT, time_col: str = TIME) -> "DataFrameTransformer":
        self.data[time_col] = self.data[dt_col].cumsum()
        return self

    def normalize(self, columns: Sequence[str]) -> "DataFrameTransformer":
        for col in columns:
            self.data[col] = _normalize(self.data[col])
        return self

    def smooth(self, columns: Sequence[str], window: int) -> "DataFrameTransformer":
        for col in columns:
            self.data[col] = _smooth(self.data[col], window=window)
        return self

    def lowpass(self, columns: Sequence[str], cutoff: float) -> "DataFrameTransformer":
        for col in columns:
            self.data[col] = lowpass_filter(self.data[col], cutoff=cutoff)
        return self

    def add_magnitude(
            self,
            source_cols: Sequence[str],
            new_col: str,
    ) -> "DataFrameTransformer":
        if len(source_cols) != 3:
            raise ValueError("source_cols musi mieć dokładnie 3 kolumny: x, y, z")

        self.data[new_col] = calc_magnitude(
            self.data[source_cols[0]],
            self.data[source_cols[1]],
            self.data[source_cols[2]],
        )
        return self

    def add_jerk(
            self,
            source_cols: Sequence[str],
            time_col: str = TIME,
            prefix: str = JERK,
    ) -> "DataFrameTransformer":
        if time_col not in self.data.columns:
            self.add_time()

        acc = self.data[list(source_cols)].to_numpy()
        time_axis = self.data[time_col].to_numpy()

        jerk = np.gradient(acc, time_axis, axis=0)

        for i, col in enumerate(source_cols):
            suffix = col.split("_")[-1]
            self.data[f"{prefix}_{suffix}"] = jerk[:, i]

        self.data[f"{prefix}_MAG"] = np.linalg.norm(jerk, axis=1)
        return self

    def add_roll(self, alpha: float = 0.98) -> "DataFrameTransformer":
        self.data[ROLL] = calc_roll(
            acc_y=self.data[ACC_Y].tolist(),
            acc_z=self.data[ACC_Z].tolist(),
            gyr_x=self.data[GYR_X].tolist(),
            dt=self.data[DT].tolist(),
            alpha=alpha,
        )
        return self

    def add_pitch(self, alpha: float = 0.98) -> "DataFrameTransformer":
        self.data[PITCH] = calc_pitch(
            acc_x_list=self.data[ACC_X].tolist(),
            acc_y_list=self.data[ACC_Y].tolist(),
            acc_z_list=self.data[ACC_Z].tolist(),
            gyr_y_list=self.data[GYR_Y].tolist(),
            dt_list=self.data[DT].tolist(),
            alpha=alpha,
        )
        return self

    def drop_columns_ending_with(self, suffix: str) -> "DataFrameTransformer":
        self.data = self.data.loc[:, ~self.data.columns.str.endswith(suffix)]
        return self

    def result(self) -> DataFrame:
        return self.data


def transform_df(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)

    return (
        DataFrameTransformer(df)
        .drop_first_row()
        .dt_ms_to_sec()
        .lowpass([ACC_X, ACC_Y, ACC_Z], cutoff=10)
        .add_magnitude([ACC_X, ACC_Y, ACC_Z], ACC_MAGNITUDE)
        .add_magnitude([GYR_X, GYR_Y, GYR_Z], GYR_MAGNITUDE)
        .add_time()
        .add_jerk([ACC_X, ACC_Y, ACC_Z])
        .add_roll()
        .add_pitch()
        .normalize([ACC_X, ACC_Y, ACC_Z, ACC_MAGNITUDE, GYR_MAGNITUDE])
        .smooth(
            [ACC_X, ACC_Y, ACC_Z, ACC_MAGNITUDE, GYR_MAGNITUDE, f"{JERK}_MAG", ROLL, PITCH],
            window=5,
        )
        .result()
    )
