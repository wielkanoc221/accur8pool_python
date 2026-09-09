from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from .const import (
    ACC_X,
    ACC_Y,
    ACC_Z,
    GYR_X,
    GYR_Y,
    JERK,
    PITCH,
    ROLL,
    TIME,
    TIMESTAMP,
)
from .utils import (
    _normalize,
    calc_magnitude,
    calc_pitch,
    calc_roll,
    lowpass_filter,
)


class DataFrameTransformerBase:
    def __init__(self, data: DataFrame, copy: bool = True):
        self.data = data.copy() if copy else data
        self.new_columns = []

    def downsample(self) -> 'DataFrameTransformerBase':

        float_columns = self.data.select_dtypes(include=['float64']).columns
        if len(float_columns) > 0:
            self.data[float_columns] = self.data[float_columns].astype(np.float32)

        int_columns = self.data.select_dtypes(include=['int64']).columns
        if len(int_columns) > 0:
            self.data[int_columns] = self.data[int_columns].astype(np.int32)

        return self

    def drop_first_row(self) -> "DataFrameTransformerBase":
        self.data = self.data.iloc[1:].reset_index(drop=True)
        return self

    def dt_ms_to_sec(self, dt_col: str = TIMESTAMP) -> "DataFrameTransformerBase":
        self.data[dt_col] = self.data[dt_col] / 1000.0
        return self

    def add_time(self, dt_col: str = TIMESTAMP, time_col: str = TIME) -> "DataFrameTransformerBase":
        self.data[time_col] = self.data[dt_col].cumsum()
        self.new_columns.append(time_col)
        return self

    def normalize(self, columns: Sequence[str] | None = None) -> "DataFrameTransformerBase":
        """
        Jezeli columns is None to bierze wszystkie kolumny
        z datafram z wykluczeniem TIME i DT
        """
        if columns is None:
            columns = [
                column for column in self.data.select_dtypes(include="number").columns
                if column not in (TIME, TIMESTAMP)
            ]

        columns = list(columns)
        missing = [column for column in columns if column not in self.data.columns]
        if missing:
            raise KeyError(f"Brakujace kolumny do normalizacji: {missing}")

        self.data[columns] = self.data[columns].apply(_normalize)
        return self

    def smooth(self, columns: Sequence[str], window: int = 5) -> "DataFrameTransformerBase":
        self.data[columns] = (
            self.data[columns]
            .rolling(window=window, center=True)
            .mean()
        )
        return self

    def lowpass(self, columns: Sequence[str], cutoff: float) -> "DataFrameTransformerBase":
        for col in columns:
            if self.data[col].isna().any():
                raise ValueError(f"Kolumna {col} zawiera NaN - filtfilt zwrocilby same NaN-y")

            self.data[col] = lowpass_filter(self.data[col], cutoff=cutoff)
        return self

    def add_magnitude(
            self,
            source_cols: Sequence[str],
            new_col: str,
    ) -> "DataFrameTransformerBase":
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
    ) -> "DataFrameTransformerBase":
        if time_col not in self.data.columns:
            self.add_time()

        acc = self.data[list(source_cols)].to_numpy()
        time_axis = self.data[time_col].to_numpy()

        jerk = np.gradient(acc, time_axis, axis=0)

        for i, col in enumerate(source_cols):
            suffix = col.split("_")[-1]
            self.data[f"jerk_{suffix}"] = jerk[:, i]

        self.data[f"{prefix}_magnitude_jerk"] = np.linalg.norm(jerk, axis=1)
        return self

    def add_roll(self, alpha: float = 0.98) -> "DataFrameTransformerBase":
        self.data[ROLL] = calc_roll(
            acc_y=self.data[ACC_Y].tolist(),
            acc_z=self.data[ACC_Z].tolist(),
            gyr_x=self.data[GYR_X].tolist(),
            dt=self.data[TIMESTAMP].tolist(),
            alpha=alpha,
        )
        return self

    def add_pitch(self, alpha: float = 0.98) -> "DataFrameTransformerBase":
        self.data[PITCH] = calc_pitch(
            acc_x_list=self.data[ACC_X].tolist(),
            acc_y_list=self.data[ACC_Y].tolist(),
            acc_z_list=self.data[ACC_Z].tolist(),
            gyr_y_list=self.data[GYR_Y].tolist(),
            dt_list=self.data[TIMESTAMP].tolist(),
            alpha=alpha,
        )
        return self

    def drop_columns_ending_with(self, suffix: str) -> "DataFrameTransformerBase":
        self.data = self.data.loc[:, ~self.data.columns.str.endswith(suffix)]
        return self

    def result(self) -> DataFrame:
        return self.data


class DataFrameTransformerV2(DataFrameTransformerBase):
    def add_time(self, dt_col: str = TIMESTAMP, time_col: str = TIME) -> "DataFrameTransformerV2":
        deltas = self.data[dt_col].diff().fillna(0.0)
        self.data[dt_col] = deltas / 1_000_000
        self.data[time_col] = self.data[dt_col].cumsum()
        if time_col not in self.new_columns:
            self.new_columns.append(time_col)

        return self

    def add_roll(self, alpha: float = 0.98) -> "DataFrameTransformerV2":
        self.data[ROLL + '_calculated'] = calc_roll(
            acc_y=self.data[ACC_Y].tolist(),
            acc_z=self.data[ACC_Z].tolist(),
            gyr_x=self.data[GYR_X].tolist(),
            dt=self.data[TIMESTAMP].tolist(),
            alpha=alpha,
        )
        return self

    def add_pitch(self, alpha: float = 0.98) -> "DataFrameTransformerV2":
        self.data[PITCH + "_calculated"] = calc_pitch(
            acc_x_list=self.data[ACC_X].tolist(),
            acc_y_list=self.data[ACC_Y].tolist(),
            acc_z_list=self.data[ACC_Z].tolist(),
            gyr_y_list=self.data[GYR_Y].tolist(),
            dt_list=self.data[TIMESTAMP].tolist(),
            alpha=alpha,
        )
        return self


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("uzycie: python -m accur8pool.data_processing.data_transformations <plik.csv>")

    df = pd.read_csv(sys.argv[1])
    df = DataFrameTransformerV2(df).add_time().result()
    print(df[TIME])
