from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from const import (
    DT,
    GYR_X,
    GYR_Y,
    GYR_Z,
    JERK,
    LIN_ACC_X,
    LIN_ACC_Y,
    LIN_ACC_Z,
    TIME,
)
from dataframe_transformations import DataFrameTransformations


@dataclass
class ShotWindow:
    phase: str
    start_idx: int
    end_idx: int
    start_time_s: float
    end_time_s: float
    confidence: float


class ShotPipeline:
    """
    Heuristic pipeline for detecting billiards shot phases in IMU CSV files.

    Phases:
    - prepare: transition from stillness to active movement
    - aim: low dynamics just before strike
    - strike: high-dynamics event around impact
    """

    def __init__(
            self,
            dt: float = 0.01,
            smooth_window: int = 9,
            min_aim_duration_s: float = 0.2,
            max_aim_duration_s: float = 1.2,
            prepare_duration_s: float = 0.7,
            strike_half_window_s: float = 0.12,
    ):
        self.dt = dt
        self.smooth_window = smooth_window
        self.min_aim_duration_s = min_aim_duration_s
        self.max_aim_duration_s = max_aim_duration_s
        self.prepare_duration_s = prepare_duration_s
        self.strike_half_window_s = strike_half_window_s

    def load_and_prepare(self, csv_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        if DT not in df.columns:
            df[DT] = self.dt

        transformed = DataFrameTransformations(df)
        transformed.add_pitch().add_roll().add_linear_acceleration().add_magnitude().add_jerk().add_time_row(
            dt=self.dt
        )

        df = transformed.data
        df["gyro_magnitude"] = np.sqrt(df[GYR_X] ** 2 + df[GYR_Y] ** 2 + df[GYR_Z] ** 2)
        df["lin_acc_magnitude"] = np.sqrt(df[LIN_ACC_X] ** 2 + df[LIN_ACC_Y] ** 2 + df[LIN_ACC_Z] ** 2)
        df["movement_score"] = (
                self._smooth(np.abs(df[JERK]))
                + 0.6 * self._smooth(df["gyro_magnitude"])
                + 0.4 * self._smooth(df["lin_acc_magnitude"])
        )
        return df

    def detect_windows(self, df: pd.DataFrame) -> List[ShotWindow]:
        strike_idx = self._detect_strike_index(df)
        strike = self._build_strike_window(df, strike_idx)
        aim = self._detect_aim_window(df, strike.start_idx)
        prepare = self._detect_prepare_window(df, aim.start_idx)

        return [prepare, aim, strike]

    def export_windows(
            self,
            df: pd.DataFrame,
            windows: List[ShotWindow],
            output_dir: str | Path,
            file_prefix: str,
    ) -> pd.DataFrame:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for window in windows:
            chunk = df.iloc[window.start_idx: window.end_idx + 1].copy()
            chunk.to_csv(output_path / f"{file_prefix}_{window.phase}.csv", index=False)

        summary = pd.DataFrame([asdict(w) for w in windows])
        summary.to_csv(output_path / f"{file_prefix}_windows_summary.csv", index=False)
        return summary

    def run(
            self,
            csv_path: str | Path,
            output_dir: Optional[str | Path] = None,
    ) -> Dict[str, object]:
        csv_path = Path(csv_path)
        df = self.load_and_prepare(csv_path)
        windows = self.detect_windows(df)

        summary = None
        if output_dir is not None:
            summary = self.export_windows(df, windows, output_dir, csv_path.stem)

        return {
            "data": df,
            "windows": windows,
            "summary": summary,
        }

    def _detect_strike_index(self, df: pd.DataFrame) -> int:
        score = (
                0.7 * self._normalize(np.abs(df[JERK]).values)
                + 0.2 * self._normalize(df["lin_acc_magnitude"].values)
                + 0.1 * self._normalize(df["gyro_magnitude"].values)
        )
        return int(np.argmax(score))

    def _build_strike_window(self, df: pd.DataFrame, strike_idx: int) -> ShotWindow:
        half_window = max(1, int(self.strike_half_window_s / self.dt))
        start_idx = max(0, strike_idx - half_window)
        end_idx = min(len(df) - 1, strike_idx + half_window)

        local_jerk = np.abs(df[JERK].iloc[start_idx: end_idx + 1].values)
        confidence = float(np.clip(np.percentile(self._normalize(local_jerk), 90), 0.0, 1.0))

        return ShotWindow(
            phase="strike",
            start_idx=start_idx,
            end_idx=end_idx,
            start_time_s=float(df[TIME].iloc[start_idx]),
            end_time_s=float(df[TIME].iloc[end_idx]),
            confidence=confidence,
        )

    def _detect_aim_window(self, df: pd.DataFrame, strike_start_idx: int) -> ShotWindow:
        max_len = max(1, int(self.max_aim_duration_s / self.dt))
        min_len = max(1, int(self.min_aim_duration_s / self.dt))

        search_start = max(0, strike_start_idx - max_len)
        search_end = max(search_start + min_len, strike_start_idx - 1)

        move = self._smooth(df["movement_score"].values)
        threshold = np.percentile(move[search_start: search_end + 1], 35)

        best_start = max(search_start, strike_start_idx - min_len)
        for idx in range(strike_start_idx - 1, search_start - 1, -1):
            if move[idx] > threshold:
                best_start = idx + 1
                break
            best_start = idx

        if (strike_start_idx - best_start) < min_len:
            best_start = max(search_start, strike_start_idx - min_len)

        score_region = move[best_start:strike_start_idx]
        confidence = float(np.clip(1.0 - np.mean(self._normalize(score_region)), 0.0, 1.0))

        end_idx = strike_start_idx - 1
        return ShotWindow(
            phase="aim",
            start_idx=best_start,
            end_idx=end_idx,
            start_time_s=float(df[TIME].iloc[best_start]),
            end_time_s=float(df[TIME].iloc[end_idx]),
            confidence=confidence,
        )

    def _detect_prepare_window(self, df: pd.DataFrame, aim_start_idx: int) -> ShotWindow:
        duration = max(1, int(self.prepare_duration_s / self.dt))
        end_idx = max(0, aim_start_idx - 1)
        start_idx = max(0, end_idx - duration)

        move = self._smooth(df["movement_score"].values)
        region = move[start_idx: end_idx + 1]
        trend = np.polyfit(np.arange(len(region)), region, 1)[0] if len(region) > 1 else 0.0
        confidence = float(np.clip(0.5 + np.sign(trend) * 0.25 + np.std(self._normalize(region)) * 0.25, 0.0, 1.0))

        return ShotWindow(
            phase="prepare",
            start_idx=start_idx,
            end_idx=end_idx,
            start_time_s=float(df[TIME].iloc[start_idx]),
            end_time_s=float(df[TIME].iloc[end_idx]),
            confidence=confidence,
        )

    def _smooth(self, signal: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(signal, dtype=float)
        if len(arr) < 3:
            return arr
        window = max(3, self.smooth_window)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(arr, kernel, mode="same")

    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=float)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if np.isclose(min_val, max_val):
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)


if __name__ == "__main__":
    pipeline = ShotPipeline(dt=0.01)
    result = pipeline.run("data/rozbicieandrzej2.csv", output_dir="output_windows")
    print(pd.DataFrame([asdict(w) for w in result["windows"]]))
