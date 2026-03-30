import itertools
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, butter
from const import *


def calc_pitch(acc_x_list, acc_y_list, acc_z_list, gyr_y_list, dt_list, alpha):
    pitch = []
    iterations = min(len(acc_x_list), len(acc_y_list), len(acc_z_list), len(gyr_y_list))
    pitch_gyr = 0
    for index in range(iterations):
        acc_x = acc_x_list[index]
        acc_y = acc_y_list[index]
        acc_z = acc_z_list[index]
        gyr_y = gyr_y_list[index]
        dt = dt_list[index]
        pitch_acc = np.arctan2(-acc_x, np.sqrt(acc_y ** 2 + acc_z ** 2))
        pitch_gyr = pitch_gyr + gyr_y * dt
        pitch_gyr = alpha * pitch_gyr + (1 - alpha) * pitch_acc
        pitch.append(pitch_gyr)

    return pitch


def calc_roll(acc_y_list, acc_z_list, gyr_x_list, deltatime_list, alpha):
    roll_gyr = 0
    roll = []
    iterations = min(len(acc_y_list), len(gyr_x_list), len(acc_z_list))
    for index in range(iterations):
        acc_z = acc_z_list[index]
        acc_y = acc_y_list[index]
        gyr_x = gyr_x_list[index]
        dt = deltatime_list[index]
        roll_acc = np.arctan2(acc_y, acc_z)
        roll_gyr = roll_gyr + gyr_x * dt
        roll_gyr = alpha * roll_gyr + (1 - alpha) * roll_acc
        roll.append(roll_gyr)
    return roll


def _normalize(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if np.isclose(min_val, max_val):
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def _smooth(signal: np.ndarray | pd.Series,window=5) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 3:
        return arr
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def calc_jerk(acc_x_list, acc_y_list, acc_z_list, dt_list):
    magnitude = calc_magnitude(acc_x_list, acc_y_list, acc_z_list)
    magnitude = np.array(magnitude)
    dt_array = np.array(dt_list)

    # Obliczamy różnice między kolejnymi próbkami
    jerk = np.zeros_like(magnitude)
    # (mag[i] - mag[i-1]) / dt[i]
    jerk[1:] = (magnitude[1:] - magnitude[:-1]) / dt_array[1:]
    return jerk.tolist()


def calc_magnitude(acc_x_list, acc_y_list, acc_z_list):
    ax = np.array(acc_x_list)
    ay = np.array(acc_y_list)
    az = np.array(acc_z_list)
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2).tolist()


def get_df_from_csv(path):
    df = pd.read_csv(path)
    return df


def lowpass_filter(data, cutoff=8, fs=100):
    b, a = butter(2, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, data, axis=0)


if __name__ == '__main__':
    import plotly.express as px

    df = get_df_from_csv(r'C:\Users\apietka\PycharmProjects\accur8pool\data\wolneuderzeniewbutle.csv')
    mag = calc_magnitude(df[ACC_X], df[ACC_Y], df[ACC_Z])
    fil_mag = lowpass_filter(mag, cutoff=20)
    df_plot = pd.DataFrame({'fil_mag': fil_mag, 'mag': mag})
    px.line(df_plot, y=['fil_mag', 'mag'], labels={'value': 'Magnitude', 'index': 'Sample'}).show()
