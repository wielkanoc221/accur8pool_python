import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from const import *
from utils import *


def get_df_from_csv(path):
    df = pd.read_csv(path)
    return df


def add_jerk(df: DataFrame, dt):
    try:
        df[JERK] = calc_jerk(df[LIN_ACC_X], df[LIN_ACC_Y], df[LIN_ACC_Z], dt)
    except KeyError as ke:
        print(f"First calculete: {ke}")
    return df


def add_pitch(df: DataFrame):
    try:
        df[PITCH] = calc_pitch(acc_x_list=df[ACC_X], acc_y_list=df[ACC_Y], acc_z_list=df[ACC_Z], gyr_y_list=df[GYR_Y],
                               alpha=0.98, dt=0.01)

    except KeyError as ke:
        print(f"First calculete: {ke}")
    return df


def add_roll(df: DataFrame):
    df[ROLL] = calc_roll(acc_y_list=df[ACC_Y], acc_z_list=df[ACC_Z], gyr_x_list=df[GYR_X], dt=0.01, alpha=0.98)
    return df


def add_linear_acceleration(df: DataFrame):
    try:
        lin_acc_x, lin_acc_y, lin_acc_z = transform_to_linear_acceleration(pitch=df[PITCH], roll=df[ROLL],
                                                                           acc_x=df[ACC_X],
                                                                           acc_y=df[ACC_Y], acc_z=df[ACC_Z])
        df[LIN_ACC_X] = lin_acc_x
        df[LIN_ACC_Y] = lin_acc_y
        df[LIN_ACC_Z] = lin_acc_z
    except KeyError as ke:
        print(f"First calculete: {ke}")
    return df


def add_time_row(df: DataFrame, dt):
    row_counts = df.shape[0]
    df[TIME] = np.arange(0, row_counts * dt, dt)
    return df


def add_magnitude(df: DataFrame):
    df[MAGNITUDE] = calc_magnitude(df[LIN_ACC_X], df[LIN_ACC_Y], df[LIN_ACC_Z])
    return df


def plot_magnitude(df: DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # linia 1 – JERK (duża skala)
    fig.add_trace(go.Scatter(x=df[TIME], y=df[JERK], name="jerk"), secondary_y=False)

    # linia 2 – ROLL (mała skala)
    fig.add_trace(go.Scatter(x=df[TIME], y=df[ROLL], name="roll"), secondary_y=True)
    fig.show()


def plot_fft(df):
    N = df.shape[0]
    for axis in ["accx", "accy", "accz"]:
        signal = df[axis].values
        fft_values = np.abs(np.fft.fft(signal)[:100 // 2])
        fft_freq = np.fft.fftfreq(N, d=0.01)[:100 // 2]
        plt.plot(fft_freq, fft_values, label=axis)

    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.show()


def plot_3d_motion(df: DataFrame):
    vx = vy = vz = 0
    vel_x = []
    vel_y = []
    vel_z = []

    for ax, ay, az, dt in zip(df[LIN_ACC_X], df[LIN_ACC_Y], df[LIN_ACC_Z], [0.01] * df.shape[0]):
        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        # opcjonalnie: ZUPT, jeżeli mag < threshold
        # if np.sqrt(ax**2 + ay**2 + az**2) < 0.3:
        #     vx = vy = vz = 0

        vel_x.append(vx)
        vel_y.append(vy)
        vel_z.append(vz)
    df["vel_x"] = vel_x
    df["vel_y"] = vel_y
    df["vel_z"] = vel_z

    posx = posy = posz = 0
    pos_x = []
    pos_y = []
    pos_z = []

    for vx, vy, vz, dt in zip(df["vel_x"], df["vel_y"], df["vel_z"], [0.01] * df.shape[0]):
        posx += vx * dt
        posy += vy * dt
        posz += vz * dt

        pos_x.append(posx)
        pos_y.append(posy)
        pos_z.append(posz)

    df["x"] = pos_x
    df["y"] = pos_y
    df["z"] = pos_z

    fig = px.scatter_3d(
        df,
        x=df['x'],
        y=df['y'],
        z=df['z'],
        color=TIME,

    )
    fig.show()


if __name__ == '__main__':
    paths = {'jarozbicie': r'C:\Users\apietka\PycharmProjects\accur8pool\data\andrzej rozbicie.csv',
             'arek rozbicie': r'C:\Users\apietka\PycharmProjects\accur8pool\data\arek rozbicie.csv',
             'andrzejwbicie': r'C:\Users\apietka\PycharmProjects\accur8pool\data\andrzejwbiciedosycmocne.csv'
             }
    df = get_df_from_csv(r'C:\Users\apietka\PycharmProjects\accur8pool\data\rozbicieandrzej2.csv')

    add_roll(df)
    add_pitch(df)
    add_linear_acceleration(df)
    add_magnitude(df)
    add_jerk(df, 0.01)
    add_time_row(df, 0.01)

    window = np.hanning(len(df[MAGNITUDE]))
    acc_windowed = df[MAGNITUDE] * window

    values = zip(df[MAGNITUDE], range(len(df[MAGNITUDE])))
    print(list(values))
    plt.plot(df[JERK])
    plt.show()