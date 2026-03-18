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


class DataFrameTransformations:
    def __init__(self, data: DataFrame):
        self.data = data

    def add_jerk(self):
        try:
            self.data[JERK] = calc_jerk(
                self.data[LIN_ACC_X],
                self.data[LIN_ACC_Y],
                self.data[LIN_ACC_Z],
                self.data[DT]
            )
        except KeyError as ke:
            print(f"First calculate: {ke}")
        return self

    def add_pitch(self, dt=0.01, alpha=0.98):
        try:
            self.data[PITCH] = calc_pitch(
                acc_x_list=self.data[ACC_X],
                acc_y_list=self.data[ACC_Y],
                acc_z_list=self.data[ACC_Z],
                gyr_y_list=self.data[GYR_Y],
                alpha=alpha,
                dt=dt
            )
        except KeyError as ke:
            print(f"First calculate: {ke}")
        return self

    def add_roll(self, alpha=0.98):
        try:
            self.data[ROLL] = calc_roll(
                acc_y_list=self.data[ACC_Y],
                acc_z_list=self.data[ACC_Z],
                gyr_x_list=self.data[GYR_X],
                deltatime_list=self.data[DT],
                alpha=alpha
            )
        except KeyError as ke:
            print(f"First calculate: {ke}")
        return self

    def add_linear_acceleration(self):
        try:
            lin_acc_x, lin_acc_y, lin_acc_z = transform_to_linear_acceleration(
                pitch=self.data[PITCH],
                roll=self.data[ROLL],
                acc_x=self.data[ACC_X],
                acc_y=self.data[ACC_Y],
                acc_z=self.data[ACC_Z]
            )
            self.data[LIN_ACC_X] = lin_acc_x
            self.data[LIN_ACC_Y] = lin_acc_y
            self.data[LIN_ACC_Z] = lin_acc_z
        except KeyError as ke:
            print(f"First calculate: {ke}")
        return self

    def add_time_row(self, dt):
        row_counts = self.data.shape[0]
        self.data[TIME] = np.arange(0, row_counts * dt, dt)
        return self

    def add_magnitude(self):
        try:
            self.data[MAGNITUDE] = calc_magnitude(
                self.data[LIN_ACC_X],
                self.data[LIN_ACC_Y],
                self.data[LIN_ACC_Z]
            )
        except KeyError as ke:
            print('First calculate linear_acceleration')
        return self


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

    data = DataFrameTransformations(df)
    data.add_pitch().add_roll().add_linear_acceleration().add_magnitude().add_jerk().add_time_row(dt=0.01)
    print(data)
