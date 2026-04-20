from matplotlib import pyplot as plt
import plotly.graph_objects as go
from pandas import DataFrame
import plotly.express as px
from const import *
from utils import *
from utils import _normalize, _smooth


# ---------------------
# Funkcja do wczytania CSV
# ---------------------
def get_df_from_csv(path):
    df = pd.read_csv(path)
    return df


# ---------------------
# Klasa transformacji
# ---------------------
class DataFrameTransformations:
    def __init__(self, data: DataFrame):
        self.data = data.copy()  # pracujemy na kopii

    def dt2sec(self):
        try:
            # Zabezpieczenie: dzielimy tylko jeśli dane są w ms (duże wartości)

            self.data[DT] = self.data[DT] / 1000
        except Exception as e:
            print(f'Błąd zamiany DT z ms na sekundy: {e}')
        return self

    # usuń pierwszy wiersz
    def drop_first_row(self):
        self.data = self.data.iloc[1:].reset_index(drop=True)
        return self

    def normalize(self, columns):
        for col in columns:
            self.data[col] = _normalize(self.data[col])
        return self

    # oblicz pitch
    def add_pitch(self, alpha=0.98):
        try:
            self.data[PITCH] = calc_pitch(
                acc_x_list=self.data[ACC_X].tolist(),
                acc_y_list=self.data[ACC_Y].tolist(),
                acc_z_list=self.data[ACC_Z].tolist(),
                gyr_y_list=self.data[GYR_Y].tolist(),
                dt_list=self.data[DT].tolist(),
                alpha=alpha
            )
        except Exception as e:
            print(f"Błąd obliczania pitch: {e}")
        return self

    def add_yaw(self):
        self.data[YAW]= calc_yaw_complementary(
            self.data[MAG_X],
            self.data[MAG_Y],
            self.data[MAG_Z],
            self.data[GYR_Z],
            self.data[ROLL],
            self.data[PITCH],
            self.data[DT]

        )
        return self

    # oblicz roll
    def add_roll(self, alpha=0.98):
        try:
            self.data[ROLL] = calc_roll(
                acc_y=self.data[ACC_Y].tolist(),
                acc_z=self.data[ACC_Z].tolist(),
                gyr_x=self.data[GYR_X].tolist(),
                dt=self.data[DT].tolist(),
                alpha=alpha
            )
        except Exception as e:
            print(f"Błąd obliczania roll: {e}")
        return self

    def cut_by_time(self, timestart, timestop=None):
        df = self.data

        index_start = (df["time"] - timestart).abs().idxmin()

        if timestop is not None:
            index_stop = (df["time"] - timestop).abs().idxmin()

            if index_start > index_stop:
                index_start, index_stop = index_stop, index_start

            indices_to_drop = df.loc[index_start:index_stop].index
        else:
            indices_to_drop = df.loc[index_start:].index

        self.data = df.drop(index=indices_to_drop).reset_index(drop=True)

        return self

    # oblicz magnitude
    def add_magnitude(self, source_cols, new_col):
        try:

            self.data[new_col] = calc_magnitude(
                self.data[source_cols[0]],
                self.data[source_cols[1]],
                self.data[source_cols[2]]
            )
        except Exception as e:
            print(f"Błąd add_magnitude: {e}")
        return self

    def add_jerk(self, source_cols):
        try:
            # Do poprawnego działania np.gradient potrzebujemy osi czasu (TIME), nie delt (DT)
            if TIME not in self.data.columns:
                self.add_time_row()

            time_axis = self.data[TIME].values

            # Źródło danych do jerka
            source_cols = source_cols

            acc = self.data[source_cols].values

            # np.gradient z podanym time_axis poprawnie oblicza d(acc)/dt
            jerk = np.gradient(acc, time_axis, axis=0)

            for i, col in enumerate(source_cols):
                self.data[f"{JERK}_{col.split('_')[-1]}"] = jerk[:, i]

            self.data[JERK + "_MAG"] = np.linalg.norm(jerk, axis=1)
        except Exception as e:
            print(f"Błąd add_jerk: {e}")
        return self

    def lowpass_filter(self, columns: list, cutoff=8):
        for column in columns:
            self.data[column] = lowpass_filter(self.data[column], cutoff=cutoff)
        return self

    def smooth(self, columns, window):
        for col in columns:
            self.data[col] = _smooth(self.data[col], window=window)
        return self

    # dodaj kolumnę czasu w sekundach
    def add_time_row(self):
        try:
            self.data[TIME] = np.cumsum(self.data[DT])
        except Exception as e:
            print(f'Bład obliczania TIME: {e}')
        return self

    # ustawienie wyświetlania
    def set_display_max_data(self):
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

    # wykres FFT
    def plot_fft(self):
        source_cols = [LIN_ACC_X, LIN_ACC_Y, LIN_ACC_Z]
        if not all(c in self.data.columns for c in source_cols):
            source_cols = [ACC_X, ACC_Y, ACC_Z]

        N = self.data.shape[0]
        dt_sec = self.data[DT].mean()
        for axis, col in zip(["X", "Y", "Z"], source_cols):
            signal = self.data[col].values
            fft_values = np.abs(np.fft.fft(signal))[:N // 2]
            fft_freqs = np.fft.fftfreq(N, d=dt_sec)[:N // 2]
            plt.plot(fft_freqs, fft_values, label=axis)

        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Amplituda")
        plt.legend()
        plt.show()

    # wykres
    def plot_linear_acc(self):
        fig = go.Figure()
        time = self.data[TIME]

        source_cols = [LIN_ACC_X, LIN_ACC_Y, LIN_ACC_Z]
        if not all(c in self.data.columns for c in source_cols):
            source_cols = [ACC_X, ACC_Y, ACC_Z]

        fig.add_trace(go.Scatter(x=time, y=self.data[source_cols[0]], mode='lines', name='X'))
        fig.add_trace(go.Scatter(x=time, y=self.data[source_cols[1]], mode='lines', name='Y'))
        fig.add_trace(go.Scatter(x=time, y=self.data[source_cols[2]], mode='lines', name='Z'))

        fig.update_layout(title="Linear acceleration",
                          xaxis_title="Time [s]",
                          yaxis_title="Acceleration [m/s^2]")
        fig.show()


# ---------------------
# Pipeline
# ---------------------
if __name__ == '__main__':
    path = r'data/danejaroty.csv'
    try:
        df = get_df_from_csv(path)
        data = DataFrameTransformations(df)
        (data.drop_first_row()
         .dt2sec()
         .add_time_row()
         .cut_by_time(1)
         .lowpass_filter(columns=[ACC_X, ACC_Y, ACC_Z], cutoff=10)
         .add_magnitude(source_cols=[ACC_X, ACC_Y, ACC_Z], new_col=ACC_MAGNITUDE)
         .add_magnitude(source_cols=[GYR_X, GYR_Y, GYR_Z], new_col=GYR_MAGNITUDE)
         .add_jerk(source_cols=[ACC_X, ACC_Y, ACC_Z])
         .add_roll()
         .add_pitch()
        .add_yaw()
         .normalize([GYR_X, GYR_Y, GYR_Z, ROLL, PITCH, ACC_MAGNITUDE, GYR_MAGNITUDE])

         .smooth([ACC_X, ACC_Y, ACC_Z, ACC_MAGNITUDE, GYR_MAGNITUDE, 'jerk_MAG', 'roll', 'pitch'], window=5)

         )

        data.set_display_max_data()

        df_res = data.data
        print(data.data.shape)

        print(data.data.shape)
    except Exception as e:
        print(f"Błąd: {e}")
    #     fig = px.line(df_res, x=df_res[TIME],
    #                   y=[ACC_X, ACC_Y, ACC_Z, 'jerk_MAG', 'jerk_X', 'jerk_Y', 'jerk_Z', ROLL, PITCH, ACC_MAGNITUDE,
    #                      GYR_MAGNITUDE])
    #
    #     fig.show()
    # except Exception as e:
    #     print(f"Błąd: {e}")
