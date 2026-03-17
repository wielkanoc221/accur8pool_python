import pandas as pd
import numpy as np

from const import ROLL

df = pd.read_csv(r'../data/andrzej rozbicie.csv')

dt = 0.01

roll = 0
pitch = 0

roll_list = []
pitch_list = []
time = np.arange(stop=df.shape[0] * dt, step=dt)
df['time'] = time
df['magnitude'] = np.sqrt(df['accx'] ** 2 + df['accy'] ** 2, df['accz'] ** 2)
for i in range(len(df)):
    ax = df.accx[i]
    ay = df.accy[i]
    az = df.accz[i]

    gx = df.gyrx[i]
    gy = df.gyry[i]

    # kąty z akcelerometru
    roll_acc = np.arctan2(ay, az)
    pitch_acc = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

    # integracja żyroskopu
    roll = roll + gx * dt
    pitch = pitch + gy * dt

    # filtr komplementarny
    roll = 0.98 * roll + 0.02 * roll_acc
    pitch = 0.98 * pitch + 0.02 * pitch_acc

    roll_list.append(roll)
    pitch_list.append(pitch)

df["roll"] = roll_list
df["pitch"] = pitch_list

import plotly.graph_objects as go
import plotly.express as px

# fig = go.Figure()
#
# fig = px.scatter_3d(
#     df,
#     x=np.cos(df.roll),
#     y=np.sin(df.pitch),
#     z=np.sin(df.roll),
#
#     color="time",
# )
#
# fig.show()
print(df.iloc[600:650][ROLL])