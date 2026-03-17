import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv(r"C:\Users\apietka\Desktop\acc data\data20260312_203942.csv")


def off_grav(acc):
    alpha = 0.9

    g = np.zeros_like(acc)

    for i in range(1, len(acc)):
        g[i] = alpha * g[i - 1] + (1 - alpha) * acc[i]

    linear_acc = acc - g
    return linear_acc


time_delta = 0.01
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
time = np.arange(stop=df.shape[0] * time_delta, step=time_delta)
df['time'] = time
df['accx'] = off_grav(df['accx'])
df['accy'] = off_grav(df['accy'])
df['accz'] = off_grav(df['accz'])
df['magnitude'] = np.sqrt(df['accx'] ** 2 + df['accy'] ** 2, df['accz'] ** 2)
df['diff'] = df['magnitude'].shift(-1) - df['magnitude']

print(df)
# sns.lineplot(x='time', y='magnitude', data=df, label='magnitude')
# sns.lineplot(x='time', y='accx', data=df, label='accx')
# sns.lineplot(x='time', y='accy', data=df, label='accy')
# sns.lineplot(x='time', y='accz', data=df, label='accz')
# sns.lineplot(x='time', y='gyrx', data=df, label='gyrx')
# sns.lineplot(x='time', y='gyry', data=df, label='gyry')
# sns.lineplot(x='time', y='gyrz', data=df, label='gyry')
# sns.lineplot(x='time', y='diff', data=df, label='diff')
#
# plt.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['time'], y=df['accx'], mode="lines", name="ax"))
fig.add_trace(go.Scatter(x=df['time'], y=df['accy'], mode="lines", name="ay"))
fig.add_trace(go.Scatter(x=df['time'], y=df['accz'], mode="lines", name="az"))
fig.add_trace(go.Scatter(x=df['time'], y=df['magnitude'], mode="lines", name="magnitude"))
fig.add_trace(go.Scatter(x=df['time'], y=df['diff'], mode="lines", name="diff"))
fig.add_trace(go.Scatter(x=df['time'], y=df['gyrx'], mode="lines", name="gyrx"))
fig.add_trace(go.Scatter(x=df['time'], y=df['gyry'], mode="lines", name="gyry"))
fig.add_trace(go.Scatter(x=df['time'], y=df['gyrz'], mode="lines", name="gyrz"))
fig.show()
