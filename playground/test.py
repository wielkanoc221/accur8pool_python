import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from const import *
from utils import lowpass_filter
import plotly.graph_objects as go

df = pd.read_csv(r'C:\Users\apietka\PycharmProjects\accur8pool\data\andrzej rozbicie.csv')
fig = go.Figure()
# fig = px.line(df, y=[ACC_X, ACC_Y, ACC_Z, GYR_X, GYR_Y, GYR_Z])
# fig.show()
fitred_x = lowpass_filter(df[ACC_X], cutoff=5, fs=100)
fitred_y = lowpass_filter(df[ACC_Y], cutoff=5, fs=100)
fitred_z = lowpass_filter(df[ACC_Z], cutoff=5, fs=100)
# px.line(x=fitred_x)
# px.line(x=fitred_y)
# px.line(x=fitred_z)
# px.show
fig.add_trace(go.Scatter(y=fitred_x,text='siema',mode='markers'))
#fig.add_trace(fitred_y, row=1, col=1)
fig.show()