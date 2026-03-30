import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import butter

from const import *
from dataframe_transformations import DataFrameTransformations

df = pd.read_csv(r'C:\Users\apietka\PycharmProjects\accur8pool\data\data20260317_214122.csv')
data = DataFrameTransformations(df)
data.dt2sec().add_magnitude().add_jerk()
probkowanie = 100
dt = 0.1
cuttoff = 0.5
(b, a) = butter(4, cuttoff / (probkowanie / 2), btype='low')
filtered_x = scipy.signal.filtfilt(b, a, data.data[ACC_Z])
print(np.mean(filtered_x))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax3.plot(data.data[ACC_Z])
ax1.plot(data.data[JERK + '_Z'])
ax2.plot(filtered_x)
plt.show()
