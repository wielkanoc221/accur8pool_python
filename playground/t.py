import pandas
import pandas as pd
from ahrs.filters import Madgwick
import numpy as np
from matplotlib import pyplot as plt

from const import ACC_X, ACC_Y, ACC_Z, GYR_Z, GYR_Y, GYR_X, MAG_X, MAG_Z, MAG_Y

Q = []
q = np.array([1, 0, 0, 0])  # start

df = pandas.read_csv(r'C:\Users\apietka\PycharmProjects\accur8pool\data\data20260317_214122.csv')

acc = np.column_stack((df[ACC_X], df[ACC_Y], df[ACC_Z]))
gyro = np.column_stack((df[GYR_X], df[GYR_Y], df[GYR_Z]))
mag = np.column_stack((df[MAG_X], df[MAG_Y], df[MAG_Z]))
dt = 0.01  # 100 Hz

from scipy.signal import butter, filtfilt


def lowpass(data, cutoff=8, fs=100):
    b, a = butter(2, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, data, axis=0)


acc_f = lowpass(acc, 10)
gyro_f = lowpass(gyro, 10)
mag_f = lowpass(mag, 5)  # mag wolniejszy

from ahrs.filters import Madgwick
import numpy as np

madgwick = Madgwick(sampleperiod=dt, beta=0.05)

Q = []
q = np.array([1, 0, 0, 0])

for i in range(len(acc_f)):
    q = madgwick.updateMARG(q,
                            gyr=gyro_f[i],
                            acc=acc_f[i],
                            mag=mag_f[i])
    Q.append(q)

Q = np.array(Q)

from scipy.spatial.transform import Rotation as R

# konwersja (WAŻNE!)
Q_scipy = np.column_stack([Q[:, 1], Q[:, 2], Q[:, 3], Q[:, 0]])

rot = R.from_quat(Q_scipy)

acc_world = rot.apply(acc_f)

g = np.array([0, 0, 9.81])
acc_linear = acc_world - g

angles = rot.as_euler('xyz', degrees=True)

roll = angles[:, 0]
pitch = angles[:, 1]
yaw = angles[:, 2]

import matplotlib.pyplot as plt

plt.figure()
plt.plot(acc_linear[:, 0], label="X world")
plt.plot(acc_linear[:, 1], label="Y world")
plt.plot(acc_linear[:, 2], label="Z world")
plt.legend()
plt.figure()
plt.plot(gyro_f[:, 0], label="roll rate")
plt.plot(gyro_f[:, 1], label="pitch rate")
plt.plot(gyro_f[:, 2], label="yaw rate")
plt.legend()
plt.title("Gyro")
plt.title("Linear acceleration (world)")
plt.show()

df = pd.DataFrame()
