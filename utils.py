import itertools

import numpy as np
import pandas as pd

from const import *


def calc_pitch(acc_x_list, acc_y_list, acc_z_list, gyr_y_list, dt, alpha):
    pitch = []
    iterations = min(len(acc_x_list), len(acc_y_list), len(acc_z_list), len(gyr_y_list))
    pitch_gyr = 0
    for index in range(iterations):
        acc_x = acc_x_list[index]
        acc_y = acc_y_list[index]
        acc_z = acc_z_list[index]
        gyr_y = gyr_y_list[index]

        pitch_acc = np.arctan2(-acc_x, np.sqrt(acc_y ** 2 + acc_z ** 2))
        pitch_gyr = pitch_gyr + gyr_y * dt
        pitch_gyr = alpha * pitch_gyr + (1 - alpha) * pitch_acc
        pitch.append(pitch_gyr)

    return pitch


def calc_roll(acc_y_list, acc_z_list, gyr_x_list, dt, alpha):
    roll_gyr = 0
    roll = []
    iterations = min(len(acc_y_list), len(gyr_x_list), len(acc_z_list))
    for index in range(iterations):
        acc_z = acc_z_list[index]
        acc_y = acc_y_list[index]
        gyr_x = gyr_x_list[index]
        roll_acc = np.arctan2(acc_y, acc_z)
        roll_gyr = roll_gyr + gyr_x * dt
        roll_gyr = alpha * roll_gyr + (1 - alpha) * roll_acc
        roll.append(roll_gyr)
    return roll


def get_position_from_rotation(pitch, roll, yaw=None):
    x = np.cos(roll)
    y = np.sin(pitch)
    z = np.sin(roll)
    return x, y, z





def calc_jerk(acc_x_list, acc_y_list, acc_z_list, dt):
    jerk = []
    iterations = min(len(acc_x_list), len(acc_y_list), len(acc_z_list))
    for index in range(iterations):
        acc_x = acc_x_list[index]
        acc_y = acc_y_list[index]
        acc_z = acc_z_list[index]
        a = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        if index == 0:
            jerk.append(a)
            continue

        last_acc_x = acc_x_list[index - 1]
        last_acc_y = acc_y_list[index - 1]
        last_acc_z = acc_z_list[index - 1]

        last_a = np.sqrt(last_acc_x ** 2 + last_acc_y ** 2 + last_acc_z ** 2)

        jerk.append((a - last_a) / dt)
    return jerk


def calc_magnitude(acc_x_list, acc_y_list, acc_z_list):
    magnitude = []
    iterations = min(len(acc_x_list), len(acc_y_list), len(acc_z_list))
    for index in range(iterations):
        acc_x = acc_x_list[index]
        acc_y = acc_y_list[index]
        acc_z = acc_z_list[index]
        a = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        magnitude.append(a)

    return magnitude


def transform_to_linear_acceleration(pitch, roll, acc_x, acc_y, acc_z):
    g = 9.81
    pitch = np.array(pitch)
    roll = np.array(roll)
    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)

    gravity_x = -g * np.sin(pitch)
    gravity_y = g * np.sin(roll) * np.cos(pitch)
    gravity_z = g * np.cos(roll) * np.cos(pitch)
    lin_acc_x = acc_x - gravity_x
    lin_acc_y = acc_y - gravity_y
    lin_acc_z = acc_z - gravity_z
    return lin_acc_x, lin_acc_y, lin_acc_z


def get_df_from_csv(path):
    df = pd.read_csv(path)
    return df


if __name__ == '__main__':
    df = get_df_from_csv(r'/data/data20260312_203448.csv')
    df['jerk'] = calc_jerk(df[ACC_X], df[ACC_Y], df[ACC_Z])
    df[PITCH] = calc_pitch(acc_x_list=df[ACC_X], acc_y_list=df[ACC_Y], acc_z_list=df[ACC_Z], gyr_y_list=df[GYR_Y],
                           alpha=0.98, dt=0.01)

    df[ROLL] = calc_roll(acc_y_list=df[ACC_Y], acc_z_list=df[ACC_Z], gyr_x_list=df[GYR_X], dt=0.01, alpha=0.98)
