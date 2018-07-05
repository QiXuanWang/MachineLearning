# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


def x_sin(x):
    return x * np.sin(x)


def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


def rnn_data(data, time_steps, labels=False, transpose=False):
    """
    creates new data frame based on previous observation
    when labels is True, this return labels data based on input data.
    labels are the expected value for every bucket
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> if labels == False: return [[1, 2], [2, 3], [3, 4]]
        -> if labels == True: return [3, 4, 5]
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        time_steps = 3
        -> if labels == False: return [[1, 2, 3], [2, 3, 4], [3, 4, 5]], ..., [7,8, 9 ]]
        -> if labels == True: return [4, 5, 6, ..., 10]
    """
    #import pdb
    #pdb.set_trace()
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                if transpose:
                    rnn_df.append(data.iloc[i + time_steps].as_matrix().transpose()[0])
                else:
                    rnn_df.append(data.iloc[i + time_steps].as_matrix())

            except AttributeError:
                if transpose:
                    rnn_df.append(data.iloc[i + time_steps].transpose()[0])
                else:
                    rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            if transpose:
                rnn_df.append(data_.transpose()[0])
            else:
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1, transpose=False):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels, transpose=transpose),
            rnn_data(df_val, time_steps, labels=labels, transpose=transpose),
            rnn_data(df_test, time_steps, labels=labels, transpose=transpose))


def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def generate_data(fct, x, time_steps, seperate=False, transpose=False):
    """generates data with based on a function fct
    returns data buckets(*_x) and labels(*_y)
    time_steps decide how many duplication is kept, not sure of the purpose/useage"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps, transpose=transpose)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True, transpose=transpose)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)
