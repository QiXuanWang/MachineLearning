# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np

class DataIter(object):
    def __init__(self, y, x, seq_len, use_last=True, seq_type="NTC"):
        pass

    def dump(self):
        print("data:")
        print(self._data)
        print("rnn data: (%d, %d)"%self.data.shape)
        print(self.data)
        print("rnn labels: (%d, %d)"%self.labels.shape)
        print(self.labels)

    def next(self, batch_size):
        if self.batch_id == len(self.data):
           self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
            batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
            batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
            batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

    def describe(self):
        print("Data: ",self.data.shape)
        print("Label: ",self.labels.shape)

    @property
    def provide_data(self):
        return self.data

    @property
    def provide_label(self):
        return self.labels

    def gen_data(self):
        """
        return np.array object
        data shape: (n-seq_len, seq_len)
        label shape: (n-seq_len, seq_len) or (n-seq_len, 1) if use_last
        """
        return self.data, self.labels

    def raw_data(self):
        """
        return raw sequence without label
        """
        return self._data

class NumpyIter(DataIter):
    """
    Input:
      y: np.array, label
      x: np.array, input
    """
    def __init__(self, y, x, seq_len, use_last=True, seq_type="NTC"):
        assert(isinstance(y, np.array))
        assert(isinstance(x, np.array))
        self._data = y
        self.data, self.labels = rnn_data3(y, x, seq_len, use_last)

class DataFuncIter(DataIter):
    """
    Currently we use fixed seq_len, we hope to use dynamic length
    """
    def __init__(self, func, x, seq_len, use_last=True, seq_type="TNC"):
        assert(len(x)>seq_len)
        self._data = func(x) # raw data
        if seq_type == "TN":
            self.data, self.labels = rnn_data(self._data, seq_len, use_last)
        elif seq_type == "TNC":
            self.data, self.labels = rnn_data2(self._data, seq_len, use_last)
        else:
            print("Error: seq_type should be TN (sample#,seq_len) or TNC (sample#, time_step, 1(feature#))")
        self.seqlen = [seq_len]*len(self.data)
        self.batch_id = 0

def square_x(x):
    return x*x

def log_sin(x):
    return np.log(x) + np.sin(x)

def x_sin(x):
    return x * np.sin(x)


def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


def rnn_data3(y, x, seq_len):
    """
    Input:
        y: np.array, labels, e.g. (874, 803)
        x: np.array, data, e.g. (874, 803)
    Output:
        y: np.array, labels, with seq_len each, ( , seq_len)
        x: np.array, data, with seq_len each, ( , seq_len)
    """
    data = []
    label = []
    for i in range(len(x)):
        sample_x = x[i]
        sample_y = y[i]
        j = 0
        while (j+seq_len) < len(sample_x):
            data.append(sample_x[j:j+seq_len])
            label.append(sample_y[j:j+seq_len])
            j += 1
    return np.array(data), np.array(label)


def rnn_data(data, seq_len, use_last=False):
    """
    creates new data frame based on previous observation
    labels are the expected value for every bucket
      * example:
        l = [1, 2, 3, 4, 5]
        seq_len = 2
        -> Data: [[1, 2], [2, 3], [3, 4]]
        -> Labels: [[2,3], [3, 4], [4,5]]
            -> if use_last: [[3], [4], [5]]
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        seq_len = 3
        -> Data: [[1, 2, 3], [2, 3, 4], [3, 4, 5]], ..., [7,8, 9 ]]
        -> Labels: [4, 5, 6, ..., 10]
    """
    #import pdb
    #pdb.set_trace()
    rnn_df = []
    label_df= []
    for i in range(len(data) - seq_len):
        rnn_df.append(data[i:i+seq_len])
        if use_last:
            label_df.append(data[i+seq_len])
        else:
            label_df.append(data[i+1:i+1+seq_len])

    return np.array(rnn_df, dtype=np.float32), np.array(label_df, dtype=np.float32)

def rnn_data2(data, seq_len, use_last=False):
    """
    here seq_len is more of time_steps
    creates new data frame based on previous observation
    labels are the expected value for every bucket
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> Data: [[[1], [2]], [[2], [3]], [[3], [4]]]
        -> Labels: [[[2], [3]], [[3], [4]], [[4],[5]]]
            -> if use_last: [3, 4, 5]
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        time_steps = 3
        -> Data: [[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]], ..., [[7],[8], [9]]]
        -> Labels: [[4], [5], [6], ..., [10]]
    """
    #import pdb
    #pdb.set_trace()
    rnn_df = []
    label_df= []
    for i in range(len(data) - seq_len):
        rnn_df.append([ [item] for item in data[i:i+seq_len] ])
        if use_last:
            label_df.append([data[i+seq_len]])
        else:
            label_df.append([ [item] for item in data[i+1:i+1+seq_len] ])

    return np.array(rnn_df, dtype=np.float32), np.array(label_df, dtype=np.float32)



# not used
def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


# not used
def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1, transpose=False):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels, transpose=transpose),
            rnn_data(df_val, time_steps, labels=labels, transpose=transpose),
            rnn_data(df_test, time_steps, labels=labels, transpose=transpose))


    # not used
def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


# not used
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


if __name__ == "__main__":
    x = np.linspace(1, 100, 100)
    myIter = DataFuncIter(np.square, x, 3, use_last=False)
    myIter.describe()
    #myIter.dump()
    #print(myIter.next(2))
import numpy as np
import pandas as pd
import os

