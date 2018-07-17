import pdb
import sys,os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import gen_data
import argparse 

def plot_raw(f, input_list, output_list, sweep=None):
    _data, _label = gen_data(filename=f, input_list=input_list, output_list=output_list)
    assert(_data.shape[2]==1)
    assert(_label.shape[2]==1)
    _data = _data.squeeze(axis=-1)
    _label = _label.squeeze(axis=-1)
    print(_data.shape)
    sweepNo = _data.shape[0]
    if sweep is None:
        sweep = np.random.choice(sweepNo, size=10)
    else:
        assert(isinstance(sweep, list))
    print("Sweeps: ",sweep)
    for i in sweep:
        indexes = np.arange(_data[i].size)
        plt.figure(i)
        #plt.plot(indexes, _data[i], label='X')
        plt.plot(indexes, _label[i], label='y')
        plt.legend(loc='upper right')
    plt.show()


def plot_pickle_out(f, sweep=None, shape='nwc'):
    """
    for 1d data, X is (n,w) shape
    """
    assert(shape in ['nwc', 'ncw'])
    df = pd.read_pickle(f)
    X = df.x[0]
    y = df.y[0]
    predicted = df.p[0]
    data_shape = X.shape
    if len(data_shape) > 3:
        print("Max dimension that can be handled is 3")
        sys.exit(1)
    elif len(data_shape) == 2: # v or i
        print("ndim == 2, single signal plot")
        plot_1d(X,y,predicted,sweep=sweep)
    elif len(data_shape) == 3: # v and i
        print("ndim == 3, multiple signal plot")
        plot_2d(X,y,predicted, sweep=sweep, shape=shape)

def plot_1d(X, y, predicted, sweep=None):
    sweepNo = X.shape[0]
    indexes = np.arange(len(X[0]))
    print("Total sweeps: %d"%sweepNo)
    if sweep is None:
        sweep = np.random.choice(sweepNo, size=min(10,sweepNo))
    else:
        assert(isinstance(sweep, list))
        assert(max(sweep)<sweepNo)
    for i in sweep:
        plt.figure(i)
        #plt.plot(indexes, X[i], label='X')
        plt.plot(indexes, y[i], label='y')
        plt.plot(indexes, predicted[i], '-r', label='pred')
        plt.legend(loc='upper right')
        if i>10:
            print("data contains more than 10 plots. double check")
            break
    plt.show()

def plot_2d(X, y, predicted, sweep=None, shape='nwc'):
    if shape == 'ncw':
        X = np.transpose(X, axes=(0,2,1))
        y = np.transpose(y, axes=(0,2,1))
        predicted = np.transpose(predicted, axes=(0,2,1))
    assert(X.shape[1] > X.shape[2])
    indexes = np.arange(len(X[0]))
    print("Total sweeps: %d"%X.shape[0])
    for idx in range(X.shape[2]): # feature
        newX = X[:,:,idx]
        newy = y[:,:,idx]
        newP = predicted[:,:,idx]

        sweepNo = newX.shape[0]
        if sweep is None:
            sweep = np.random.choice(sweepNo, size=min(10, sweepNo))
        else:
            assert(isinstance(sweep, list))
            assert(max(sweep)<sweepNo)
        for i in sweep: # sample
            plt.figure(i)
            #plt.plot(indexes, newX[i], label='X')
            plt.plot(indexes,  newy[i], label='y')
            plt.plot(indexes, newP[i], '-r', label='pred')
            plt.legend(loc='upper right')
            if i>10:
                print("data contains more than 10 plots. double check")
                break
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="input file")
    parser.add_argument("--input", type=str, default="v(i)", help="input signal, usually v(i)")
    parser.add_argument("--output", type=str, default="i(vi)", help="output signal, usually v(pad),i(vi),i(vpad)")
    parser.add_argument("--sweeps", type=str, default="",  help="sweeps to be plotted, could be intergers, or random")
    parser.add_argument("--shape", type=str, default="nwc",  help="input data shape")
    parser.add_argument("-raw", action='store_true', default=False, help="plot raw data")
    parser.add_argument("-pickle", action='store_true', default=True,  help="plot predicted data in pickle format")
    args = parser.parse_args()
    data_file = args.file
    assert(os.path.exists(data_file))
    assert(args.shape in ['ncw', 'nwc'])
    if args.sweeps != "":
        sweeps  = list(map(int, args.sweeps.split(',')))
        if len(sweeps) == 1:
            sweeps = list(range(int(sweeps[0])))
    else:
        sweeps = None
    if args.raw:
        input_list = args.input.split(",")
        output_list = args.output.split(",")
        plot_raw(data_file, input_list, output_list, sweep=sweeps)
    else:
        plot_pickle_out(data_file, sweep=sweeps, shape=args.shape)
