import numpy as np
import pandas as pd
import os

def gen_data(filename, input_list=None, output_list=None, seq_len=None, shape='nwc'):
    """
    shape:
        n: number of samples; 
        w: width or sequence length; 
            = seq_len for one sweep multiple cycles
            = original sequence length for mulitple sweep one cycle
        h: height or feature numbers
    """
    assert(isinstance(input_list, list))
    assert(isinstance(output_list, list))
    assert(os.path.exists(filename))
    #x = {'v(i)':[], 'v(pad)':[] }
    #initialize
    x = {}
    sig = None
    maxLen = 0
    for sig in input_list + output_list:
        x[sig] = []
    with open(filename, 'r') as fd:
        for line in fd.readlines():
            l = line.split()
            sig = l[0]
            if sig not in input_list + output_list:
                continue
            sweep = l[1]
            vals = [np.float32(v) for v in l[2:]]
            if len(vals) > maxLen:
                maxLen = len(vals)
            x[sig].append(vals)
    # padding
    maxLen += 8 - maxLen%8
    #print("maxLen: %d"%maxLen)
    for key,val in x.items():
        for l in val:
            l.extend([l[-1]]*(maxLen-len(l)))
    #build array
    sig = list(x.keys())[0]
    #import pdb
    #pdb.set_trace()
    if len(x[sig]) > 1: # multiple run, each sequence is one seq
        arrays = [np.array(x[sig], dtype=np.float32) for sig in input_list]
        X = np.stack(arrays) # (sig#, n_samples, seq_len)
        X = X.transpose([1, 2, 0])  # (n ,w ,c)
        arrays = [np.array(x[sig], dtype=np.float32) for sig in output_list]
        y = np.stack(arrays)
        y = y.transpose([1, 2, 0])
    elif len(x[sig]) == 1: # one run, need gen sequence
        assert(seq_len and seq_len>1)
        arrays = [np.array(x[sig][0], dtype=np.float32) for sig in input_list]
        X = np.stack(arrays).T # (total_seq_len, sig#)
        arrays = [np.array(x[sig][0], dtype=np.float32) for sig in output_list]
        y = np.stack(arrays).T
        X = np.atleast_3d([X[start:start+seq_len] for start in range(X.shape[0] - seq_len)]) # (no_of_sample, seq_len, sig#)
        y = np.atleast_3d(np.array([y[start:start+seq_len] for start in range(y.shape[0] - seq_len)], dtype=np.float32))
    #print("Shape: ", X.shape)
    if shape == 'ncw':
        X = X.transpose([0,2,1])
        y = y.transpose([0,2,1])
    return X, y

def normalized(a, axis=-1, order=2):
    """
    https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    normalize current when needed
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a/np.expand_dims(l2, axis)

def normalize(a, axis=1):
    """
    axis=1: (n,w,c)
    axis=2: (n,c,w)
    """
    scale = np.amax(a, axis)
    for i in range(a.shape[0]):
        if axis==1:
            a[i] = a[i]/scale[i]
        elif axis==2:
            a[i] = (a[i].T/scale[i]).T
    return scale

def denormalize(a, scale, axis=1):
    """
    axis=1: (n,w,c)
    axis=2: (n,c,w)
    scale is (n,1) shape??
    """
    assert(a.shape[0] == scale.shape[0]) # number of sample is same
    for i in range(a.shape[0]):
        if axis == 1:
            a[i] = a[i] * scale[i][0]
        elif axis == 2:
            a[i] = (a[i].T * scale[i][0]).T
    return a

def duplicate(a, times=2):
    """
    duplicate samples
    """
    return np.concatenate([a]*times)


def save_plot(X, y, predicted, filename, style="mxnet"):
    """
    fro keras: input: (n, seq, c)
    for mxnet: input is: [(sample, seq)]
    """
    if style == "keras":
        index = np.random.randint(1, X.shape[0], 20) 
        # squeeze is for keras output
        X = X.squeeze()[index]
        y = y.squeeze()[index]
        predicted = predicted.squeeze()[index]
    if style == "mxnet":
        assert(isinstance(X, list))
        assert(isinstance(X[0], np.ndarray))
        X = np.stack(X[:10])
        y = np.stack(y[:10])
        predicted = np.stack(predicted[:10])
    d={'x':X, 'y': y, 'p':predicted}
    df = pd.DataFrame().append(d, ignore_index=True)
    df.to_pickle(filename)


def test1():
    # 1 cycle, 1 in 1 out, multiple run example
    # changed to output (nwc) format instaed of 2-d format
    _data, _label = gen_data('1cycle_short.txt', input_list=['v(i)'], output_list=['v(pad)'])
    import pdb
    pdb.set_trace()
    assert(_data.shape == (200, 552, 1))
    assert(np.abs(_data[0][120][0]-1.2)<1e-6)
    assert(np.abs(_label[0][350][0]-1.0712)<1e-6)

def test2():
    _data, _label = gen_data('500cycle.txt', input_list=['v(i)', 'i(vi)'], output_list=['v(pad)','i(vpad)'], seq_len=6)
    print(_data[0][1])
    assert(_data.shape == (97434, 6, 2))
    assert(np.abs(_data[0][1][0]-0.12)<1e-6)
    assert(np.abs(_data[0][1][1]+0.0495477)<1e-6)

def test3():
    _data, _label = gen_data('500cycle.txt', input_list=['v(i)'], output_list=['v(pad)'], seq_len=6)
    print(_data[0][1])
    assert(_data.shape == (97434, 6, 1))
    assert(np.abs(_data[0][1][0]-0.12)<1e-6)
    assert(np.abs(_data[0][1][1]+0.0495477)<1e-6)



if __name__ == '__main__':
    test1()
    #test2()
    #test3()
