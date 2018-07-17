import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
import os
import time
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
from data import gen_data, save_plot

ctx = mx.gpu()

num_layers = 1    # multi-layer
batch_size = 16     # 

_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vi)']) 
data_scale = np.max(_data)
label_scale = np.max(_label)
_data = _data/data_scale
_label = _label/label_scale

sequence_length = len(_data[0])
num_hidden = sequence_length # too big will hang
print("Sequence_len: ", sequence_length)
#num_hidden = 64

data_len = len(_data)
m = int(0.9*data_len) 
m += m%batch_size
idx = np.random.choice(data_len, size=data_len, replace=False)
train_idx = idx[:m]
test_idx = idx[m:]

X = _data[train_idx, :]
y = _label[train_idx, :]
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=False)
print("train_data shape: ", X.shape, y.shape)

X = _data[test_idx, :]
y = _label[test_idx, :]
eval_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=1, shuffle=False)
print("eval_data shape: ", X.shape, y.shape)

net = gluon.nn.Sequential()
#net = gluon.nn.HybridSequential() # doesn't work since LSTM is not hybrid
# input: (sequence_length, batch_size, input_size)
# input_size is feature_size or vector dimension for one-hot word vector?
with net.name_scope():
    net.add(rnn.GRU(num_hidden, num_layers, layout='NTC')) # T: sequence_length, N: batch_size, C: feature_dimension
    net.add(nn.BatchNorm())
    net.add(nn.Dense(sequence_length)) #input: (batch, sequence_length, num_hidden) --> #output:(batch, sequence_length)

net.collect_params().initialize(mx.init.Normal(sigma=0.02), ctx=ctx)
print(net.collect_params)
square_loss = gluon.loss.L2Loss()
learning_settings = {'learning_rate': 0.005, 'momentum':0.9}
trainer = gluon.Trainer(net.collect_params(), 'sgd', learning_settings)
#metric = mx.metric.MSE()

epochs = 30
loss_sequence = []
for e in range(epochs):
    start = time.time()
    cumulative_loss = 0
    if e > 10:
        learning_settings = {'learning_rate': 0.001, 'momentum':0.9}
    for i, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx,)
        label = label.as_in_context(ctx,)
        with autograd.record():
            if data.shape[0] < batch_size:
                # don't know how to handle
                continue
            # LSTM requires TNC data by default, C: feature size is missing
            data = data.reshape((batch_size, sequence_length, 1))
            output = net(data) # out: (batch_size, sequence_length)
            #data.reshape((batch_size, sequence_length))
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch: %s, loss: %e, time: %.2f" % (e, cumulative_loss, time.time()-start))
    loss_sequence.append(cumulative_loss)

plt.figure(num=None, figsize=(8,6))
plt.plot(loss_sequence)
plt.grid(True, which="both")
plt.xlabel('epoch', fontsize=14)
plt.ylabel('average loss', fontsize=14)
plt.show()

#net.save_params('try2_load2.params')
params = net.collect_params()
params.save('try2.params')

def model_predict(net, data_iter):
    i = 0
    X = []
    y = []
    p = []
    for data,label in data_iter:
        data = data.as_in_context(ctx,)
        data = data.reshape((1, sequence_length, 1))
        output = net(data)
        data = data.reshape((1, sequence_length))
        X.append(data[0].asnumpy())
        y.append(label[0].asnumpy())
        p.append(output[0].asnumpy()) # since later we use _data directly
        if i == 0:
            print(data.shape, label.shape, output.shape)
            i += 1
    return X,y,p

X,y,predicted = model_predict(net, eval_iter)

save_plot(X*data_scale, y*label_scale, predicted*label_scale, 'try2.out')
