# Not working yet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, contrib
from data import gen_data, save_plot
from Model import UpSampling1D

#Yu: This TCN is based on https://github.com/colincsl/TemporalConvolutionalNetworks

ctx = mx.gpu()

num_layers = 1    # multi-layer
batch_size = 64     # 
#sequence_length = 6 # seq len
num_hidden = 8    # 

#_data, _label = gen_data(filename='500cycle.txt',input_list=['v(i)'], output_list=['v(pad)'], seq_len=6)
_data, _label = gen_data('1cycle_short.txt', input_list=['v(i)'], output_list=['v(pad)']) # nwc
data_len = len(_data)
sequence_length = len(_data[0])
print("Len: ",data_len)

m = int(0.8*data_len) 
m += m%batch_size

X = _data[:m]
y = _label[:m]
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)
print("train_data shape: ", X.shape, y.shape)

X = _data[m:]
y = _label[m:]
eval_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=1, shuffle=True)
print("eval_data shape: ", X.shape, y.shape)

nb_filter = 16
filter_length = 20
net = gluon.nn.Sequential()
with net.name_scope():
    # encoder
    net.add(nn.Conv2D(nb_filter, filter_length, padding=0))
    net.add(nn.AvgPool2D(pool_size=2))
    net.add(nn.BatchNorm())
    net.add(nn.Conv2D(int(nb_filter/2), filter_length, padding=0))
    net.add(nn.AvgPool2D(pool_size=2))
    net.add(nn.BatchNorm())

    # decoder
    net.add(nn.Conv2D(int(nb_filter/2), filter_length, padding=0))
    net.add(UpSampling1D(scale=2, sample_type='nearest'))
    net.add(nn.BatchNorm())
    net.add(nn.Conv2D(nb_filter, filter_length, padding=0))
    net.add(UpSampling1D(scale=2, sample_type='nearest'))
    net.add(nn.BatchNorm())
    net.add(nn.Dense(sequence_length))

#net.collect_params().initialize(mx.init.Normal(sigma=0.1), ctx=ctx)
net.initialize(mx.init.Normal(sigma=0.1), ctx=ctx)
square_loss = gluon.loss.L1Loss()
learning_settings = {'learning_rate': 0.02, 'momentum':0.9}
trainer = gluon.Trainer(net.collect_params(), 'sgd', learning_settings)
epochs = 15
loss_sequence = []
for e in range(epochs):
    cumulative_loss = 0
    if e > 10:
        learning_settings = {'learning_rate': 0.002, 'momentum':0.9}
    for i, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            if data.shape[0] < batch_size:
                # don't know how to handle
                continue
            # for conv2d, we need 'nchw' shape
            data = data.reshape((batch_size, sequence_length, 1))
            #data = data.reshape((batch_size, 1, sequence_length))
            print("Input: ", data.shape)
            output = net(data) # out: (batch_size, sequence_length)
            print("Output: ",output.shape)
            output.reshape(label.shape)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch: %s, loss: %e" % (e, cumulative_loss))
    loss_sequence.append(cumulative_loss)

plt.figure(num=None, figsize=(8,6))
plt.plot(loss_sequence)
plt.grid(True, which="both")
plt.xlabel('epoch', fontsize=14)
plt.ylabel('average loss', fontsize=14)
plt.show()

params = net.collect_params()
params.save('try5.params')
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

save_plot(X, y, predicted, 'try5.out')
