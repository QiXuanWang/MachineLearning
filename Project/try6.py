import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
import os
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, contrib
from data import gen_data, save_plot

ctx = mx.gpu()

batch_size = 16
#_data, _label = gen_data(filename='1cycle.txt',input_list=['v(i)'], output_list=['v(pad)'])
_data, _label = gen_data(filename='1cycle_short.txt',input_list=['v(i)'], output_list=['v(pad)'])
data_len = len(_data)
sequence_length = len(_data[0])
#num_hidden = int(sequence_length/2)
#num_hidden = 128
print("Len: ",data_len)

m = int(0.8*data_len) 
m += m%batch_size

X = _data[:m]
y = _label[:m]
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)
print("train_data shape: ", X.shape, y.shape)

X = _data[m:]
y = _label[m:]
#idx = np.random.choice(_data.shape[0], 10, replace=False)
#X = _data[idx, :]
#y = _label[idx, :]
eval_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=1, shuffle=True)
print("eval_data shape: ", X.shape, y.shape)

num_layers = 1    # multi-layer
num_hidden = 8    # 
def get_net(num_hidden, num_layers, dense_layers):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(rnn.LSTM(num_hidden, num_layers, layout='NTC'))
        net.add(nn.BatchNorm())
        for i in range(dense_layers-1):
            net.add(nn.Dense(sequence_length, activation='relu'))
            #net.add(nn.BatchNorm())
        net.add(nn.Dense(sequence_length))
    return net

print("Build net")
net = get_net(8, 1, 2)
print("Initilize")
net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=ctx)
#square_loss = gluon.loss.HuberLoss(rho=0.1)
square_loss = gluon.loss.L2Loss()
learning_settings = {'learning_rate': 0.005}
trainer = gluon.Trainer(net.collect_params(), 'adam', learning_settings)

print("Begin training")
epochs = 20
loss_sequence = []
for e in range(epochs):
    import pdb
    pdb.set_trace()
    cumulative_loss = 0
    if e > 10:
        learning_settings = {'learning_rate': 0.001}
    for i, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            if data.shape[0] < batch_size:
                # don't know how to handle
                continue
            #data = data.reshape((batch_size, sequence_length, 1))
            output = net(data) # out: (batch_size, sequence_length)
            #output.reshape((batch_size, sequence_length))
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
params.save('try6.params')

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
save_plot(X,y,predicted,'try6.out')
