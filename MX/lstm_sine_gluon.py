import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import os
from data_processing import DataFuncIter
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

import logging
logging.getLogger().setLevel(logging.DEBUG)

ctx = mx.cpu()
model_ctx = mx.cpu()

num_hidden = 5    # use 5 cells in hidden layer
batch_size = 2     # only one sine wave per batch
#data_length = 100  #  data len
data_cycle = 20     # 10*60 data points
sequence_length = 6 # seq len
sample_step_in_degrees = 6

#data_shape = (data_length - sequence_length, sequence_length)
#x = np.linspace(1, data_length, data_length) # not smooth
x = np.arange(0,360*data_cycle,sample_step_in_degrees) # 60 points per cycle
x = (2*np.pi * x/360.0)/2
myIter = DataFuncIter(np.sin, x, sequence_length, use_last=True, seq_type="TNC")
myIter.describe()
_data, _label = myIter.gen_data() # [[seq_len of data...], total data_len]
data_len = len(_data)
m = int(0.8*data_len)
#print(_data[:m])
#print(_label[:m])

#train_iter = mx.io.NDArrayIter(
        #data = _data[:m],
        #label = _label[:m], 
        #batch_size = batch_size, 
        #shuffle=False,
        #label_name='lro_label')
#eval_iter = mx.io.NDArrayIter(
#        data = _data[m:], 
#        label = _label[m:], 
#        batch_size = batch_size, 
#        shuffle=False)

X = _data[:m]
y = _label[:m]
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=False)
print("train_data shape: ", X.shape, y.shape)

X = _data[m:]
y = _label[m:]
eval_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=1, shuffle=False)
print("eval_data shape: ", X.shape, y.shape)

net = gluon.nn.Sequential()
#net = gluon.nn.HybridSequential() # doesn't work since LSTM is not hybrid
with net.name_scope():
    net.add(rnn.LSTM(num_hidden)) # note, check also: LSTMCell
    #net.add(nn.Dense(3))  # do not add this, worse accuracy
    net.add(nn.Dense(1)) # output dimension is 1, "in_units" is skipped and infered

net.collect_params().initialize(mx.init.Normal(sigma=0.1))
#softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005, 'momentum':0.9})
#metric = mx.metric.MSE()

epochs = 20
loss_sequence = []
for e in range(epochs):
    cumulative_loss = 0
    for i, (data,label) in enumerate(train_iter):
        #data = data.as_in_context(model_ctx,)
        #label = label.as_in_context(model_ctx,)
        with autograd.record():
            # because LSTM requires TNC data, C: feature size is missing
            #data = data.reshape((0,0,1)) 
            output = net(data)
            #loss = softmax_cross_entropy(output, label)
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

#params = net.collect_params()
#params.save('params.txt')

def model_predict(net, data_iter):
    y = []
    for data,label in data_iter:
        data = data.reshape((0,0,1))
        output = net(data)
        y.append(output.asnumpy()[-1][-1])
    return y

predicted = model_predict(net, eval_iter)

X = x[m+sequence_length:]
realy = [v for v in _label[m:]]
plt.plot(X, realy, label='valy', marker='1', c='r')
plt.plot(X, predicted, label='predicted', marker='+')
plt.show()
