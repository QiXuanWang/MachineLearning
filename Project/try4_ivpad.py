import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
import os
import time
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import mxnet.ndarray as F
from data import gen_data, save_plot, normalize, denormalize
from Model import SELU,Tanh

ctx = mx.gpu(1)

_data, _label = gen_data(filename='1cycle_iv_small_100p.txt',input_list=['v(i)'], output_list=['i(vpad)'])
num_samples, sequence_length, num_channel  = _data.shape
print("Sequence_len: ", sequence_length)
num_hidden = sequence_length # too big will hang
num_hidden = 256
num_layers = 1    # multi-layer
batch_size = 4

#data_scale = 1.0*np.max(_data)
#label_scale = 1.0*np.max(_label)

_data = np.concatenate([_data]*500)
_label = np.concatenate([_label]*500)
indexes = np.arange(_data[0].size)
data_scale = normalize(_data, axis=1)
label_scale = normalize(_label, axis=1)
plt.plot(indexes, _label[0,:,0], label='normalized')
plt.plot(indexes, _label[0,:,0]*label_scale[0][0], label='recoverd')
plt.legend(loc='upper right')
plt.show()

print("data: ", _data.shape)
print("scale: ",data_scale.shape)

data_len = len(_data)
m = int(0.9*data_len) 
m += m%batch_size
idx = np.random.choice(data_len, size=data_len, replace=False)
train_idx = idx[:m]
test_idx = idx[m:]

X = _data[train_idx, :]
y = _label[train_idx, :]
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True, last_batch='discard')
print("train_data shape: ", X.shape, y.shape)

X = _data[test_idx, :]
y = _label[test_idx, :]
eval_data_scale = data_scale[test_idx, :]
eval_label_scale = label_scale[test_idx, :]
eval_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=False, last_batch='discard')
print("eval_data shape: ", X.shape, y.shape)
print("eval_scale shape: ", eval_data_scale.shape, eval_label_scale.shape)


# A new layer: input: (n, c, w), out: (n, c, w)
class Attn(gluon.Block):
    def __init__(self, batch_size, input_channel, input_size, output_channel, output_size):
        super(Attn, self).__init__()
        #self.c = 0.9 # should be learned, but, how
        self.c = self.params.get('lambda', shape=1)
        self.w1 = self.params.get('weight1', shape=(batch_size, output_channel, input_channel)) # (D', D)
        self.w = self.params.get('weight', shape=(batch_size, input_size, input_size)) # (T, T)
        self.w2 = self.params.get('weight2', shape=(batch_size,input_size, output_size)) # (T, T')
        self.b = self.params.get('bias', shape=(batch_size, output_channel, output_size)) # (D', T')

    def forward(self, x):
        #x: 'nwc'
        #import pdb
        #pdb.set_trace()
        x = F.transpose(x, axes=(0,2,1)) # (nwc) -> (ncw)
        X_ = F.batch_dot(self.w1.data(ctx), x) # (n,c,w) -> (n,c,w)
        # E =  dot(X_, W)
        E = F.batch_dot(X_, self.w.data(ctx))  # (n,c,w) -> (n,c,w)
        attn_weights = F.softmax(E, axis=2) # (n, c, w)
        attn_applied = F.elemwise_mul(attn_weights, X_) #(n,c,w)
        output = self.c.data(ctx)*(attn_applied) + (1-self.c.data(ctx))*X_ # (n,c,w)
        output = F.batch_dot(output, self.w2.data(ctx)) + self.b.data(ctx) # (n, c,w)
        output = F.transpose(output, axes=(0,2,1)) # (ncw) -> (nwc)
        return output

class AttRNN(gluon.Block):
    def __init__(self, input_size, output_size, num_hidden, num_layers):
        super(AttRNN, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add(
                rnn.LSTM(num_hidden, num_layers, layout='NTC'), # 'nwc'
                #nn.BatchNorm(),
                SELU(),
                #Attn(batch_size, num_hidden, input_size, num_hidden, output_size),
                #nn.LeakyReLU(0.1),
                nn.Dense(output_size),
                Tanh()
                )
        self.dense = nn.Sequential()
        self.dense.add(nn.Dense(output_size))

    def forward(self, x):
        output = self.layer(x) # this actually call nn.Block.forward(x) of self.layer, since nn.Sequential inherits nn.Block too
        output = output + self.dense(x)
        return output 

input_size = sequence_length
output_size= sequence_length
net = AttRNN(input_size, output_size, num_hidden, num_layers)
net.collect_params().initialize(mx.init.Normal(sigma=0.05), ctx=ctx)
print(net.collect_params)
#square_loss = gluon.loss.L2Loss()
square_loss = gluon.loss.L1Loss()
learning_settings = {'learning_rate': 0.001, 'momentum':0.9}
trainer = gluon.Trainer(net.collect_params(), 'sgd', learning_settings)

epochs = 100
loss_sequence = []
for e in range(epochs):
    start = time.time()
    cumulative_loss = 0
    if e > 10:
        trainer.set_learning_rate(0.05)
    elif e > 20:
        trainer.set_learning_rate(0.0005)
    elif e > 80:
        trainer.set_learning_rate(0.0002)
    for i, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            if data.shape[0] < batch_size:
                continue
            output = net(data) # out: (batch_size, sequence_length)
            output = output.reshape((batch_size, sequence_length, 1))
            loss = square_loss(output, label) # channle 0 contains data
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

params = net.collect_params()
params.save('try4_ivpad.params')

def model_predict(net, data_iter, batch_size, num_channel=1):
    i = 0
    X = []
    y = []
    p = []
    cumulative_loss = 0
    for i, (data, label) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        if data.shape[0] < batch_size:
            continue
        output = net(data) # (n,w)
        output = output.reshape((batch_size, sequence_length, 1))
        loss = square_loss(output, label)
        cumulative_loss += nd.mean(loss).asscalar()
        print("iter %d, loss: %e"%(i, nd.mean(loss).asscalar()))
        denormalize(data, eval_data_scale[i*batch_size:(i+1)*batch_size, :])
        denormalize(label, eval_label_scale[i*batch_size:(i+1)*batch_size, :])
        denormalize(output, eval_label_scale[i*batch_size:(i+1)*batch_size, :])
        X.append(np.squeeze(data[0].asnumpy()))
        y.append(np.squeeze(label[0].asnumpy()))
        p.append(output[0].asnumpy()) 
        if i == 0:
            print(data.shape, label.shape, output.shape)
        i += 1
    print("cumulated_loss: %e," % (cumulative_loss))
    return X,y,p

X,y,predicted = model_predict(net, eval_iter,batch_size)
# X: (n, w, c)

save_plot(X, y, predicted, 'try4_ivpad.out')
