# In general, use GAN to generate sine wave is of little value
# in most cases, we want to predict y given x. GAN seems can only generate (x,y) pair based on random noise.
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import os
import time
from data_processing import DataFuncIter
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

import logging
logging.getLogger().setLevel(logging.DEBUG)

ctx = mx.cpu()

num_hidden = 5    # use 5 cells in hidden layer
batch_size = 1     # only one sine wave per batch
#data_length = 100  #  data len
data_cycle = 20 # 60*20
sequence_length = 1 # seq len
sample_step_in_degrees = 6

#data_shape = (data_length - sequence_length, sequence_length)
x = np.arange(0,360*data_cycle,sample_step_in_degrees) # 60 points per cycle
x = (2*np.pi * x/360.0)/2
_data = []
for x1 in x:
    _data.append([x1, np.sin(x1)])
_data = mx.ndarray.array(_data)
print(_data.shape)
data_len = len(_data)
m = int(0.8*data_len)
#print(_data[:m])
#print(_label[:m])

X = _data[:m]
Y = mx.nd.ones(shape=(len(X), 1))
train_data = mx.io.NDArrayIter(X, Y, batch_size)

X = _data[m:]
Y = mx.nd.ones(shape=(len(X), 1))
eval_data = mx.io.NDArrayIter(X, Y, batch_size)


# netG
netG = nn.Sequential()
with netG.name_scope():
    #netG.add(rnn.LSTM(num_hidden))
    netG.add(nn.Dense(2)) # output dim
#netD
netD = nn.Sequential()
with netD.name_scope():
    #netD.add(rnn.LSTM(num_hidden))
    netD.add(nn.Dense(5, activation='tanh'))
    netD.add(nn.Dense(3, activation='tanh'))
    netD.add(nn.Dense(2)) # output dim or (batch_size, ?)
    # input is (nc) x 64 x 64
    #netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
    #netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 32 x 32
    #netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
    #netD.add(nn.BatchNorm())
    #netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 16 x 16
    #netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
    #netD.add(nn.BatchNorm())
    #netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 8 x 8
    #netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
    #netD.add(nn.BatchNorm())
    #netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 4 x 4
    #netD.add(nn.Conv2D(1, 4, 1, 0, use_bias=False))


real_label = mx.nd.ones((batch_size, 2), ctx=ctx) # somehow missing 2 will fail
fake_label = mx.nd.zeros((batch_size, 2), ctx=ctx)
metric = mx.metric.Accuracy()

loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
netG.initialize(mx.init.Normal(0.02))
netD.initialize(mx.init.Normal(0.02))

trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})


epochs = 10
for epoch in range(epochs):
    tic = time.time()
    train_data.reset()
    for batch in train_data:
        # update D
        data = batch.data[0].as_in_context(ctx)
        #print(data)
        noise = mx.nd.random_normal(shape=(batch_size, 2))

        with autograd.record():
            #import pdb
            #pdb.set_trace()
            real_output = netD(data) # data is (1,2)
            #print(real_output)
            errD_real = loss(real_output, real_label)

            fake = netG(noise)
            fake_output = netD(fake.detach()) # (1,2)
            #print(fake_output)
            errD_fake = loss(fake_output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()

        trainerD.step(batch_size)
        metric.update([real_label], [real_output])
        metric.update([fake_label], [fake_output])

        with autograd.record():
            output = netD(fake)
            errG = loss(output, real_label)
            errG.backward()
        trainerG.step(batch_size)

    name, acc = metric.get()
    metric.reset()
    print('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    print('time: %f' % (time.time() - tic))

    # predict
    noise = mx.nd.random_normal(shape=(100, 2), ctx=ctx)
    fake = netG(noise)
    plt.scatter(noise[:,0].asnumpy(),np.sin(noise[:,1].asnumpy()), c='b', marker='o')
    plt.scatter(fake[:,0].asnumpy(),np.sin(noise[:,1].asnumpy()), c='g', marker='+')
    plt.scatter(fake[:,0].asnumpy(),fake[:,1].asnumpy(), c='r', marker='-')
    plt.show()

def model_predict2(net, train_data):
    for batch in train_data:
        data = batch.data[0].as_in_context(ctx)
        real_output = net(data) # data is (1,2)
        y.append(real_output.asnumpy()[0][0])
    return None, y

#predicted = model_predict(net, eval_iter)
#X = x[m+sequence_length:]
#realy = [v for v in _label[m:]]
#plt.plot(X, realy, label='valy', marker='1', c='r')
#plt.plot(X, predicted, label='predicted', marker='+')
#plt.show()
