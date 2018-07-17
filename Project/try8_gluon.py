# !/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

#Todo: Ensure skip connection implementation is correct
#https://github.com/opringle/multivariate_time_series_forecasting


import pdb
import os,sys
import math
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn,rnn
from mxnet import ndarray as F # for gluon.Block
from time import time
import argparse
import logging
import metrics
import utils
from utils import _get_batch
#from Model import UpSampling1D
from data import gen_data, save_plot, normalized

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Deep neural network for multivariate time series forecasting",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default='../data',
                    help='relative path to input data')
parser.add_argument('--max-records', type=int, default=None,
                    help='total records before data split')
parser.add_argument('--q', type=int, default=24*7,
                    help='number of historical measurements included in each training example')
parser.add_argument('--horizon', type=int, default=3,
                    help='number of measurements ahead to predict')
parser.add_argument('--splits', type=str, default="0.8,0.1",
                    help='fraction of data to use for train & validation. remainder used for test.')
parser.add_argument('--batch-size', type=int, default=8,
                    help='the batch size.')
parser.add_argument('--filter-list', type=str, default="7,7,7",
                    help='filter sizes, or kernel_size')
parser.add_argument('--num-filters', type=str, default="30,30,30",
                    help='number of filters, or hidden state size')
parser.add_argument('--recurrent-state-size', type=int, default=64,
                    help='number of hidden units in each unrolled recurrent cell')
parser.add_argument('--skip-rnn', type=bool, default=False,
                    help='select if skip-rnn feature should be used')
parser.add_argument('--seasonal-period', type=int, default=24,
                    help='time between seasonal measurements')
parser.add_argument('--time-interval', type=int, default=1,
                    help='time between each measurement')
parser.add_argument('--gpus', type=str, default='1',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout rate for network')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='max num of epochs')
parser.add_argument('--save-period', type=int, default=20,
                    help='save checkpoint for every n epochs')
parser.add_argument('--model-prefix', type=str, default='gluon_model',
                    help='prefix for saving model params')
parser.add_argument('--model-file', type=str, default=None,
                    help='load model params and do predict only')

data_scale = 1
label_scale = 1


def build_iters(data_dir, max_records, q, horizon, splits, batch_size):
    """
    Load & generate training examples from multivariate time series data
    :return: data iters & variables required to define network architecture
    """
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['v(pad)'], shape='ncw') 
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vpad)']) 
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vi)']) 
    _data, _label = gen_data(filename='1cycle_iv_small.txt',input_list=['v(i)'], output_list=['v(pad)']) 
    global data_scale, label_scale
    #data_scale = np.max(_data)
    #label_scale = np.max(_label)
    print("Scale: %.2e %.2e"%(data_scale, label_scale))
    _data = _data/data_scale # scale
    _label = _label/label_scale # scale
    _data = np.concatenate([_data]*200)
    _label = np.concatenate([_label]*200)

    _data = np.atleast_3d(_data)
    _label = np.atleast_3d(_label)
    data_len = len(_data)
    print("Shape: ",_data.shape) # (samples, seq_len, features)
    #sys.exit(0)

    m = int(splits[0]*data_len) 
    m += m%batch_size
    k = int(splits[1]*data_len)
    k += k%batch_size

    idx = np.random.choice(data_len, size=data_len, replace=False)
    train_idx = idx[:m]
    val_idx = idx[m:m+k]
    test_idx = idx[m+k:]

    #X = _data[:m]
    #y = _label[:m]
    X = _data[train_idx, :]
    y = _label[train_idx, :]
    train_iter = mx.io.NDArrayIter(data=X,
                                   label=y,
                                   batch_size=batch_size)
    #train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=False)
    print("train_data shape: ", X.shape, y.shape)

    #X = _data[m:m+k]
    #y = _label[m:m+k]
    X = _data[val_idx, :]
    y = _label[val_idx, :]
    val_iter = mx.io.NDArrayIter(data=X,
                                 label=y,
                                 batch_size=batch_size)
    #val_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=False)
    print("val_data shape: ", X.shape, y.shape)

    #X = _data[m+k:]
    #y = _label[m+k:]
    X = _data[test_idx, :]
    y = _label[test_idx, :]
    test_iter = mx.io.NDArrayIter(data=X,
                                  label=y,
                                  batch_size=batch_size)
    #test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=1, shuffle=False)
    print("test_data shape: ", X.shape, y.shape)
    return train_iter, val_iter, test_iter

class Chomp1d(gluon.Block):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] # remove right padding

class TemporalConvNet(gluon.Block):

    def __init__(self, input_feature_shape, q, filter_list, num_filters, dropout, rcells, skiprcells, p, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.c = input_feature_shape[1] # input: n,c,w
        print("Input_feature_shape: ",input_feature_shape) # 16, 168, 2; 168 is q, we can change it

        with self.name_scope():
            self._convNet = nn.Sequential()
            for i,kernel_size in enumerate(filter_list):
                dilation = 2 ** i
                padding = (kernel_size-1)*dilation
                #padding = int((kernel_size-1)*dilation/2)
                nb_filter = num_filters[i]
                self._convNet.add(
                    nn.Conv1D(channels=nb_filter, kernel_size=kernel_size, strides=1, padding=padding, dilation=dilation, layout='NCW', activation='relu'),
                    nn.BatchNorm(axis=2),
                    Chomp1d(padding),
                    nn.Dropout(dropout),

                    nn.Conv1D(channels=nb_filter, kernel_size=kernel_size, strides=1, padding=padding, dilation=dilation, layout='NCW', activation='relu'),
                    nn.BatchNorm(axis=2),
                    Chomp1d(padding),
                    nn.Dropout(dropout),
                )
            #convNet: (n, nb_filter, w)

            # in original, it's checked for input and output dimension to make sure it's same, now since we use implicit input_channel, we enable it always
            #self._downsample = nn.Conv1D(channels=nb_filter, kernel_size=1)
            self._downsample = nn.Conv1D(channels=self.c, kernel_size=1)

            # Dense output: 
            #self._dense = nn.Sequential()
            #self._dense.add(nn.Dense(input_feature_shape[1], flatten=False))

    #def hybrid_forward(self, F, x): # won't work with rnn
    def forward(self, x):
        """
        we separate two feature sequence and feed into two separate net and then stack them for loss 
        """
        #input: (batch, seq_len, features) for 'nwc'
        #input: (batch, features, seq_len) for 'ncw'
        #pdb.set_trace()
        convi = self._convNet(x)  #O: (n, num_filter, w)
        if self._downsample is not None: # differ from original
            convi = self._downsample(convi) 
        out = convi+ x # (n, c, w)
        return F.relu(out)

class TCN(gluon.Block):
    def __init__(self, input_feature_shape, q, filter_list, num_filters, dropout, rcells, skiprcells, p, **kwargs):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_feature_shape, args.q, args.filter_list, args.num_filters, args.dropout, rcells, skiprcells, p)
        self.linear = nn.Dense(input_feature_shape[2], flatten=False) # ncw

    def forward(self, x):
        #pdb.set_trace()
        y1 = self.tcn(x)
        y2 = self.linear(y1)
        return self.linear(y1)

def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        outputs = [net(X) for X in data]
        losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
        acc += nd.sum(losses).copyto(mx.cpu())
        n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

            
def train(train_data, test_data, net, loss, trainer, ctx, num_epochs):
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc, n = 0.0, 0.0, 0
        train_data.reset()
        if epoch > 10:
            trainer.set_learning_rate(0.001)
        #val_iter.reset()
        start = time()
        for batch in train_data:
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in losses])
            n += batch_size
        print("Epoch %d. Cumulative_Loss: %.3f, Loss: %.3f Time %.1f sec" % (epoch, train_loss, train_loss/n, time() - start))

        if epoch % args.save_period == 0 and epoch > 1:
            print("saving parameters for %d"%epoch)
            net.collect_params().save(os.path.join("./models/", args.model_prefix+str(epoch)+".params"))
        elif epoch == args.num_epochs:
            print("saving parameters for %d"%epoch)
            net.collect_params().save(os.path.join("./models/", args.model_prefix+".params"))
 

def load(net, model_file):
    ctx = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    net.collect_params().load(model_file, ctx=ctx)
    return net


def predict(net, data_iter):
    X = []
    y = []
    p = []
    predict_loss,n = 0.0, 0.0
    for batch in data_iter:
        data, label, batch_size = _get_batch(batch, ctx)
        losses = []
        outputs = [net(X) for X in data]
        losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
        predict_loss += sum([l.sum().asscalar() for l in losses])
        n += batch_size
        X.append(data[0][0].asnumpy())
        y.append(label[0][0].asnumpy())
        p.append(outputs[0][0].asnumpy())
    print("Cumulative_Loss: %.3f, Predict_Loss: %.3f " % (predict_loss, predict_loss/n))
    return X,y,p

if __name__ == '__main__':
    # parse args
    args = parser.parse_args()
    args.splits = list(map(float, args.splits.split(',')))
    args.num_filters = list(map(int, args.num_filters.split(',')))
    args.filter_list = list(map(int, args.filter_list.split(',')))
    assert(len(args.num_filters) == len(args.filter_list))

    # Check valid args
    if not max(args.filter_list) <= args.q:
        raise AssertionError("no filter can be larger than q")
    if not args.q >= math.ceil(args.seasonal_period / args.time_interval):
        raise AssertionError("size of skip connections cannot exceed q")

    # Build data iterators
    train_iter, val_iter, test_iter = build_iters(args.data_dir, args.max_records, args.q, args.horizon, args.splits, args.batch_size)
    input_feature_shape = train_iter.provide_data[0][1]

    # Choose cells for recurrent layers: each cell will take the output of the previous cell in the list
    rcells = [rnn.GRU(hidden_size=args.recurrent_state_size, layout='NTC')]
    skiprcells = [rnn.LSTM(hidden_size=args.recurrent_state_size, layout='NTC')]

    # Define net
    p = int(args.seasonal_period/args.time_interval)
    net = TCN(input_feature_shape, args.q, args.filter_list, args.num_filters, args.dropout, rcells, skiprcells, p)
    ctx = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    net.initialize(mx.initializer.Uniform(0.1), ctx=ctx)
    loss = gluon.loss.HuberLoss(rho=0.1)
    #print("Loss weight: %e"%float(1.0/label_scale))
    #loss = gluon.loss.L2Loss(weight=float(1.0/label_scale)) # won't help!
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.005})

    if args.model_file:
        assert(os.path.exists(args.model_file))
        net = load(net, model_file=args.model_file)
    else:
        train(train_iter, val_iter, net, loss, trainer, ctx, args.num_epochs)
        #utils.train(train_iter, val_iter, net, loss, trainer, ctx, args.num_epochs )

    # predict
    x_test, y_test, pred = predict(net, test_iter)
    # restore scale
    x_test = [a * data_scale for a in x_test]
    y_test = [a * label_scale for a in y_test]
    pred = [a * label_scale for a in pred]
    # plot requires nwc but we now have ncw
    save_plot(x_test, y_test, pred, 'try8_gluon.out')
