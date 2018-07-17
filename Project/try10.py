# !/usr/bin/env python
# -*- coding: utf-8 -*-

# densenet
import pdb
import os,sys
import math
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from time import time
import argparse
import logging
import utils
from data import gen_data, save_plot, normalize, denormalize
from Model import Chomp1d, Tanh

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Deep neural network for multivariate time series forecasting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=8,
                    help='the batch size.')
parser.add_argument('--growth-rate', type=int, default=64,
                    help='nb_filters or nb_channels')
parser.add_argument('--splits', type=str, default="0.8,0.1,0.1",
                    help='train/eval/test data percent')
parser.add_argument('--units', type=str, default="3,6,12,8",
                    help='nb of units')
parser.add_argument('--gpus', type=str, default='0',
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
parser.add_argument('--model-prefix', type=str, default='try10',
                    help='prefix for saving model params')
parser.add_argument('--model-file', type=str, default=None,
                    help='load model params and do predict only')



def build_iters(splits, batch_size):
    """
    Load & generate training examples from multivariate time series data
    :return: data iters & variables required to define network architecture
    """
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['v(pad)'], shape='ncw') 
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vpad)']) 
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vi)']) 
    #_data, _label = gen_data(filename='1cycle_iv_small_100p.txt',input_list=['v(i)'], output_list=['v(pad)'], shape='ncw') 
    _data, _label = gen_data(filename='1cycle_iv_small_100p.txt',input_list=['v(i)'], output_list=['i(vi)'], shape='ncw') 
    global num_samples, sequence_length, num_channel
    num_samples, num_channel, sequence_length = _data.shape
    _data = np.concatenate([_data]*200)
    _label = np.concatenate([_label]*200)
    data_scale = normalize(_data, axis=2)
    label_scale = normalize(_label, axis=2)

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
    global eval_data_scale, eval_label_scale
    eval_data_scale = data_scale[test_idx, :]
    eval_label_scale = label_scale[test_idx, :]
    test_iter = mx.io.NDArrayIter(data=X,
                                  label=y,
                                  batch_size=batch_size)
    #test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=1, shuffle=False)
    print("test_data shape: ", X.shape, y.shape)
    return train_iter, val_iter, test_iter

class TemporalConvNet(gluon.HybridBlock):
    """
    This class only provide one layer of convnet
    growth_rate is nb_filter or channels
    """
    def __init__(self, nb_filter, kernel_size=1, dilation=1, dropout=0.):
        super(TemporalConvNet, self).__init__()
        with self.name_scope():
            self._convNet = nn.HybridSequential()
            padding = (kernel_size-1)*dilation
            self._convNet.add(
                nn.Conv1D(channels=nb_filter, kernel_size=kernel_size, strides=1, padding=padding, dilation=dilation, layout='NCW', activation='relu'),
                nn.BatchNorm(axis=2),
                Chomp1d(padding),
                nn.Dropout(dropout),
            )
            #convNet: (n, nb_filter, w)

    def hybrid_forward(self, F, x): 
        """
        """
        convi = self._convNet(x)  # (n, num_filter, w)
        return convi
        #return F.relu(convi)

class TransitionBlock(gluon.HybridBlock):
    """
    It's the temporalnet with pooling?
    """
    def __init__(self, num_stage, nb_filter, kernel_size=1, dilation=1, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self._convNet = nn.HybridSequential()
            padding = (kernel_size-1)*dilation
            self._convNet.add(
                nn.Conv1D(channels=nb_filter, kernel_size=kernel_size, strides=1, padding=padding, dilation=dilation, layout='NCW', activation='relu'),
                nn.BatchNorm(axis=2),
                Chomp1d(padding),
                #nn.Dropout(dropout),
                nn.AvgPool1D(pool_size=2),
            )
    def hybrid_forward(self, F, x): 
        convi = self._convNet(x)  #O: (n, num_filter, w)
        return convi
        #return F.relu(convi)

class DenseBlock(gluon.HybridBlock):
    def __init__(self, units_num, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        with self.name_scope():
            self._convNet = nn.HybridSequential()
            for i in range(units_num):
                block = TemporalConvNet(growth_rate)
                self._convNet.add(block)
            #convNet: (n, nb_filter, w)

    def hybrid_forward(self, F, x):
        #pdb.set_trace()
        for block in self._convNet:
            data = block(x)
            x = F.concat(x, data, dim=1)
        return x


class DenseNet(gluon.HybridBlock):
    def __init__(self, units, num_stage, growth_rate, reduction=0.5, drop_out=0.0, bottle_neck=True, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        assert(len(units) == num_stage)
        init_channels = 2 * growth_rate
        n_channels = init_channels
        with self.name_scope():
            self._net = nn.HybridSequential()
            for i in range(num_stage):
                n_channels += units[i]*growth_rate
                n_channels = int(math.floor(n_channels*reduction))
                print("i: %d, unit: %d, n_channels: %d"%(i, units[i], n_channels))
                self._net.add(DenseBlock(units[i], growth_rate))
                self._net.add(TransitionBlock(i, n_channels))

    def hybrid_forward(self, F, x):
        out = self._net(x)
        return out

    
def get_net(units, num_stage, growth_rate):
    net = nn.HybridSequential()
    net.add(DenseNet(units, num_stage, growth_rate)) # ncw
    net.add(Tanh()) #  put here seems better?
    net.add(nn.Dense(sequence_length)) # nw
    return net


def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = utils._get_batch(batch, ctx)
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
        if epoch > 20:
            trainer.set_learning_rate(0.0002)
        #val_iter.reset()
        start = time()
        for batch in train_data:
            data, label, batch_size = utils._get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                outputs = [D.reshape((batch_size, 1, sequence_length)) for D in outputs]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            #trainer.step(batch_size, ignore_stale_grad=True)
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
    i = 0
    for batch in data_iter:
        data, label, batch_size = utils._get_batch(batch, ctx)
        losses = []
        outputs = [net(X) for X in data]
        losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
        predict_loss += sum([l.sum().asscalar() for l in losses])
        n += batch_size
        #pdb.set_trace()
        data = [denormalize(D, eval_data_scale[i*batch_size:(i+1)*batch_size, :], axis=2) for D in data]
        label = [denormalize(D, eval_label_scale[i*batch_size:(i+1)*batch_size, :], axis=2) for D in label]
        outputs = [denormalize(D, eval_label_scale[i*batch_size:(i+1)*batch_size, :], axis=2) for D in outputs]
        outputs = [D.reshape((batch_size, 1, sequence_length)) for D in outputs]
        X.append(data[0][0].asnumpy())
        y.append(label[0][0].asnumpy())
        p.append(outputs[0][0].asnumpy())
        i += 1
    print("Cumulative_Loss: %.3f, Predict_Loss: %.3f " % (predict_loss, predict_loss/n))
    return X,y,p

if __name__ == '__main__':
    args = parser.parse_args()
    args.splits = list(map(float, args.splits.split(',')))
    args.units = list(map(int, args.units.split(',')))

    # Build data iterators
    train_iter, val_iter, test_iter = build_iters(args.splits, args.batch_size)

    # Define net
    num_stage = len(args.units)
    net = get_net(units=args.units, num_stage=num_stage, growth_rate=args.growth_rate)
    print(net)
    #mx.viz.plot_network(net)

    #net = DenseNet(units=units, num_stage=4, growth_rate=48 if args.depth == 161 else args.growth_rate, num_class=args.num_classes, data_type="msface", reduction=args.reduction, drop_out=args.drop_out, bottle_neck=True, bn_mom=args.bn_mom, workspace=args.workspace)

    ctx = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    params = net.collect_params()
    #print(params)
    params.initialize(mx.initializer.Normal(sigma=0.05), ctx=ctx)

    #loss = gluon.loss.HuberLoss(rho=0.1)
    loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.001})

    if args.model_file:
        assert(os.path.exists(args.model_file))
        net = load(net, model_file=args.model_file)
    else:
        train(train_iter, val_iter, net, loss, trainer, ctx, args.num_epochs)

    # predict
    x_test, y_test, pred = predict(net, test_iter)
    # plot requires nwc but we now have ncw
    save_plot(x_test, y_test, pred, 'try10.out')
