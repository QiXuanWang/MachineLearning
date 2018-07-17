# !/usr/bin/env python


import pdb
import os,sys
import math
import numpy as np
#import pandas as pd
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn,rnn
from time import time
import argparse
import logging
from utils import _get_batch
from Model import TemporalConvNet, Tanh
from data import gen_data, save_plot, normalize, denormalize

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Deep neural network for multivariate time series forecasting",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default='../data',
                    help='relative path to input data')
parser.add_argument('--splits', type=str, default="0.8,0.1",
                    help='fraction of data to use for train & validation. remainder used for test.')
parser.add_argument('--batch-size', type=int, default=8,
                    help='the batch size.')
parser.add_argument('--filter-list', type=str, default="7,5,5,7",
                    help='list of filter sizes, or kernel_size')
parser.add_argument('--num-filters', type=str, default="128,64,64,128",
                    help='number of filters, or hidden state size')
parser.add_argument('--gpus', type=str, default='1',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate for network')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='max num of epochs')
parser.add_argument('--save-period', type=int, default=20,
                    help='save checkpoint for every n epochs')
parser.add_argument('--model-prefix', type=str, default='try8_gluon2',
                    help='prefix for saving model params')
parser.add_argument('--model-file', type=str, default=None,
                    help='load model params and do predict only')


def build_iters(filename, input_list, output_list, splits, batch_size):
    """
    Load & generate training examples from multivariate time series data
    :return: data iters & variables required to define network architecture
    """
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['v(pad)'], shape='ncw') 
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vpad)']) 
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vi)']) 
    #_data, _label = gen_data(filename='1cycle_iv_small_100p.txt',input_list=['v(i)'], output_list=['v(pad)'], shape='ncw') 
    _data, _label = gen_data(filename='1cycle_iv_small_100p.txt',input_list=['v(i)'], output_list=['i(vpad)'], shape='ncw') 
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


class TCN(gluon.HybridBlock):
    def __init__(self, input_shape, filter_list, num_filters, dropout, **kwargs):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_shape, args.filter_list, args.num_filters, args.dropout, Attention=True)

    def hybrid_forward(self, F, x):
        out = self.tcn(x)
        return out

def get_net(input_shape, filter_list, num_filters, dropout):
    net = nn.HybridSequential()
    net.add(TCN(input_shape, filter_list, num_filters, dropout))
    net.add(Tanh())
    net.add(nn.Dense(sequence_length))
    return net

            
def train(train_data, test_data, net, loss, trainer, ctx, num_epochs):
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc, n = 0.0, 0.0, 0
        train_data.reset()
        if epoch > 10:
            trainer.set_learning_rate(0.001)
        elif epoch > 30:
            trainer.set_learning_rate(0.0001)
        #val_iter.reset()
        start = time()
        for batch in train_data:
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                outputs = [D.reshape((batch_size, 1, sequence_length)) for D in outputs]
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
    i=0
    for batch in data_iter:
        data, label, batch_size = _get_batch(batch, ctx)
        losses = []
        outputs = [net(X) for X in data]
        losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
        predict_loss += sum([l.sum().asscalar() for l in losses])
        n += batch_size
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
    # parse args
    args = parser.parse_args()
    args.splits = list(map(float, args.splits.split(',')))
    args.num_filters = list(map(int, args.num_filters.split(',')))
    args.filter_list = list(map(int, args.filter_list.split(',')))
    assert(len(args.num_filters) == len(args.filter_list))

    # Build data iterators
    filename = '1cycle_iv_small_100p.txt'
    input_list=['v(i)']
    output_list=['v(pad)']
    train_iter, val_iter, test_iter = build_iters(filename, input_list, output_list, args.splits, args.batch_size)
    input_shape = train_iter.provide_data[0][1]

    # Define net
    net = get_net(input_shape, args.filter_list, args.num_filters, args.dropout)
    print(net)
    ctx = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    params = net.collect_params()
    #print(params)
    params.initialize(mx.initializer.Uniform(0.01), ctx=ctx)
    loss = gluon.loss.HuberLoss(rho=0.1)
    #trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':0.005})
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.005})

    if args.model_file:
        assert(os.path.exists(args.model_file))
        net = load(net, model_file=args.model_file)
    else:
        train(train_iter, val_iter, net, loss, trainer, ctx, args.num_epochs)

    # predict
    x_test, y_test, pred = predict(net, test_iter)
    save_plot(x_test, y_test, pred, 'try8_gluon2.out')
