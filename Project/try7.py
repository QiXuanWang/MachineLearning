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
#Yu: This TCN is based on  https://github.com/locuslab/TCN

import os,sys
import math
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon
import argparse
import logging
import metrics
from data import gen_data

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
parser.add_argument('--splits', type=str, default="0.6,0.2",
                    help='fraction of data to use for train & validation. remainder used for test.')
parser.add_argument('--batch-size', type=int, default=16,
                    help='the batch size.')
parser.add_argument('--filter-list', type=str, default="3,6,12",
                    help='unique filter sizes')
parser.add_argument('--num-filters', type=int, default=6,
                    help='number of each filter size')
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
parser.add_argument('--model-prefix', type=str, default='skip_rnn_model',
                    help='prefix for saving model params')
parser.add_argument('--model-file', type=str, default=None,
                    help='load model params and do predict only')


def build_iters(data_dir, max_records, q, horizon, splits, batch_size):
    """
    Load & generate training examples from multivariate time series data
    :return: data iters & variables required to define network architecture
    """
    _data, _label = gen_data(filename='500cycle.txt',input_list=['v(i)', 'i(vi)'], output_list=['v(pad)', 'i(vpad)'], seq_len=q)
    #_data, _label = gen_data(filename='1cycle.txt',input_list=['v(i)'], output_list=['v(pad)']) # doesn't work because of shape or else?
    _data = np.atleast_3d(_data)
    _label = np.atleast_3d(_label)
    data_len = len(_data)
    print("Shape: ",_data.shape)
    #sys.exit(0)

    m = int(splits[0]*data_len) 
    m += m%batch_size
    k = int(splits[1]*data_len)
    k += k%batch_size

    X = _data[:m]
    y = _label[:m]
    train_iter = mx.io.NDArrayIter(data=X,
                                   label=y,
                                   batch_size=batch_size)
    #train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=False)
    print("train_data shape: ", X.shape, y.shape)

    X = _data[m:m+k]
    y = _label[m:m+k]
    val_iter = mx.io.NDArrayIter(data=X,
                                 label=y,
                                 batch_size=batch_size)
    #val_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=False)
    print("val_data shape: ", X.shape, y.shape)

    X = _data[m+k:]
    y = _label[m+k:]
    test_iter = mx.io.NDArrayIter(data=X,
                                  label=y,
                                  batch_size=batch_size)
    #test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=1, shuffle=False)
    print("test_data shape: ", X.shape, y.shape)
    return train_iter, val_iter, test_iter

def _rnn_layer(input_data=None, rcells=None, q=None, dropout=None):
    stacked_rnn_cells = mx.rnn.SequentialRNNCell()
    for i, recurrent_cell in enumerate(rcells): # 100 rcells
        stacked_rnn_cells.add(recurrent_cell)
        stacked_rnn_cells.add(mx.rnn.DropoutCell(dropout))
    #outputs, states = stacked_rnn_cells.unroll(length=q, inputs=cnn_reg_features, merge_outputs=False)
    outputs, states = stacked_rnn_cells.unroll(length=q, inputs=input_data, merge_outputs=False)
    # outputs: [(16, 100)] * 168 , 100 is hidden_unit, should be 2?
    #rnn_features = outputs[-1] #only take value from final unrolled cell for use later
    #original rnn_features: (16, 100)
    rnn_features = mx.sym.stack(*outputs, axis=-1)
    # new rnn_features: (16, 168, 100)
    return rnn_features

def _skip_rnn_layer(input_data=None, skiprcells=None, q=None, p=None, dropout=None):
    # data is cnn_reg_features
    stacked_rnn_cells = mx.rnn.SequentialRNNCell()
    for i, recurrent_cell in enumerate(skiprcells): # 100
        stacked_rnn_cells.add(recurrent_cell)
        stacked_rnn_cells.add(mx.rnn.DropoutCell(dropout))
    #outputs, states = stacked_rnn_cells.unroll(length=q, inputs=cnn_reg_features, merge_outputs=False) 
    outputs, states = stacked_rnn_cells.unroll(length=q, inputs=input_data, merge_outputs=False) 
    # outputs: [(16, 168)]x100

    # Take output from cells p steps apart
    #p = int(seasonal_period / time_interval) # 24
    output_indices = list(range(0, q, p)) # (0, 168, 24),
    outputs.reverse()
    skip_outputs = [outputs[i] for i in output_indices] # [(16,168)]x7
    #skip_rnn_features = mx.sym.concat(*skip_outputs, dim=1) # (16, 168*7)
    skip_rnn_features = mx.sym.stack(*skip_outputs, axis=-1) # (16, 168*7)
    return skip_rnn_features


def sym_gen(train_iter, q, filter_list, num_filter, dropout, rcells, skiprcells, seasonal_period, time_interval):

    #Yu: input_feature_shape: (16, 168, 2)
    input_feature_shape = train_iter.provide_data[0][1]
    X = mx.symbol.Variable(train_iter.provide_data[0].name)
    Y = mx.sym.Variable(train_iter.provide_label[0].name)

    # reshape data before applying convolutional layer (takes 4D shape incase you ever work with images)
    conv_input = mx.sym.reshape(data=X, shape=(0, 1, q, -1))
    # Yu: conv_input: (16, 1, 168, 2)  

    ###############
    # CNN Component
    ###############
    outputs = [] 
    for i, filter_size in enumerate(filter_list): # 3
        # pad input array to ensure number output rows = number input rows after applying kernel
        padi = mx.sym.pad(data=conv_input, mode="constant", constant_value=0,
                          pad_width=(0, 0, 0, 0, filter_size - 1, 0, 0, 0))
        convi = mx.sym.Convolution(data=padi, kernel=(filter_size, input_feature_shape[2]), num_filter=num_filter)
        # convi: (16, 100, 168, 2) # num_filter==100
        acti = mx.sym.Activation(data=convi, act_type='relu')
        trans = mx.sym.reshape(mx.sym.transpose(data=acti, axes=(0, 2, 1, 3)), shape=(0, 0, 0))
        # trans: (16, 168, 100, 2) -> (16, 168, 200)
        outputs.append(trans)
    cnn_features = mx.sym.Concat(*outputs, dim=2)
    # cnn_features: (16, 168, 600)
    cnn_reg_features = mx.sym.Dropout(cnn_features, p=dropout)

    ###############
    # RNN Component
    ###############
    rnn_features = _rnn_layer(input_data=cnn_reg_features, rcells=rcells, q=q, dropout=dropout)

    ####################
    # Skip-RNN Component
    ####################
    skip_rnn_features = None
    if args.skip_rnn:
        p = int(seasonal_period / time_interval) # 24
        skip_rnn_features = _skip_rnn_layer(input_data=cnn_reg_features, skiprcells=skiprcells, q=q, p=p, dropout=dropout)

    ##########################
    # Autoregressive Component
    ##########################
    auto_list = []
    for i in list(range(input_feature_shape[2])):
        time_series = mx.sym.slice_axis(data=X, axis=2, begin=i, end=i+1)
        #fc_ts = mx.sym.FullyConnected(data=time_series, num_hidden=1)
        # original: fc_ts: (16, 1)
        fc_ts = mx.sym.FullyConnected(data=time_series, num_hidden=input_feature_shape[1]) # Yu: new fc_ts: (batch, num_hidden) = (16, 168)
        auto_list.append(fc_ts)
    ar_output = mx.sym.stack(*auto_list, axis=-1) # (16, 168, 2)
    #ar_output = mx.sym.concat(*auto_list, dim=1) 
    # original ar_output: (16, 2)

    ######################
    # Prediction Component
    ######################
    # Yu: input should be (16, 168) so we could get (16, 168, 2)
    # currently we have (16, 2) because rnn_features (16, 100), skip_rnn_features: (16, 100), concat to (16, 200)
    neural_components = None
    if rnn_features is None and skip_rnn_features is None:
        print("Error, no rnn features")
        sys.exit(-1)
    elif rnn_features is None:
        neural_components = skip_rnn_features
    elif skip_rnn_features is None:
        neural_components = rnn_features
    else:
        neural_components = mx.sym.concat(*[rnn_features, skip_rnn_features], dim=-1) # (16, 168, 107)

    neural_output = []
    for i in range(input_feature_shape[1]):
        neural_output.append(mx.sym.FullyConnected(data=neural_components, num_hidden=input_feature_shape[2])) #Yu: (16,2) each
    neural_output = mx.sym.stack(*neural_output,axis=1) # generate (16, 168,2)
    model_output = neural_output + ar_output
    loss_grad = mx.sym.LinearRegressionOutput(data=model_output, label=Y)
    return loss_grad, [v.name for v in train_iter.provide_data], [v.name for v in train_iter.provide_label]

def train(symbol, train_iter, valid_iter, data_names, label_names):
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.Module(symbol, data_names=data_names, label_names=label_names, context=devs)
    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(mx.initializer.Uniform(0.1))
    module.init_optimizer(optimizer=args.optimizer, optimizer_params={'learning_rate': args.lr})

    last_metric = None
    print("Begin training ...")
    for epoch in range(1, args.num_epochs+1):
        train_iter.reset()
        val_iter.reset()
        for batch in train_iter:
            module.forward(batch, is_train=True)  # compute predictions
            module.backward()  # compute gradients
            module.update() # update parameters

        train_pred = module.predict(train_iter).asnumpy()
        train_label = train_iter.label[0][1].asnumpy()
        current_metric = metrics.evaluate(train_pred, train_label)
        #if last_metric is None:
        #    pass
        #elif (current_metric['RAE'] - last_metric['RAE']) < 1e-5:
        #    break
        #    last_metric = current_metric
        print('\nMetrics: Epoch %d, Training %s' % (epoch, current_metric))

        val_pred = module.predict(val_iter).asnumpy()
        val_label = val_iter.label[0][1].asnumpy()
        print('Metrics: Epoch %d, Validation %s' % (epoch, current_metric))

        if epoch % args.save_period == 0 and epoch > 1:
            module.save_checkpoint(prefix=os.path.join("./models/", args.model_prefix), epoch=epoch, save_optimizer_states=False)
        elif epoch == args.num_epochs:
            module.save_checkpoint(prefix=os.path.join("./models/", args.model_prefix), epoch=epoch, save_optimizer_states=False)
    return module
 

def load(symbol, train_iter, data_names, label_names, model_file):
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.Module(symbol, data_names=data_names, label_names=label_names, context=devs)
    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.load_params(model_file)
    return module
    #module.init_optimizer(optimizer=args.optimizer, optimizer_params={'learning_rate': args.lr})


def predict(module, test_iter, data_name, label_names):
    test_pred = module.predict(test_iter).asnumpy()
    test_label = test_iter.label[0][1].asnumpy()
    # to save for ploting, we need 
    save(test_iter.data[0][1].asnumpy(), test_label, test_pred, 'try8.out')

def save(X, y, predicted, filename=None):
    """
    all should be of same shape and dim=2
    """
    d={'x':X, 'y': y, 'p':predicted}
    df = pd.DataFrame().append(d, ignore_index=True)
    df.to_pickle(filename)


if __name__ == '__main__':
    # parse args
    args = parser.parse_args()
    args.splits = list(map(float, args.splits.split(',')))
    args.filter_list = list(map(int, args.filter_list.split(',')))

    # Check valid args
    if not max(args.filter_list) <= args.q:
        raise AssertionError("no filter can be larger than q")
    if not args.q >= math.ceil(args.seasonal_period / args.time_interval):
        raise AssertionError("size of skip connections cannot exceed q")

    # Build data iterators
    train_iter, val_iter, test_iter = build_iters(args.data_dir, args.max_records, args.q, args.horizon, args.splits, args.batch_size)

    # Choose cells for recurrent layers: each cell will take the output of the previous cell in the list
    rcells = [mx.rnn.GRUCell(num_hidden=args.recurrent_state_size)]
    skiprcells = [mx.rnn.LSTMCell(num_hidden=args.recurrent_state_size)]

    # Define network symbol
    symbol, data_names, label_names = sym_gen(train_iter, args.q, args.filter_list, args.num_filters, args.dropout, rcells, skiprcells, args.seasonal_period, args.time_interval)


    module = None
    # load model
    if args.model_file:
        assert(os.path.exists(args.model_file))
        module = load(symbol, train_iter, data_names, label_names, model_file=args.model_file)
    else:
        # train cnn model
        module = train(symbol, train_iter, val_iter, data_names, label_names)

    # predict
    predict(module, test_iter, data_names, label_names)


