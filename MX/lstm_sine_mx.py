# Yu: This file somehow implemented a symbolic algorithm
# This one is much faster than lstm_sine_gluon and much accurate
# Don't know why yet
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import os
from data_processing import DataFuncIter

import logging
logging.getLogger().setLevel(logging.DEBUG)

num_hidden = 5    # use 5 cells in hidden layer
batch_size = 1     # only one sine wave per batch
data_cycle = 20  #  data len = 10 * 60
sequence_length = 6 # seq len
sample_step_in_degrees = 6

#data_shape = (data_length - sequence_length, sequence_length)
x = np.arange(0,360*data_cycle,sample_step_in_degrees) # 60 points per cycle
x = (2*np.pi * x/360.0)/2
#plt.plot(x, np.sin(x), label='x')
#plt.show()
#x = np.linspace(1, data_length, data_length) # not a smooth curve
# if use np.square, the number will get too big and no meaningful things are trained
# so, to limit the value range is essential, this is why we always to normalize it into some -1 ~ 1 range
myIter = DataFuncIter(np.sin, x, sequence_length, use_last=True)
myIter.describe()
_data, _label = myIter.gen_data()
data_len = len(_data)
m = int(0.8*data_len)
#print(_data[:m])
#print(_label[:m])

train_iter = mx.io.NDArrayIter(
        data = _data[:m],
        label = _label[:m], 
        batch_size = batch_size, 
        shuffle=False,
        label_name='lro_label')

eval_iter = mx.io.NDArrayIter(
        data = _data[m:], 
        label = _label[m:], 
        batch_size = batch_size, 
        shuffle=False)

# Symbolic network built
data_input = mx.symbol.Variable('data')
label_input = mx.symbol.Variable('lro_label')

# This is experimental cell symbol 
lstm_cell = mx.rnn.LSTMCell(num_hidden=num_hidden) 

# we don't need embedding for our usage, embedding is only for word processing
# Embed a sequence. 'seq_data' has the shape of (batch_size, sequence_length).
#embedded_seq = mx.symbol.Embedding(data=seq_input, \
        #input_dim=input_dim, \
        #output_dim=embed_dim)
# Note that when unrolling, if 'merge_outputs' is set to True, the 'outputs' is merged into a single symbol
# In the layout, 'N' represents batch size, 'T' represents sequence length, and 'C' represents the number of dimensions in hidden states.
outputs, states = lstm_cell.unroll(length=sequence_length, \
        inputs=data_input, \
        layout='NTC', \
        merge_outputs=True)
# 'outputs' is concat0_output of shape (batch_size, sequence_length, hidden_dim).
# The hidden state and cell state from the final time step is returned:
# Both 'lstm_t4_out_output' and 'lstm_t4_state_output' have shape (batch_size, hidden_dim).

fc_layer = mx.sym.FullyConnected(data=outputs, name='fc1', num_hidden = 1)
#fc_layer = mx.sym.reshape(fc_layer, (batch_size, -1))
# fc_layer output: (batch, num_hidden)
lro = mx.sym.LinearRegressionOutput(data=fc_layer, label=label_input, name="output") 

model = mx.mod.Module(
    symbol = lro , # this is the symbol variable, while not name
    data_names=['data'], # name in symbol
    label_names = ['lro_label'] # name in symbol
)  # network structure

#viz = mx.viz.plot_network(symbol=lro)
#viz.view()


model.fit(train_iter,
          eval_data=eval_iter,
          optimizer='sgd',
          optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
          num_epoch=20, # the more the better
          eval_metric='mse',
          batch_end_callback=mx.callback.Speedometer(batch_size, 10))
#model.save_params('good.params')


y = model.predict(eval_iter).asnumpy()
metric = mx.metric.MSE()
model.score(eval_iter, metric)

X = x[m+sequence_length:]
# +sequence_length because predicted values have 3 less values? why?
realy = [v for v in _label[m:]]
plt.plot(X, realy, label='valy', marker='1', c='r')
plt.plot(X, y, label='predicted', marker='+')
plt.show()
