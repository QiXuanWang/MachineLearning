import mxnet as mx
import numpy as np

import logging
logging.getLogger().setLevel(logging.DEBUG)

#Training data
train_data = np.random.uniform(0, 1, [100, 2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])
batch_size = 1

#Evaluation Data / Validation Data
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11,26,16])

# Iterator required for Module fit process
train_iter = mx.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name='lin_reg_label')  # this name should be used in Module
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

# Symbolic network built
X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label') # we use name to map real data
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro") 

model = mx.mod.Module(
    symbol = lro , # this is the symbol variable, while not name
    data_names=['data'], # symbol name
    label_names = ['lin_reg_label'] # symbol name
)  # network structure

#viz = mx.viz.plot_network(symbol=lro)
#viz.view()

import pdb
pdb.set_trace()
model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
model.fit(train_iter, eval_iter, # train data and validation data
            optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
            num_epoch=20,
            eval_metric='mse',
            batch_end_callback = mx.callback.Speedometer(batch_size, 2))

model.predict(eval_iter).asnumpy()

metric = mx.metric.MSE()
model.score(eval_iter, metric)

eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11.1,26.1,16.1]) #Adding 0.1 to each of the values
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)
model.score(eval_iter, metric)

