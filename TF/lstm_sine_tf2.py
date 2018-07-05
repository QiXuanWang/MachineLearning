import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from lstm2 import lstm_model
from data_processing import generate_data

# constants
FEATURE_NAME = 'point' # name of the input feature
LOG_DIR = './ops_logs/sin'
TIMESTEPS = 3 # define the T length
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01


X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)


#X['train']: (8097,3,1)
#y['train']: (8097,1)
#import pdb
#pdb.set_trace()
#print(X['train'].shape)
#print(y['train'].shape)

model_params = {"learning_rate": LEARNING_RATE}
regressor = tf.estimator.Estimator(model_fn=lstm_model, params=model_params)

# this is a function object, when called, return features dict and labels Tensor
train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={FEATURE_NAME : X['train']},
        y=y['train'],
        batch_size=1,
        num_epochs=None,
        shuffle=False)

regressor.train(input_fn=train_input_fn, steps=200)
#import pdb
#pdb.set_trace()

print("DEBUG: train end")
print("="*80)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={FEATURE_NAME : X['test']},
        y=y['test'],
        num_epochs=1,
        shuffle=False)

print("DEBUG: predict begin")
predicted = regressor.predict(input_fn=test_input_fn)

print("DEBUG: predict end")
print("="*80)
print(type(predicted))

scores = regressor.evaluate(input_fn=test_input_fn)
print(type(scores))
print('Accuracy(tf): {0:f}'.format(scores['accuracy']))


for i in enumerate(predicted):
    print("Prediction: %s:%s"%str(i))


#rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
#score = mean_squared_error(predicted, y['test'])
#print ("MSE: %f" % score)

#plot_predicted, = plt.plot(predicted, label='predicted')
#plot_test, = plt.plot(y['test'], label='test')
#plt.legend(handles=[plot_predicted, plot_test])
#plt.show()
