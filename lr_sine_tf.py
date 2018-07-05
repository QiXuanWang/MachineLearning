# not working
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from data_processing import generate_data

tf.logging.set_verbosity(tf.logging.INFO)

# constants
FEATURES = ['TIME'] # name of the input feature
LABEL = "VALUE"
LOG_DIR = './ops_logs/lr'
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100
LEARNING_RATE = 0.001

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame({k: data_set[k] for k in FEATURES}),
            y=pd.Series(data_set[LABEL]),
            num_epochs=num_epochs,
            shuffle=shuffle)

# main
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
#regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
        #hidden_units=[32, 32],
        #model_dir=LOG_DIR)
regressor = tf.estimator.LinearRegressor(feature_columns=feature_cols)

# Train
X = np.linspace(0, 100, 10000, dtype=np.float32)
y = np.sin(X)
training_set = {"TIME":X, "VALUE":y}
regressor.train(input_fn=get_input_fn(training_set), steps=TRAINING_STEPS)

# use  1 epoch to evaluate
X = np.linspace(90, 100, 1000, dtype=np.float32)
y = np.sin(X)
test_set = {"TIME":X, "VALUE":y}
ev = regressor.evaluate(
        input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
loss_score=ev['loss']
print("Loss: {0:f}".format(loss_score))

# predict
X = np.linspace(180, 190, 100, dtype=np.float32)
y = np.sin(X)
predict_set = {"TIME":X, "VALUE":y}
predicted = regressor.predict(input_fn=get_input_fn(predict_set, num_epochs=1, shuffle=False))

predictions = list(p["predictions"] for p in predicted)
#print("Predictions: {}".format(str(predictions)))

rmse = np.sqrt(((predictions-y)**2).mean(axis=0))
score = mean_squared_error(predictions, y)
print("MSE: %f"%score)

plot_predicted, = plt.plot(predictions, label='predicted')
plot_test, = plt.plot(y, label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
