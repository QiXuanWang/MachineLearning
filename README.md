# MachineLearning
Testing of Estimator:
Ref: http://blog.mdda.net/ai/2017/02/25/estimator-input-fn

new style in tf.estimator.Estimator is recommended since they appear later. 
Someone guesses that tf.contrib.estimator

we need to test whether LSTM model itself is OK, and estimator.Estimator usage is OK

Test LSTM model:
1. DNNLayer is not used. removed.
2. we only have 1 lstm layer, write lstm_cells as fixed
3. use tf.train.AdagradOptimizer() to remove the optimizer parameter works
4. replace train_op = tf.contrib.layers.optimize_loss() with train_op = optimizer.minimize(...) failed
