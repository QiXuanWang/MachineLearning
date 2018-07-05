import tensorflow as tf
from tensorflow.python.framework import dtypes
# for MNIST, input is (N, 28, 28) shape
def lstm_model(features, labels, mode, params):
    # connect first hidden layer to input layer
    # features["point"] is (N, STEPS, 1) shape
    # labels y is (N, 1) shape
    # N is points number, STEPS==3, (as for 28x28 image, step is 28)

    # input layer, not userd, (N, steps, 10) shape
    #first_hidden_layer = tf.layers.dense(X, 10, activation=tf.nn.relu)

    # lstm cells, will generate a (?, num_units) matrix
    lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=5)]
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)

    # reshape X, why?
    # x_ is [(N,1), (N,1), (N,1)]
    #import pdb
    #pdb.set_trace()
    x_ = None
    if type(features) is type({}):
        # new style
        x_ = tf.unstack(features['point'], axis=1)
    else:
        x_ = tf.unstack(features, axis=1)
    assert(x_ is not None)
    # For static_rnn, inputs x_ is a length T list of Tensor shape [batch_size, input_size], so here T==3, batch_size == N, input_size==1??
    # output is a length T of outputs (one for each input), or a nested tuple of such
    # so output is [(N,1), (N,1), (N,1)]
    output, state = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
    # dnn layers, not used 
    #output_layer = tf.layers.dense(output, 1)
    #predictions = tf.reshape(output_layer, [-1])
    #loss = tf.losses.mean_squared_error(labels, predictions)
    #output = tf.reshape(output, [-1])

    # output is unfolded
    output = output[-1]
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)

    # method 1 from reference
    predictions, loss = tf.contrib.learn.models.linear_regression(output, labels)
    train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=0.1)
    # method 2, doesn't work
    #predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=tf.nn.relu)
    #loss = tf.losses.mean_squared_error(predictions, labels)
    # This is not a real train_op?
    #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    #print("prediction shape: ", predictions.shape)
    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)}


    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = loss
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train_op
    else:
        train_op = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = predictions
    else:
        predictions = None

    return tf.estimator.EstimatorSpec(
             mode=mode,
             predictions=predictions,
             loss=loss, 
             train_op=train_op,
             eval_metric_ops=eval_metric_ops
             )
