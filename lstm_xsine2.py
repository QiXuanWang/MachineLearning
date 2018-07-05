# ref: https://github.com/sunsided/tensorflow-lstm-sin/blob/master/tf-recurrent-sin-5.1.py
# Yu: This is a modified version to train (sinx + logx), only when use 2 layer lstm could this model simulate it well. And the selection of val data is quite important
# Yu: version 2, try to use dynamic_rnn.py same trick
#     -- The benefit of dynamic_rnn is: 1. it supports dynamic length sequence; 2. it used tf built in functions to reduce code (and complexity?)
#     -- In this version, we still use 1 batch and predict the whole period
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# network parameters
num_inputs = 1  # input is sinx(x), a scalar
num_outputs = 1 # output is a series of sin(x), the future length?
num_layers = 2 # number of stacked LSTM layers
sample_step_in_degrees = 6 # sine wave sampled each 6 degrees
num_steps = int(360/sample_step_in_degrees) # number of time steps for the rnn, also, it's sequence_lengths
num_hidden = 10    # use 5 cells/units in hidden layer
num_epochs = 1000  # 1000 iterations
learning_rate = 0.005 # learning rate
training_iter_step_down_every = 250000
batch_size = 1     # only one sine wave per batch; Yu: what if we use 2? we can't since we only have num_steps data in one sequence. The purpose of batch is to make similar data separated into batches, but if we have several sine waves in sequence, then we may change batch_size to 2. Can we say that it's the major difference between word rnn and these periodic waves. What if the wave is x*sin(x)?
display_step = 100

lr = tf.placeholder(tf.float32, []) # learning_rate
input_data = tf.placeholder(shape=[batch_size, num_steps, num_inputs],dtype=tf.float32)
result_data = tf.placeholder(shape=[batch_size, num_steps, num_outputs],dtype=tf.float32)

weights =  { 
        'out' : tf.Variable(tf.random_normal([batch_size, num_hidden, num_outputs], stddev=0.01, dtype=tf.float32))
}
biases = {
        'out' : tf.Variable(tf.random_normal([batch_size, num_steps, num_outputs], stddev=0.01, dtype=tf.float32))
}


# generate two full sine cycle for one epoch, but use only one
def gen_data(distort=False, epoch=1):
    # generate sin(x)
    sinx = np.arange(0,360,sample_step_in_degrees) # 60 points every sequence
    sinx = np.sin(2*np.pi * sinx/360.0)/2  # sine wave between -0.5 and +0.5
    if distort:
        sinx = np.add(sinx, np.random.uniform(-0.1,0.1,size=num_steps))

    sinx2 = np.stack([sinx, sinx]).reshape(60*2)
    # add logx
    logx2 = 0.1*np.log(np.linspace(epoch, epoch+60*2,60*2))
    if epoch == 0:
        logx2[0] = 1e-8
    X2 = sinx2 + logx2
    X = X2[0:60]
    X = X.reshape(num_steps,1) # num_steps  == 60

    #This actually shift X for 1 timestep, but X[0] needs to multiply next logx
    #y = np.concatenate((X[1:num_steps,:],X[0:1,:]))
    y = X2[1:61]

    X = X.reshape(batch_size,num_steps,1)
    y = y.reshape(batch_size,num_steps,1)

    #a, = plt.plot(X.reshape(60,1), label="X", marker='+',c='b')
    #b, = plt.plot(y.reshape(60,1), label="Y", marker='*',c='r')
    #c, = plt.plot(logx2, label="LOG", marker='^',c='g')
    #d, = plt.plot(logx2+sinx2, label="LOG+SIN", marker='o', c='y')
    #plt.show([a,b,c,d])

    return (X,y)


# dynamicRNN model function
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True)
lstm_cells = [lstm_cell]*num_layers
stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)

lstm_cell2 = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)

# inputs:  (batch_size, n_steps, n_inputs)
# BasicLSTMCell and LSTMCell is different?
# outputs: (batch_size, n_steps, n_hidden)
outputs,states = tf.nn.dynamic_rnn(lstm_cell2, inputs=input_data, dtype=tf.float32) 
#h = tf.transpose(outputs, [1, 0 , 2]) # (n_step,batch_size,n_hidden)
#return tf.matmul(outputs, weights) + biases
# outputs: (batch_size, n_step, n_hidden) 
# weights: (batch_size, n_hidden, 1)
# biases: (batch_size, n_step, 1)
#pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])
pred = tf.matmul(outputs, weights['out']) + biases['out']
individual_losses = tf.reduce_sum(tf.squared_difference(pred, result_data), reduction_indices=1)
cost = tf.reduce_mean(individual_losses)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# validation data
valX,valy = gen_data(False, 900)  # use 1001 as validate produce best test result, use 10 or 100 will produce a shift, and the bigger the number, the better the correlation, why?
testX, testy = gen_data(False, 2200)  # test data doesn't matter


# start training
with tf.Session() as sess:

    #print("gen data")
    #tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter("/Users/joost/tensorflow/log",flush_secs=10,graph_def=sess.graph_def)

    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()

    for k in range(1, num_epochs):

        current_learning_rate = learning_rate
        #current_learning_rate *= 0.1 ** ((step * batch_size) // training_iter_step_down_every)

        batch_x, batch_y = gen_data(False, k)
        # batch_x: (batch_size, n_steps, n_inputs)

        # print((tempX,y))
        print((batch_x.shape, batch_y.shape))

        traindict = {input_data: batch_x, result_data: batch_y, lr: current_learning_rate}

        sess.run(train_op, feed_dict=traindict)

        valdict = {input_data: valX, result_data: valy} # use fixed validation data

        costval,outputval = sess.run((cost,pred), feed_dict=valdict)

        if k == num_epochs-1:
            print("output shape: {}".format(outputval.shape))
            o = np.transpose(outputval,(2,0,1))
            print ("o={}".format(o))
            print("end k={}, cost={}, output={}, y={}".format(k,costval,o,valy))
            print("diff={}".format(np.subtract(o,valy)))
            predicted = [v[0] for v in o[0]]
            plot_predicted, = plt.plot(predicted, label="predicted", marker='+',c='b')
            realy = [v[0] for v in valy[0]]
            plot_valy, = plt.plot(realy, label='valy', marker='1', c='r')
            realX = [v[0] for v in valX[0]]
            plot_valx, = plt.plot(realX, label='valx', marker='*', c='y')
            plt.show()
        else:
            print("Iter k={}, cost={}".format(k,costval))

