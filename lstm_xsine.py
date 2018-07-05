# ref: https://raw.githubusercontent.com/joostbr/tensorflow_lstm_example/master/lstm_sine_example.py
# Yu: This is a modified version to train (sinx + logx), only when use 2 layer lstm could this model simulate it well. And the selection of val data is quite important, num_hidden impact only accuracy little when it > 5
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

num_inputs = 1  # input dimension
num_outputs = 1 # output dimension

sample_step_in_degrees = 6 # sine wave sampled each 6 degrees
num_steps = int(360/sample_step_in_degrees) # number of time steps for the rnn, also, it's sequence_lengths
num_hidden = 5    # use 5 cells/units in hidden layer
num_epochs = 1000  # 1000 iterations
batch_size = 1     # only one sine wave per batch; Yu: what if we use 2? we can't since we only have num_steps data in one sequence. The purpose of batch is to make similar data separated into batches, but if we have several sine waves in sequence, then we may change batch_size to 2. Can we say that it's the major difference between word rnn and these periodic waves. What if the wave is x*sin(x)?

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


inputs = tf.placeholder(shape=[batch_size, num_steps, num_inputs],dtype=tf.float32)
result = tf.placeholder(shape=[batch_size, num_steps, 1],dtype=tf.float32)
w_output = tf.Variable(tf.random_normal([num_steps, num_hidden], stddev=0.01, dtype=tf.float32))
b_output = tf.Variable(tf.random_normal([num_steps, 1], stddev=0.01, dtype=tf.float32))

seqlen = tf.placeholder(tf.int32, [None])
seq_max_len = num_steps # 

def dynamicRNN(x, seqlen, weights, biases):
    # input: (batch_size, n_step, n_input)
    cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_hidden)
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_hidden)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)
    X = tf.transpose(x,[1,0,2]) # num_steps (T) elements of shape (batch_size x num_inputs) 
    X = tf.reshape(X,[-1, num_inputs]) # flatten the data with num_inputs values on each row
    X = tf.split(X, num_steps, axis=0) # create a list with an element per timestep, cause that is what the static_rnn needs as input

    print(X)
    # static_rnn:
    # inputs: [(batch_size, n_input)] * n_step
    # outputs: [(batch_size, self.output_size)] * n_step, output_size is num_hidden, so it's [(batch_size, num_hiddeln)]*n_step
    outputs,states = tf.nn.static_rnn(lstm_cell, inputs=X, dtype=tf.float32) # initial_state=init_state) # outputs & states for each time step

    # the_output: [(batch_size, output_size)]*num_steps
    the_output = []
    for i in range(num_steps):
        print("Outputs[%d]:"%i, outputs[i])
        print(weights[i:i+1,:])
        # matmul(a, b, tranpose_b) --> a*b^T --> (1, n_hidden) * (n_hidden, 1) --> (1, 1)
        print(tf.matmul(outputs[i],weights[i:i+1,:],transpose_b=True))
        the_output.append(tf.matmul(outputs[i], weights[i:i+1,:], transpose_b=True) + b_output[i])
    return the_output

# the_output: [(batch_size, n_input)]*num_steps
the_output = dynamicRNN(inputs, seqlen, w_output, b_output)
# change it to a Tensor
outputY = tf.stack(the_output) # back to: (num_steps, batch_size, n_input)
print("outputY shape: {}".format(outputY.shape))
resultY = tf.transpose(result,[1,0,2]) # swap n_step with batch_size 
cost = tf.reduce_mean(tf.pow(outputY - resultY,2))
#cross_entropy = -tf.reduce_sum(resultY * tf.log(tf.clip_by_value(outputY,1e-10,1.0)))
#train_op = tf.train.RMSPropOptimizer(0.005,0.2).minimize(cost)
#train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
valX,valy = gen_data(False, 900)  # use 1001 as validate produce best test result, use 10 or 100 will produce a shift, and the bigger the number, the better the correlation, why?
testX, testy = gen_data(False, 2200)  # test data doesn't matter


# start training
with tf.Session() as sess:

    print("gen data")
    #tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter("/Users/joost/tensorflow/log",flush_secs=10,graph_def=sess.graph_def)

    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()

    for k in range(1, num_epochs):

        # print("start k={}".format(k))

        tempX,y = gen_data(False, k)
        # tempX has batch_size elements of shape ( num_steps x num_inputs)

        # print((tempX,y))

        traindict = {inputs: tempX, result: y}

        sess.run(train_op, feed_dict=traindict)

        valdict = {inputs: valX, result: valy}

        costval,outputval = sess.run((cost,outputY), feed_dict=valdict)

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
            print("end k={}, cost={}".format(k,costval))

