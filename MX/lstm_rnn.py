num_inputs = 77
num_hidden = 256
num_outputs = 77

########################
#  Weights connecting the inputs to the hidden layer
########################
Wxg = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxi = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxf = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxo = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01

########################
#  Recurrent weights connecting the hidden layer across time steps
########################
Whg = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whi = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whf = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Who = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01

########################
#  Bias vector for hidden layer
########################
bg = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bi = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bf = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bo = nd.random_normal(shape=num_hidden, ctx=ctx) * .01


########################
# Weights to the output nodes
########################
Why = nd.random_normal(shape=(num_hidden,num_inputs), ctx=ctx) * .01
by = nd.random_normal(shape=num_inputs, ctx=ctx) * .01

params = [Wxg, Wxi, Wxf, Wxo, Whg, Whi, Whf, Who, bg, bi, bf, bo]
params += [Why, by]

for param in params:
    param.attach_grad()
for param in params:
    param.attach_grad()

def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition =nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition =nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
