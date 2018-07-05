import mxnet as mx
from mxnet import gluon

ctx = mx.gpu()
#ctx = mx.cpu()

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.rnn.LSTM(3, 1))
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
params = net.collect_params()
params.reset_ctx(mx.cpu())
print(params)

"""
def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
net = get_net()
net.initialize()
net(x)
w = net[0].weight
b = net[0].bias
print(w)
params = net.collect_params()
print(params)
"""
