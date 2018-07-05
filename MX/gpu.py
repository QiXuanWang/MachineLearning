import sys
from mxnet import gpu
from mxnet import cpu
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import utils
from mxnet.gluon.model_zoo import vision

#net = utils.resnet18(10)
net = vision.resnet18_v1()
ctx = [gpu(0), gpu(1)]
net.initialize(ctx=ctx)

x = nd.random.uniform(shape=(4, 1, 28, 28))
x_list = gluon.utils.split_and_load(x, ctx)
print(net(x_list[0]))
print(net(x_list[1]))

#weight = net[1].params.get('weight')
#print(weight.data(ctx[0])[0])
#print(weight.data(ctx[1])[0])
#try:
#    weight.data(cpu())
#except:
#    print('Not initialized on', cpu())

from mxnet import gluon
from mxnet import autograd
from time import time
from mxnet import init

def train(num_gpus, batch_size, lr):
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)

    ctx = [gpu(i) for i in range(num_gpus)]
    print('Running on', ctx)

    net = utils.resnet18(10)
    net.initialize(init=init.Xavier(), ctx=ctx)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(
        net.collect_params(),'sgd', {'learning_rate': lr})

    for epoch in range(5):
        start = time()
        total_loss = 0
        for data, label in train_data:
            data_list = gluon.utils.split_and_load(data, ctx)
            label_list = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                losses = [loss(net(X), y) for X, y in zip(
                    data_list, label_list)]
            for l in losses:
                l.backward()
            total_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)

        nd.waitall()
        print('Epoch %d, training time = %.1f sec'%(
            epoch, time()-start))

        test_acc = utils.evaluate_accuracy(test_data, net, ctx[0])
        print('         validation accuracy = %.4f'%(test_acc))


train(2, 256, .1)
