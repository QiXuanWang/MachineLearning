#dbsheta/cgan-mxnet
import mxnet as mx
import mxnet.gluon as gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet import autograd

import numpy as np
import time
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.DEBUG)
ctx = mx.gpu()
#ctx = mx.cpu()

batch_size = 128
num_epochs = 5

lr = 0.0002
beta1 = 0.5
beta2 = 0.999
latent_z_size = 100

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=False)

class Generator(gluon.HybridBlock):
    def __init__(self, n_dims=128, **kwargs):
        super(Generator, self).__init__(**kwargs)
        with self.name_scope():
            self.deconv_z = nn.Conv2DTranspose(n_dims * 2, 4, 1, 0)
            self.deconv_label = nn.Conv2DTranspose(n_dims * 2, 4, 1, 0)
            self.deconv2 = nn.Conv2DTranspose(n_dims * 2, 4, 1, 0)
            self.deconv3 = nn.Conv2DTranspose(n_dims, 4, 2, 1)
            self.deconv4 = nn.Conv2DTranspose(1, 4, 2, 1)
            
            self.bn_z = nn.BatchNorm()
            self.bn_label = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()
            self.bn3 = nn.BatchNorm()
    
    
    def hybrid_forward(self, F, x, y):
        x = F.relu(self.bn_z(self.deconv_z(x)))
        
        y = F.expand_dims(y, axis=2)
        y = F.expand_dims(y, axis=2)
        y = F.relu(self.bn_label(self.deconv_label(y)))
        
        z = F.concat(x, y, dim=1)
        
        x = F.relu(self.bn2(self.deconv2(z)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        
        return x
    
    def save(self, filename):
        self.save_params(filename)
        
    def load(self, filename, ctx):
        self.load_params(filename, ctx)


class Discriminator(gluon.HybridBlock):
    def __init__(self, n_dims=128, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(n_dims, 4, 2, 1)
            self.conv2 = nn.Conv2D(n_dims * 2, 4, 2, 1)
            self.conv3 = nn.Conv2D(n_dims * 4, 4, 1, 0)
            self.conv4 = nn.Conv2D(1, 4, 1, 0)
            
            self.bn2 = nn.BatchNorm()
            self.bn3 = nn.BatchNorm()
        
    def hybrid_forward(self, F, x, y):
        x = F.LeakyReLU(self.conv1(x), slope=0.2)
        x = F.LeakyReLU(self.bn2(self.conv2(x)), slope=0.2)
        x = F.LeakyReLU(self.bn3(self.conv3(x)), slope=0.2)
        
        y = F.expand_dims(y, axis=2)
        y = F.expand_dims(y, axis=2)
        y = F.tile(y, [4,4])
        
        x = F.concat(x, y, dim=1)
        x = self.conv4(x)
        
        return x
    
    def save(self, filename):
        self.save_params(filename)
    
    def load(self, filename, ctx):
        self.load_params(filename, ctx)

def build_net():
    netG = Generator()
    netD = Discriminator()

    # initialize the generator and the discriminator
    netG.initialize(mx.init.Normal(0.02), ctx=ctx)
    netD.initialize(mx.init.Normal(0.02), ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2})

    netG.hybridize()
    netD.hybridize()
    return netG, netD, trainerG, trainerD

netG, netD, trainerG, trainerD = build_net()
# loss
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

def train():
    for epoch in range(num_epochs):
        btic = time.time()
        i = 0
        #import pdb
        #pdb.set_trace()
        
        for data, labels in test_data:
            real_label = nd.ones([labels.shape[0], ], ctx=ctx)
            fake_label = nd.zeros([labels.shape[0] ], ctx=ctx)
            labels = labels.as_in_context(ctx)
            x = data.as_in_context(ctx)

            y = nd.one_hot(labels, depth=10)
            #z = mx.nd.random_normal(0, 1, shape=(batch_size, latent_z_size, 1, 1), ctx=ctx)
            z = mx.nd.random_normal(0, 1, shape=(labels.shape[0], latent_z_size, 1, 1), ctx=ctx)
            
            #y_z = mx.nd.array(np.random.randint(0, 9, size=batch_size), ctx=ctx)
            y_z = mx.nd.array(np.random.randint(0, 9, size=labels.shape[0]), ctx=ctx)
            y_z = nd.one_hot(y_z, depth=10)
            
            # Train Discriminator
            with autograd.record():
                output = netD(x, y)
                errD_real = loss(output, real_label)
                logging.info(f"YuWang: shapes: x: {x.shape}, y:{y.shape}, out: {output.shape}, real_label: {real_label.shape}")
                
                fake = netG(z, y_z)
                output = netD(fake.detach(), y_z)
                errD_fake = loss(output, fake_label)
                
                logging.info(f"YuWang: shapes: out: {output.shape}, real_label: {real_label.shape}, fake_label: {fake_label.shape}, errD_real: {errD_real.shape}, errD_fake: {errD_fake.shape}")
                errD = errD_real + errD_fake
                errD.backward()
            trainerD.step(data.shape[0])
            
            # Train Generator
            with autograd.record():
                fake = netG(z, y_z)
                output = netD(fake, y_z)
                errG = loss(output, real_label)
                errG.backward()
            trainerG.step(data.shape[0])
            
            if i % 50 == 0:
                logging.info(f'speed: {batch_size / (time.time() - btic)} samples/s')
                logging.info(f'discriminator loss = {nd.mean(errD).asscalar()}, generator loss = {nd.mean(errG).asscalar()} at iter {i} epoch {epoch}')

            i = i + 1
            btic = time.time()
        if epoch % 5 == 0:
            netD.save_params("netD.params")
            netG.save_params("netG.params")



def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

def plot(netG):
    num_image = 8
    for i in range(num_image):
        latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
        y_z = mx.nd.array(np.random.randint(0, 9, size=1),ctx=ctx)
        y_z = nd.one_hot(y_z, depth=10)

        img = netG(latent_z, y_z)

        plt.subplot(2,4,i+1)
        visualize(img[0])
    plt.show()

if __name__ == "__main__":
    if os.path.exists("netG.params"):
        print("Loading netG.params")
        netG = Generator()
        netG.load("netG.params", ctx)
        print("Done")
    else:
        train()
    plot(netG)
