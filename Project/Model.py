import pdb
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn
from mxnet.gluon.loss import L1Loss

class weighted_L1Loss(L1Loss):
    """
    need a weighted label
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.abs(pred - label)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class Chomp1d(gluon.HybridBlock):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def hybrid_forward(self, F, x):
        if self.chomp_size == 0:
            return x
        else:
            return x[:, :, :-self.chomp_size] # remove right padding

class UpSampling(gluon.HybridBlock):
    """
    upsampling layer, require 4-D tensor
    """
    def __init__(self, scale, sample_type):
        super(UpSampling, self).__init__()
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x):
        ret = F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)
        return ret

class UpSampling1D(gluon.Block):
    """
    upsampling require 4-D tensor, for conv1d, we need reshape
    input: ncw, need to convert to nchw and back
    """
    def __init__(self, scale, sample_type):
        super(UpSampling, self).__init__()
        self.scale = scale
        self.sample_type = sample_type

    def forward(self, x):
        input_shape = x.shape
        assert(len(x.shape) == 3)
        x = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
        ret = x.UpSampling(x, scale=self.scale, sample_type=self.sample_type)
        ret = ret.reshape(input_shape)
        return ret

class SELU(gluon.HybridBlock):
    r"""
    Scaled Exponential Linear Unit (SELU)
        "Self-Normalizing Neural Networks", Klambauer et al, 2017
        https://arxiv.org/abs/1706.02515


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, **kwargs):
        super(SELU, self).__init__(**kwargs)
        self._scale = 1.0507009873554804934193349852946
        self._alpha = 1.6732632423543772848170429916717

    def hybrid_forward(self, F, x):
        return self._scale * F.where(x > 0, x, self._alpha * (F.exp(x) - 1.0))

class Sigmoid(gluon.HybridBlock):
    """
    """
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.sigmoid(x)

class Tanh(gluon.HybridBlock):
    """
    """
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.tanh(x)

class Self_Attn1D(gluon.HybridBlock):
    def __init__(self, in_dim, activation, **kwargs):
        super(Self_Attn1D, self).__init__(**kwargs)
        self.activation = activation
        #self.query_conv = nn.Conv1D(channels=in_dim//8, kernel_size=1)
        #self.key_conv = nn.Conv1D(channels=in_dim//8, kernel_size=1)
        self.query_conv = nn.Conv1D(channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1D(channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1D(channels=in_dim, kernel_size=1)
        self.gamma = self.params.get('gamma', shape=1)

    def hybrid_forward(self, F, x, *args, **params):
        """
        x: (n,c,w)
        """
        #pdb.set_trace()
        batch_size = x.shape[0]
        proj_query = F.transpose(self.query_conv(x), axes=(0,2,1)) # nwc
        proj_key = self.key_conv(x) # ncw
        energy = F.batch_dot(proj_query, proj_key) # nww
        attention = F.softmax(energy) # nww
        proj_value = self.value_conv(x) # ncw

        out = F.batch_dot(proj_value, attention) # ncw
        out = params['gamma']*out + x
        return out

class Self_Attn(gluon.HybridBlock):
    def __init__(self, in_dim, activation, **kwargs):
        super(Self_Attn, self).__init__(**kwargs)
        self.activation = activation
        
        self.query_conv = nn.Conv2d(channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels=in_dim, kernel_size=1)
        self.gamma = gluon.Parameter('gamma', shape=(1,), init=mx.initializer.Xavier(), allow_deferred_init=True)

    def hybrid_forward(self, F, x, *args, **params):
        """
        x: (n,c,w,h)
        """
        batch_size,C,width,height = x.shape
        proj_query = F.transpose(F.reshape(self.query_conv(x), (batch_size,C,-1)), axes=(0,2,1)) # (n,w*h,c)
        proj_key = F.reshape(self.key_conv(x), (batch_size,C,-1)) # (n,c,wh)
        energy = F.batch_dot(proj_query, proj_key) # nww
        attention = F.softmax(energy) # nww
        proj_value = self.value_conv(x) # ncw

        out = F.batch_dot(proj_value, attention) # ncw
        out = F.reshape(batch_size, C, width, height)
        out = params['gamma']*out + x
        return out

class TemporalConvNet(gluon.HybridBlock):

    def __init__(self, input_shape, filter_list, num_filters, dropout=0., Attention=True, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.c = input_shape[1] # input: n,c,w
        print("Input_feature_shape: ",input_shape) 

        with self.name_scope():
            self._convNet = nn.HybridSequential()
            for i,kernel_size in enumerate(filter_list):
                dilation = 2 ** i
                padding = (kernel_size-1)*dilation
                nb_filter = num_filters[i]
                self._convNet.add(
                    nn.Conv1D(channels=nb_filter, kernel_size=kernel_size, strides=1, padding=padding, dilation=dilation, layout='NCW', activation='relu'),
                    nn.BatchNorm(axis=2),
                    #nn.Conv1D(channels=nb_filter, kernel_size=kernel_size, strides=1, padding=padding, dilation=dilation, layout='NCW', activation='relu'),
                    #nn.BatchNorm(axis=2),
                    Chomp1d(padding),
                    nn.Dropout(dropout),
                    Self_Attn1D(nb_filter, 'relu'),
                )
                #print("Attention: ",Attention)
                #if Attention:
                    #self._convNet.add(Self_Attn1D(nb_filter, 'relu'))
            #convNet: (n, nb_filter, w)
            #self._downsample = nn.Conv1D(channels=self.c, kernel_size=1)
            self._downsample = None

    def hybrid_forward(self, F, x, *args, **params):
        """
        """
        #pdb.set_trace()
        convi = self._convNet(x)  #O: (n, num_filter, w)
        if self._downsample is not None: # differ from original
            convi = self._downsample(convi)
        #out = convi + x # (n, c, w)
        out = convi
        #return F.relu(out)
        return out



def build_iters(_data, _label, batch_size=8, splits=[0.9, 0.1, 0.1]):
    _scale = normalize(_data, axis=2)
    label_scale = normalize(_label, axis=2)

    _data = np.atleast_3d(_data)
    _label = np.atleast_3d(_label)
    data_len = len(_data)

    m = int(splits[0]*data_len) 
    k = int(splits[1]*data_len)

    idx = np.random.choice(data_len, size=data_len, replace=False)
    train_idx = idx[:m]
    val_idx = idx[m:m+k]
    test_idx = idx[m+k:]

    X = _data[train_idx, :]
    y = _label[train_idx, :]
    train_iter = mx.io.NDArrayIter(data=X,
                                   label=y,
                                   batch_size=batch_size)
    print("train_data shape: ", X.shape, y.shape)

    X = _data[val_idx, :]
    y = _label[val_idx, :]
    val_iter = mx.io.NDArrayIter(data=X,
                                 label=y,
                                 batch_size=batch_size)
    print("val_data shape: ", X.shape, y.shape)

    X = _data[test_idx, :]
    y = _label[test_idx, :]
    global eval_data_scale, eval_label_scale
    eval_data_scale = data_scale[test_idx, :]
    eval_label_scale = label_scale[test_idx, :]
    test_iter = mx.io.NDArrayIter(data=X,
                                  label=y,
                                  batch_size=batch_size)
    print("test_data shape: ", X.shape, y.shape)
    return train_iter, val_iter, test_iter



class RNNModel(gluon.Block):
    """
    input:
        mode: select which model to use
        vocab_size: corpus dictionary size for word processing, also, it's the one-hot vector dim. It defines nn.Embedding input dim.
        num_embed: embeding/feature size, define nn.Embedding output dim. For word processing, it's usually uses vocab_size.
        num_hidden: hidden unit length/dim. define LSTM output dim. H = (batch_size, num_hidden). This usually means how many features you want to simulate the system. This should be irrelavent of input/output dim.
        num_layers: RNN layer; more layer, more complex, each layer should have same size
        drop_out: to fix over-fitting
        tie_weigths: use Dense before output

    """
    def __init__(self, mode, vocab_size, num_embed, num_hidden, num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            if 0:
                self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            else:
                self.encoder = None
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        if self.encoder:
            output = self.drop(self.encoder(inputs)) # emb

        output, hidden = self.rnn(output, hidden)

        if self.drop:
            output = self.drop(output)

        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


    def predict(self, net, data_iter):
        i = 0
        X = []
        y = []
        p = []
        for data,label in data_iter:
            data = data.as_in_context(model_ctx,)
            data = data.reshape((1, sequence_length, 1))
            output = net(data)
            X.append(data[0].asnumpy())
            y.append(label[0].asnumpy()) # since later we use _data directly
            p.append(output[0].asnumpy()) # since later we use _data directly
        return X,y,p
