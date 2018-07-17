import pdb
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
#from TCN.adding_problem.model import TCN
#from TCN.adding_problem.utils import data_generator
from tcn import TemporalConvNet
from data import gen_data, save_plot

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x) # x: [16,1,376], y1: [16,30,376]
        #pdb.set_trace()
        y2 = self.linear(y1[:, :, -1])  # y2: [16,376]
        return self.linear(y1[:, :, -1]) 
    # torch.nn.Linear
    # input: (N, *, in_features), output: (N, *, out_features)



parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print("Producing data...")
#_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['i(vi)'], shape='ncw') 
_data, _label = gen_data(filename='1cycle_iv_small.txt',input_list=['v(i)'], output_list=['v(pad)'], shape='ncw') 
_label = _label.squeeze()
data_scale = np.max(_data)
label_scale = np.max(_label)
print("Scale: %.2e %.2e"%(data_scale, label_scale))
_data = _data/data_scale # scale
_label = _label/label_scale # scale
_data = np.concatenate([_data]*20)
_label = np.concatenate([_label]*20)


data_len = len(_data)
seq_len = len(_data[0][0])
idx = np.random.choice(data_len, size=data_len, replace=False)
m = int(0.8*data_len)
k = int(0.1*data_len)
train_idx = idx[:m]
val_idx = idx[m:m+k]
test_idx = idx[m+k:]
# TODO: this actually makes tensor (n,w) instead of (n,c,w)
X_train = torch.Tensor(_data[train_idx, :]) # (n,w)
Y_train = torch.Tensor(_label[train_idx, :])
X_val = torch.Tensor(_data[val_idx, :]) # (n,w)
Y_val = torch.Tensor(_label[val_idx, :])
X_test = torch.Tensor(_data[test_idx, :]) # (n,w)
Y_test = torch.Tensor(_label[test_idx, :])


input_channels = 1
n_classes = seq_len
batch_size = args.batch_size
epochs = args.epochs
print(args)

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

if args.cuda:
    cuda = torch.device('cuda')
    model.cuda(device=cuda)
    X_train = X_train.cuda(device=cuda)
    Y_train = Y_train.cuda(device=cuda)
    X_val = X_val.cuda(device=cuda)
    Y_val = Y_val.cuda(device=cuda)
    X_test = X_test.cuda(device=cuda)
    Y_test = Y_test.cuda(device=cuda)

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size()[0], batch_size):
        if i + batch_size > X_train.size()[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        #total_loss += loss.data[0] # valid in 0.3 and error in 0.5
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, X_train.size()[0])
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.8f}'.format(
                epoch, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, cur_loss))
            total_loss = 0


def evaluate(x, y, save=False):
    model.eval()
    predicted = model(x)
    test_loss = F.mse_loss(predicted, y)
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
    if save:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        predicted = predicted.detach().cpu().numpy()
        save_plot(x*data_scale, y*label_scale, predicted*label_scale, 'try8_pt.out', style="keras")
    return test_loss.item()


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    model_name = "try9.pt"
    for ep in range(1, epochs+1):
        train(ep)
        vloss = evaluate(X_val, Y_val)
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep>10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        vloss_list.append(vloss)

    print('-' * 80)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(X_test, Y_test, save=True)
