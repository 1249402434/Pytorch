import torch
from torch import nn
from torch.autograd import Variable

rnn = nn.RNN(100, 10)
print(rnn.reset_parameters())

print("hidden size:", rnn.weight_hh_l0.shape)
print("input size:", rnn.weight_ih_l0.shape)

rnn2 = nn.RNN(input_size=100, hidden_size=20, num_layers=1)

x = torch.randn(10, 3, 100)
out, h = rnn2(x, torch.zeros(1, 3, 20))
print(out.shape, h.shape)

content = [1,2,3]
for index, ele in enumerate(content, 1):
    print(index, ele)
