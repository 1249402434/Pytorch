import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2019)
torch.random.manual_seed(2019)

#只读取第2列的数值数据
data = pd.read_csv('./flights.csv', usecols=[2])
plt.plot(data)
plt.show()

data = data.dropna()
dataset = data.values
dataset = dataset.astype('float32')

max_value = max(dataset)
min_value = min(dataset)

scalar = max_value - min_value
dataset = list(map(lambda x: (x-min_value) / scalar, dataset))

#以前两天的数据预测下一天的数据
def create_dataset(dataset, look_back = 2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i+look_back])
        dataY.append(dataset[i+look_back])

    return np.array(dataX), np.array(dataY)

data_X, data_Y = create_dataset(dataset)

train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size

train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]

train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

#转化为张量数据
train_X = torch.from_numpy(train_X)
train_Y = torch.from_numpy(train_Y)
test_X = torch.from_numpy(test_X)


class Lstm(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 4, output_size = 1, num_layer = 2):
        super(Lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x

model = Lstm(2, 4, 1, 2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

for e in range(1000):
    var_x = Variable(train_X)
    var_y = Variable(train_Y)

    out = model(var_x)
    loss = criterion(out, var_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e+1) % 100 == 0:
        print("Epoch[%d] loss:%f"%((e+1),loss.item()))

model.eval()
data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)

var_data = Variable(data_X)
pred = model(var_data)
pred = pred.view(-1).data.numpy()

plt.plot(dataset, 'b', label = 'real')
plt.plot(pred, 'r', label = 'prediction')
plt.legend(loc='best')
plt.show()






