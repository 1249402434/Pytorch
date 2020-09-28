import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

hidden_size = 16
output_size = 1
num_steps = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            batch_first=True,
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_hidden):
        output, prev_hidden = self.rnn(x, prev_hidden)
        output = output.view(-1, hidden_size)
        out = self.linear(output)
        '''
        原本out维度为(seq,1),执行如下代码后会在前新插入一个维度,变为(1,seq,1)
        '''
        out = out.unsqueeze(dim=0)

        return out, prev_hidden


model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

prev_hidden = torch.zeros(1,1,hidden_size)

for e in range(3000):
    start = np.random.randint(10, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_steps, 1)

    X = torch.tensor(data[:-1]).float().view(1, num_steps - 1, 1)
    Y = torch.tensor(data[1:]).float().view(1, num_steps - 1, 1)

    out, prev_hidden = model(X, prev_hidden)
    loss = criterion(out, Y)
    # detach()返回一个新的从当前图中分离的Variable, 它不需要梯度
    prev_hidden = prev_hidden.detach()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e+1) % 100 ==0:
        print("Epoch[%d] loss:%f"%((e+1), loss.item()))

# prediction
start = np.random.randint(10, size=1)[0]
test_time_steps = np.linspace(start, start+10, num_steps)
test_data = np.sin(test_time_steps)
test_data = test_data.reshape(num_steps, 1)

x = torch.tensor(test_data[:-1]).float().view(1, num_steps - 1, 1)

prediction = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    out, prev_hidden = model(input, prev_hidden)
    input = out
    prediction.append(out.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
plt.scatter(test_time_steps[:-1], x, s=90)
plt.plot(test_time_steps[:-1], x)

plt.scatter(test_time_steps[:-1], prediction)


plt.show()






