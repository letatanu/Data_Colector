import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
class LinearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        return self.linear(x)

# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

inputSize = 1 #x
outputSize = 1 #y
learningRate = 0.1
epoch = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearRegression(inputSize, outputSize).to(device)
inputs = torch.from_numpy(x_train).to(device)
labels = torch.from_numpy(y_train).to(device)
crit = nn.MSELoss()
optim = torch.optim.Adam(params=model.parameters(), lr=learningRate)
for epo in range(epoch):
    model.train()
    optim.zero_grad()
    outputs = model(inputs)
    loss = crit(outputs, labels)
    print(loss)
    loss.backward()
    optim.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad():
    predicted = model(torch.from_numpy(x_train).to(device))
    predicted = predicted.cpu().data.numpy()
    plt.clf()
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()
