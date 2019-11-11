import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


hidden_dim = 50

#Find how to handle input

#How to controlle the output



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Recurrent layer
        # YOUR CODE HERE!
        self.lstm = nn.LSTM(hidden_dim,hidden_dim)

    def forward(self, x):
        # unsqueeze to have correct dim for lstm
        x = x.unsqueeze(0)
        x, (h, c) = self.lstm(x)
        return x


net = Net()
print(net)
