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
        self.lstm = nn.LSTM(vocab_size,hidden_size)
        
        # Output layer
        self.l_out = nn.Linear(in_features=hidden_dim,
                            out_features=vocab_size,
                            bias=False)
        
    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x


net = Net()
print(net)