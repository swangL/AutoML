import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


hidden_dim = 50

#Find how to handle input

#How to controlle the output



class Controller(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Create tokens which maps the amount of options for each layer
        # Recurrent layer
        self.lstm = nn.LSTM(hidden_dim,hidden_dim)

        #Linear layer for each of the blocks
        self.num_tokens = [len(args.shared_rnn_activations)]
        for idx in range(self.args.num_blocks):
            self.num_tokens += [len(args.sizes_dict), len(args.activations_dict)]
        for i in self.num_blocks:
            decoder = torch.nn.Linear(args.controller_hid, self.size)
            self.decoders.append(decoder)
        for idx, size in enumerate(self.num_tokens):
            decoder =
            self.decoders.append(decoder)

    def forward(self, x):
        # unsqueeze to have correct dim for lstm
        x = x.unsqueeze(0)
        x, h = self.lstm(x)
        return x


net = Net()
print(net)
