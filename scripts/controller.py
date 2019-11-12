import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


hidden_dim = 50

#Find how to handle input

#How to controlle the output

#args implementation
sizes_dict={
    0:"term",
    1:"Dense(1)",
    2:"Dense(2)",
}
activations_dict={
    0:"relu",
    1:"tanh",
    2:"sigmoid",
}
num_blocks=5
hidden_dim = 50


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        # Create tokens which maps the amount of options for each layer
        # Recurrent layer
        self.lstm = nn.LSTMCell(hidden_dim,hidden_dim)

        #Linear layer for each of the block - decoder
        self.decoders=[]
        self.num_tokens = [len(activations_dict)]
        for i in range(num_blocks):
            # TODO (Mads): implement variable amount of choices / help guide
            self.num_tokens = self.num_tokens.append([len(sizes_dict), len(activations_dict)])
        for size in self.num_tokens:
            decoder = torch.nn.Linear(hidden_dim, size)
            self.decoders.append(decoder)


    # You can see the forward pass as an action taken by the agent
    def forward(self, inputs, hidden, block_id):
        # unsqueeze to have correct dim for lstm
        h, c = self.lstm(inputs, hidden)
        logits = self.decoders[block_id](h)
        # TODO(Mads): Softmax temperature and added exploration:
        # if self.args.mode == 'train':
        #     logits = (tanh_c*F.tanh(logits))
        # logits, hidden
        return logits.squeeze(), (h, c)


    # The sample here is then the whole episode where the agent takes x amounts of actions, at most num_blocks
    def sample(self):
        # default input
        inputs = torch.zeros(1,hidden_dim)
        # tuple of h and c
        hidden = (torch.zeros(1,hidden_dim), torch.zeros(1,hidden_dim))
        activations = []
        nodes = []
        # 1 block includes hidden and activation as such num_block*2 + 1 since we want to append Dense
        for block_id in range(num_blocks*2+1):
            #handle terminate argument
            #parse last hidden using overwrite
            logits, hidden = self.forward(inputs, hidden, block_id)
            # use logits to make choice
            probs = F.softmax(logits, dim=-1)
            # we need log prob for reward
            log_prob = F.log_softmax(logits, dim=-1)
            # draw from probs
            action = probs.multinomial(num_samples=1).data
            # determine whether activation or hidden
            if block_id%2==0:
                activations.append(activations_dict[int(action)])
            else:
                value = sizes_dict[int(action)]
                if value=="term":
                    break
                else:
                    nodes.append(value)
        print(activations)
        print(nodes)
        #child = self.create_model(activations, nodes)
        return activations, nodes
        #return logits, log_prob

# Test the class here:
test_class = True
if test_class:
    net = Controller()
    act,nodes = net.sample()
    print("Network archtecture:\n act: {}, nodes: {}".format(act,nodes))
