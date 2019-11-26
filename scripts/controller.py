import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helpers import get_variable


#Find how to handle input

#How to controlle the output

#args implementation
sizes_dict={
    0:"term",
    1:2,
    2:4,
    3:8,
    4:16,
    5:32,
    6:64,
}

channels_dict={
    0:"term",
    1:2,
    2:4,
    3:8,
    4:16,
    5:32,
    6:64,
}

kernels_dict={
    0:3,
    1:5,
    2:7,
}

activations_dict={
    0:"ReLU",
    1:"Tanh",
    2:"Sigmoid",
}

num_blocks = 12
hidden_dim = 50
dictionaries = [activations_dict,sizes_dict]

class Controller(nn.Module):
    def __init__(self,lr,Conv=False):

        self.Conv = Conv

        super(Controller, self).__init__()
        # Create tokens which maps the amount of options for each layer
        # Recurrent layer
        self.lstm = nn.LSTMCell(hidden_dim,hidden_dim)

        #Linear layer for each of the block - decoder
        self.decoders=[]
        if Conv:
            self.num_tokens = [len(activations_dict)]
            for i in range(num_blocks):
                # TODO (Mads): implement variable amount of choices / help guide
                # We can not use a .append here sinze num_token is a nontype object, we can just cast it to be a list, but this works just fine
                self.num_tokens += [len(channels_dict), len(kernels_dict), len(activations_dict)]
            for size in self.num_tokens:
                decoder = torch.nn.Linear(hidden_dim, size)
                self.decoders.append(decoder)
        else:
            self.num_tokens = [len(activations_dict)]
            for i in range(num_blocks):
                # TODO (Mads): implement variable amount of choices / help guide
                # We can not use a .append here sinze num_token is a nontype object, we can just cast it to be a list, but this works just fine
                self.num_tokens += [len(sizes_dict), len(activations_dict)]
            for size in self.num_tokens:
                decoder = torch.nn.Linear(hidden_dim, size)
                self.decoders.append(decoder)

        # for i in range(num_blocks):
        #     # TODO (Mads): implement get_variable amount of choices / help guide
        #     # We can not use a .append here sinze num_token is a nontype object, we can just cast it to be a list, but this works just fine
        #     self.num_tokens += [len(sizes_dict), len(activations_dict)]
        # for size in self.num_tokens:
        #     decoder = torch.nn.Linear(hidden_dim, size)
        #     self.decoders.append(decoder)

        self.optimizer = optim.Adam(self.parameters(), lr = lr)

    # You can see the forward pass as an action taken by the agent
    def forward(self, inputs, hidden, block_id):
        # unsqueeze to have correct dim for lstm
        h, c = self.lstm(inputs, hidden)
        choice = block_id%2
        logits = self.decoders[choice](h)
        # TODO(Mads): Softmax temperature and added exploration:
        # if self.args.mode == 'train':
        #     logits = (tanh_c*F.tanh(logits))
        # logits, hidden
        return logits.squeeze(), (h, c)

    #REINFORCE HERE v v v v v v v
    def loss(self, log_prob, accuracy , baseline):
        R = torch.ones(1)*accuracy
        return -torch.mean(torch.mul(log_prob, get_variable(R)))

    # The sample here is then the whole episode where the agent takes x amounts of actions, at most num_blocks
    def sample(self):
        # tuple of h and c
        hidden = (get_variable(torch.zeros(1,hidden_dim), requires_grad=False), get_variable(torch.zeros(1,hidden_dim), requires_grad=False))
        input = get_variable(torch.zeros(1,hidden_dim), requires_grad=False)
        arch = []
        prob_list = []
        logProb_list = []
        # 1 block includes hidden and activation as such num_block*2 + 1 since we want to append Dense
        
        indx = 0

        for block_id in range(1,num_blocks*2+1):
            #handle terminate argument
            #parse last hidden using overwrite
            logits, hidden = self.forward(input, hidden, block_id)
            # use logits to make choice

            probs = F.softmax(logits, dim=-1)
            print(logits.shape)
            log_prob = F.log_softmax(logits, dim=-1)
            # draw from probs
            action = probs.multinomial(num_samples=1).data
            #append to return list which is used as the probs
            logProb_list.append(log_prob.gather(0,action))
            # determine whether activation or hidden
            if self.Conv:
                choice = block_id - indx
                if choice%3==0:
                    arch.append(activations_dict[int(action)])
                    indx = block_id
                elif choice%2==0:
                    arch.append(kernels_dict[int(action)])
                else:
                    value = channels_dict[int(action)]
                    if value=="term":
                        break
                    else:
                        arch.append(value)
            else:
                if block_id%2==0:
                    arch.append(activations_dict[int(action)])
                else:
                    value = sizes_dict[int(action)]
                    if value=="term":
                        break
                    else:
                        arch.append(value)
        #child = self.create_model(activations, nodes)
        #pigerne regner nok med at vi giver en streng i form [node,act,node,act,...,node ]. Lige nu kan vi returnere [act,node,act,node], det skal vi bare lige havde afklaret mandag. Nu bygger jeg i hverfald rollout/archetectur return som [act,node,act,....,act]

        logProb_list = torch.cat(logProb_list,dim=-1)
        #return activations, nodes
        return arch, logProb_list
        #return logits, log_prob

# Test the class here:
test_class = True
if test_class:
    net = Controller(0.1, True)
    if torch.cuda.is_available():
        print('##converting network to cuda-enabled')
        net.cuda()
    arch,l = net.sample()
    print("Network archtecture:")
    print(arch)
    print()
    print("The prob for each pick: ", l)
