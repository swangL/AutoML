import torch
import numpy as np
from controller import Controller as ct
import matplotlib.pyplot as plt
from Environment import *


num_rollouts = 10
rollouts = []

#run a lot
#make "batch run" and take the average reward.
#

def trainer(epochs,lr):
    
    cont = ct(lr) 
    #TODO sample an archetecture from the controller
    #return the blocks to be able to train the controller.
    #return the porbalility of picking each action
    accuracy_hist = []
    loss_hist = []
    train_m = Train_model()
    train_m.generate_data(1000,0.2)
    train_batch_size = len(train_m.X_train)
    val_batch_size = len(train_m.X_val)

    for e in range(epochs):  

        print("{}/{}".format(e+1,epochs))
        arch,probs = cont.sample()
        acc = 0
        for i in range(num_rollouts):
            
            # Defining Network
            layers = []
            network = Net(arch, 2, layers)
            net = nn.Sequential(*layers)
            
            #train with the archetectur
            _, accuracy, _, _, _, _ = train_m.train(net,train_batch_size,val_batch_size)
            acc += accuracy
            
        acc /= num_rollouts
        accuracy_hist.append(acc)

        #Here we apply REINFORCE on the controller we optimize in respect to 
        # the accuracy on the test set, from the following equation:
        #--------------  (1/T) sum(∇log p(a(t)|a(t-1):0)R ------#

        # Here T is the totale number of blocks in the archetectur and a(t) is 
        # the action taken in that time step (what number of nodes, or 
        # activationfunction where choosen). 
        # Notice here that the state has been substituted to be a(t-1):0, which 
        # is the whole archetetur up untill timestep t-1. This is to let the 
        # REINFORCE "know" how the archetecture looked, shuch that it is reward/
        # penealised appropiatly. R is our accuracy on the given archetecture

        #TODO verify that this is the REINFORCE behaviour that we want
        cont.optimizer.zero_grad()
        baseline = torch.tensor(0)
        loss = cont.loss(probs,torch.tensor(acc),baseline)
        #print(loss)
        loss_hist.append(float(loss.data))
        loss.backward()
        cont.optimizer.step()

    return accuracy_hist, loss_hist


def main():
    epochs = 50
    lr = 0.001
    acc_his, loss_his = trainer(epochs,lr)

    plt.figure()
    plt.plot(range(epochs), acc_his, 'r', label='Val Acc')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Accuracy')

    plt.figure()
    plt.plot(range(epochs), loss_his, 'b', label='Loss Acc')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
