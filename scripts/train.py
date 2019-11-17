import torch
import numpy as np
from controller import Controller as ct
import matplotlib.pyplot as plt
# from environemt import * <-- when we get the girls


def trainer(epochs,lr):
    
    cont = ct(lr) 
    #TODO sample an archetecture from the controller
    #return the blocks to be able to train the controller.
    #return the porbalility of picking each action
    accuracy_hist = []
    loss_hist = []
    for e in range(epochs):  
        arch,probs = cont.sample()
        #Notice here we also get the probability of the termination!
    
        #TODO train the archetecture in the environment, and get the loss and accuracy as an return value
        accuracy = torch.tensor(0.5)
        accuracy_hist.append(accuracy)

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
        loss = cont.loss(probs,accuracy,baseline)
        loss_hist.append(float(loss.data))
        loss.backward()
        cont.optimizer.step()
    return accuracy_hist, loss_hist


def main():
    epochs = 50
    lr = 0.001
    acc_his, loss_his = trainer(epochs,lr)

if __name__ == "__main__":
    main()
