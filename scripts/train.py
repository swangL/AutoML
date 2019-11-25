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
    
    data_set = "MNIST"
    plot = False
    
    params = {
        "num_epochs": 1000,
        "opt": "Adam",
        "lr": 0.01
    }

    train_m = Train_model(params)

    if data_set is "MOONS":
        train_m.moon_data(num_samples=1000, noise=0.2)
    elif data_set is "MNIST":
        train_m.mnist_data(num_classes=10)

    for e in range(epochs):  

        print("{}/{}".format(e+1,epochs))
        arch,probs = cont.sample()
        #Notice here we also get the probability of the termination!
    
        #TODO train the archetecture in the environment, and get the loss and accuracy as an return value
        
        # Defining Network
        layers = []
        if data_set is "MOONS":
            network = Net_MOONS(string=arch, in_features=2, num_classes=2, layers=layers)
        elif data_set is "MNIST":
            network = Net_MNIST(string=arch, in_features=784, num_classes=10, layers=layers)

        net = nn.Sequential(*layers)
        print(net)

        # Set variables used to train neural network
        # If we do not wish to use batches, set batch_size equals to the length
        # of the dataset 
       # num_epochs = 200
       # train_batch_size = 50
       # val_batch_size = 50
       # opt = "Adam"
       # learning_rate = 0.01

        accuracy = train_m.train(net=net, train_batch_size=len(train_m.X_train), val_batch_size=len(train_m.X_val), plot=plot)

        accuracy = torch.tensor(accuracy)
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
        loss = cont.loss(probs,torch.tensor(acc),baseline)
        #print(loss)
        loss_hist.append(float(loss.data))
        loss.backward()
        cont.optimizer.step()

    return accuracy_hist, loss_hist


def main():
    epochs = 10
    lr = 0.01
    acc_his, loss_his = trainer(epochs,lr)

    plt.figure()
    plt.plot(range(epochs), acc_his, 'r', label='Val Acc')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Accuracy')

    plt.figure()
    plt.plot(range(epochs), loss_his, 'b', label='Val Loss')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
