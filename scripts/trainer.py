import torch
import numpy as np
from controller import Controller as ct
import matplotlib.pyplot as plt
from Environment import *

num_rollouts = 10

#run a lot
#make "batch run" and take the average reward.
#

def trainer(epochs,data_set,lr):
    
    # Change HERE if conv = True
    cont = ct(lr, True)
    if torch.cuda.is_available():
        print('##converting Controller to cuda-enabled')
        cont.cuda()
        for decoder in cont.decoders:
            decoder.cuda()
    #TODO sample an archetecture from the controller
    #return the blocks to be able to train the controller.
    #return the porbalility of picking each action
    accuracy_hist = []
    loss_hist = []


    plot = False

    params = {
        "num_epochs": 200,
        "opt": "SGD",
        "lr": 0.01
    }

    train_m = Train_model(params)

    if data_set == "MOONS":
        train_m.moon_data(num_samples=1000, noise_val=0.2)
    elif data_set == "MNIST":
        train_m.mnist_data(num_classes=10)
    elif data_set == "CONV":
        train_m.conv_data(batch_size_train=64, batch_size_val=32)

    for e in range(epochs):

        # if (e+1) % (epochs/10) == 0:
        print("{}/{}".format(e+1,epochs), flush=True)

        arch,probs = cont.sample()
        #Notice here we also get the probability of the termination!


        # Defining Network
        layers = []
        if data_set == "MOONS":
            network = Net_MOONS(string=arch, in_features=2, num_classes=2, layers=layers)
            net = nn.Sequential(*layers)
            print(net)
            if torch.cuda.is_available():
                print('#converting child to cuda-enabled', flush=True)
                net.cuda()
            accuracy = train_m.train(net=net, train_batch_size=len(train_m.X_train), val_batch_size=len(train_m.X_val), plot=plot)
        elif data_set == "MNIST":
            network = Net_MNIST(string=arch, in_features=784, num_classes=10, layers=layers)
            net = nn.Sequential(*layers)
            print(net)
            if torch.cuda.is_available():
                print('#converting child to cuda-enabled', flush=True)
                net.cuda()
            accuracy = train_m.train(net=net, train_batch_size=len(train_m.X_train), val_batch_size=len(train_m.X_val), plot=plot)
        elif data_set == "CONV":
            network = Net_CONV(string=arch, in_channels=1, num_classes=10, layers=layers)
            net = nn.Sequential(*layers)
            print(net)
            if torch.cuda.is_available():
                print('#converting child to cuda-enabled', flush=True)
                net.cuda()
            accuracy = train_m.train_conv(net=net, plot=plot)
        # net = nn.Sequential(*layers)

        '''
        if torch.cuda.is_available():
            #print('#converting child to cuda-enabled', flush=True)
            net.cuda()
        '''
        # accuracy = train_m.train(net=net, train_batch_size=len(train_m.X_train), val_batch_size=len(train_m.X_val), plot=plot)

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
        loss = cont.loss(probs,accuracy,baseline)
        #print(loss)
        loss_hist.append(float(loss.data))
        loss.backward()
        cont.optimizer.step()

    return accuracy_hist, loss_hist


def main():
    epochs = 10
    net_type = "CONV"
    lr = 0.01

    acc_his, loss_his = trainer(epochs,net_type,lr)
    
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
