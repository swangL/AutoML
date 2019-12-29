import torch
import numpy as np
from controller import Controller as ct
import matplotlib.pyplot as plt
from Environment import *
from particletracking import *


clip_norm = 0

#run a lot
#make "batch run" and take the average reward.
#

def trainer(epochs,data_set,lr, cttype="ct"):

    # Change HERE if conv = True
    if cttype == "econst" or cttype == "emoving" or cttype == "edynamic":
        cont = ct(lr, ent=True)
    else:
        cont = ct(lr)

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
    depth = []
    sample_networks = []

    plot = False

    params = {
        "num_epochs": 200,
        "opt": "SGD",
        "lr": 0.01
    }
    decay = 0.95
    baseline = 0.85
    train_m = Train_model(params)

    if data_set == "MOONS":
        train_m.moon_data(num_samples=1000, noise_val=0.2)
    elif data_set == "MNIST":
        train_m.mnist_data(num_classes=10)
    elif data_set == "CONV":
        cont.conv=True
        train_m.conv_data(data_set_name="MNIST", batch_size_train=64, batch_size_val=32)
    elif data_set == "PARTICLE":
        train_m = Train_model_particle(params)
        train_m.particle_data()
        particle_losses=[]
    elif data_set == "PARTICLECONV":
        cont.conv = True
        train_m = Train_model_particle(params)
        train_m.particle_data_conv()
        particle_losses=[]

    for e in range(epochs):

        arch,probs = cont.sample()
        print(arch)
        #Notice here we also get the probability of the termination!


        # Defining Network
        layers = []
        if data_set == "MOONS":
            network = Net_MOONS(string=arch, in_features=2, num_classes=2, layers=layers)
            net = nn.Sequential(*layers)

            if torch.cuda.is_available():
                #print('#converting child to cuda-enabled', flush=True)
                net.cuda()
            accuracy = train_m.train(net=net, train_batch_size=len(train_m.X_train), val_batch_size=len(train_m.X_val), plot=plot)
        elif data_set == "MNIST":
            network = Net_MNIST(string=arch, in_features=784, num_classes=10, layers=layers)
            net = nn.Sequential(*layers)

            if torch.cuda.is_available():
                #print('#converting child to cuda-enabled', flush=True)

                net.cuda()
            accuracy = train_m.train(net=net, train_batch_size=len(train_m.X_train), val_batch_size=len(train_m.X_val), plot=plot)
        elif data_set == "CONV":
            network = Net_CONV(string=arch, img_size = 28, in_channels=1, num_classes=10, layers=layers)

            net = nn.Sequential(*layers)
            print(net)
            if torch.cuda.is_available():
                #print('#converting child to cuda-enabled', flush=True)
                net.cuda()
            accuracy = train_m.train_conv(net=net, plot=plot)
        elif data_set == "PARTICLE":
            network = Net_PARTICLE(string=arch, in_features=2025, num_classes=3, layers=layers)
            net = nn.Sequential(*layers)
            if torch.cuda.is_available():
                net.cuda()
            accuracy, particle_loss = train_m.particle_train(net, plot)
            particle_losses.append(particle_loss)
        elif data_set == "PARTICLECONV":
            network = Partivle_Net_CONV(string=arch, img_size = 45, num_classes=3, layers=layers,  in_channels=1)
            net = nn.Sequential(*layers)
            if torch.cuda.is_available():
                net.cuda()
            accuracy, particle_loss = train_m.particle_train(net, plot)
            particle_losses.append(particle_loss)


        # accuracy = train_m.train(net=net, train_batch_size=len(train_m.X_train), val_batch_size=len(train_m.X_val), plot=plot)
        accuracy = torch.tensor(accuracy)
        accuracy_hist.append(accuracy)
        depth.append(len(arch))
        if e > (epochs-50):
            sample_networks.append(arch)
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

        # moving average baseline
        if cttype == "moving" or cttype == "emoving":
            rewards = accuracy
            baseline = decay * baseline + (1 - decay) * rewards
        # dynamic baseline
        if cttype == "dynamic" or cttype == "edynamic":
            if accuracy>baseline:
                baseline*=1.01
        loss = cont.loss(probs,accuracy,baseline)
        if (e+1) % (epochs/10) == 0:
            print("{}/{}".format(e+1,epochs), flush=True)
            print("Arch", arch)
            print("acc",accuracy)
            print("loss",loss)
            print("baseline", baseline)
            print()
        loss_hist.append(float(loss.data))
        loss.backward()
        if clip_norm > 0:
            nn.utils.clip_grad_norm_(cont.parameters(), clip_norm)
        cont.optimizer.step()

    if data_set == "PARTICLE" or data_set == "PARTICLECONV":
        print(particle_losses)
    return accuracy_hist, loss_hist, cont.probs_layer_1, depth, sample_networks


def main():
    
    epochs = 10
    net_type = "MOONS"
    lr = 0.01
    ct = "divnotct"

    acc_his, loss_his, probs_layer_1, depths, archs = trainer(epochs,net_type,lr,ct)

    plt.figure()
    plt.plot(range(epochs), acc_his, 'r', label='Val Acc')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Accuracy')
    print(depths)
    plt.figure()
    plt.plot(range(epochs), depths, 'r', label='Val Acc')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('depth w. activations')

    plt.figure()
    plt.plot(range(epochs), loss_his, 'b', label='Val Loss')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
