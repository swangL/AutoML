import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import torch.optim as optim
from torch.autograd import get_variable
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets
import math
import torchvision
from torchvision import datasets


def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    ys = torch.argmax(ys,dim=-1)
    ts = torch.argmax(ts,dim=-1)

    return torch.mean(torch.eq(ys,ts.type(torch.LongTensor)).type(torch.FloatTensor)).data.numpy()

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def cross_entropy(ys, ts):
    # computing cross entropy per sample
    cross_entropy = -torch.sum(ts * torch.log(ys), dim=1, keepdim=False)
    # averaging over samples
    return torch.mean(cross_entropy)

# Network
class Net_MOONS(nn.Module):

    # Constructor to build network
    def __init__(self, string, in_features, num_classes, layers):

        # Inherit from parent constructor
        super(Net_MOONS, self).__init__()

        num_input = in_features

        # Break down string sent from Controller
        # and add layers in network based on this
        for s in string:

            # If element in string is not a number (i.e. an activation function)
            if s is 'ReLU':
                layers.append(nn.ReLU())
            elif s is 'Tanh':
                layers.append(nn.Tanh())
            elif s is 'Sigmoid':
                layers.append(nn.Sigmoid())
            # If element in string is a number (i.e. number of neurons)
            else:
                s_int = int(s)
                layers.append(nn.Linear(num_input, s_int))
                num_input = s_int

        # Last layer with output 2 representing the two classes
        layers.append(nn.Linear(num_input, num_classes))
        layers.append(nn.Softmax(dim=-1))

# Network
class Net_MNIST(nn.Module):

    # ('Conv', '10', '5', 'ReLU', 'Linear', '6', '5', 'ReLU')

    # Constructor to build network
    def __init__(self, string, in_features, num_classes, layers):

        # Inherit from parent constructor
        super(Net_MNIST, self).__init__()

        num_input = in_features

        # Break down string sent from Controller
        # and add layers in network based on this
        for s in string:

            # If element in string is not a number (i.e. an activation function)
            if s is 'ReLU':
                layers.append(nn.ReLU())
            elif s is 'Tanh':
                layers.append(nn.Tanh())
            elif s is 'Sigmoid':
                layers.append(nn.Sigmoid())
            # If element in string is a number (i.e. number of neurons)
            else:
                s_int = int(s)
                layers.append(nn.Linear(num_input, s_int))
                num_input = s_int

        # Last layer with output 2 representing the two classes
        layers.append(nn.Linear(num_input, num_classes))
        layers.append(nn.Softmax(dim=-1))


class Train_model():

    def __init__(self, params):
        self.X_train = 0
        self.y_train = 0
        self.X_val = 0
        self.y_val = 0
        self.X_test = 0
        self.y_test = 0
        self.params = params

    def moon_data(self, num_samples, noise_val):
        # num_samples should be divisable by 5

        # Import dataset
        X, y = make_moons(n_samples=num_samples, noise=noise_val)

        # Define interval used to split data into
        # train, val and test
        interval_1 = int((num_samples/5)*3)
        interval_2 = int((num_samples/5)*4)

        # Define train, validation, and test sets
        self.X_train = X[:interval_1].astype('float32')
        self.X_val = X[interval_1:interval_2].astype('float32')
        self.X_test = X[interval_2:].astype('float32')

        # and labels
        self.y_train = y[:interval_1].astype('int32')
        self.y_val = y[interval_1:interval_2].astype('int32')
        self.y_test = y[interval_2:].astype('int32')

        self.X_train = get_variable(torch.from_numpy(self.X_train))
        self.y_train = get_variable(torch.from_numpy(onehot(self.y_train,2))).float()

        self.X_val = get_variable(torch.from_numpy(self.X_val))
        self.y_val = get_variable(torch.from_numpy(onehot(self.y_val,2))).float()

        self.X_test = get_variable(torch.from_numpy(self.X_test))
        self.y_test = get_variable(torch.from_numpy(onehot(self.y_test,2))).float()

    def mnist_data(self, num_classes):

        # Import dataset
        data = np.load('mnist.npz')

        # Define interval used to split data into
        # train, val and test
        interval_1 = 1000
        interval_2 = 500

        # Define train, validation, and test sets
        self.X_train = data['X_train'][:interval_1].astype('float32')
        self.X_val = data['X_valid'][:interval_2].astype('float32')
        self.X_test = data['X_test'][:interval_2].astype('float32')

        # and labels
        self.y_train = data['y_train'][:interval_1].astype('int32')
        self.y_val = data['y_valid'][:interval_2].astype('int32')
        self.y_test = data['y_test'][:interval_2].astype('int32')

        self.X_train = get_variable(torch.from_numpy(self.X_train))
        self.y_train = get_variable(torch.from_numpy(onehot(self.y_train,num_classes))).float()

        self.X_val = get_variable(torch.from_numpy(self.X_val))
        self.y_val = get_variable(torch.from_numpy(onehot(self.y_val,num_classes))).float()

        self.X_test = get_variable(torch.from_numpy(self.X_test))
        self.y_test = get_variable(torch.from_numpy(onehot(self.y_test,num_classes))).float()




    def plotter(self, accuracies, losses, val_accuracies, val_losses):

        plot_train_losses, plot_val_losses, plot_train_accuracies, plot_val_accuracies = [], [], [], []

        divider = 10
        divide_data = False

        if divide_data:
            for i in range(int(len(val_losses) / divider)):
                plot_val_losses.append(val_losses[i*divider])
                plot_val_accuracies.append(val_accuracies[i*divider])


            for i in range(int(len(losses) / divider)):
                plot_train_losses.append(losses[i*divider])
                plot_train_accuracies.append(accuracies[i*divider])

            val_epochs = int(len(val_losses) / divider)
            train_epochs = int(len(losses) / divider)

        else:

            plot_train_losses = losses
            plot_train_accuracies = accuracies
            plot_val_losses = val_losses
            plot_val_accuracies = val_accuracies

            val_epochs = len(val_losses)
            train_epochs = len(losses)

        plt.figure()
        plt.plot(range(train_epochs), plot_train_losses, 'r', label='Train Loss')
        plt.plot(range(val_epochs), plot_val_losses, 'b', label='Val Loss')
        plt.legend()
        plt.xlabel('Updates')
        plt.ylabel('Loss')

        plt.figure()
        plt.plot(range(train_epochs), plot_train_accuracies, 'r', label='Train Acc')
        plt.plot(range(val_epochs), plot_val_accuracies, 'b', label='Val Acc')
        plt.legend()
        plt.xlabel('Updates')
        plt.ylabel('Accuracy')
        plt.show()

    def train(self,net,train_batch_size,val_batch_size, plot):

        if self.params["opt"] is "Adam":
            optimizer = optim.Adam(net.parameters(), lr=self.params["lr"])

        elif self.params["opt"] is "SGD":
            optimizer = optim.SGD(net.parameters(), lr=self.params["lr"])

        early_stop = True

        accuracies, losses, val_accuracies, val_losses = [], [], [], []

        train_loader = math.ceil(len(self.X_train)/train_batch_size)
        val_loader = math.ceil(len(self.X_val)/val_batch_size)

        # Variables used for EarlyStopping
        es_old_val, es_new_val, counter = 0, 0, 0
        es_range = 0.001
        es_limit = 30

        for e in range(self.params["num_epochs"]):

            # --------------- train the model --------------- #
            for batch in range(train_loader):

                optimizer.zero_grad()

                if batch == (train_loader - 1):
                    slce = slice(batch * train_batch_size, -1)
                else:
                    slce = slice(batch * train_batch_size, (batch + 1) * train_batch_size)

                preds = net(self.X_train[slce])
                loss = cross_entropy(preds, self.y_train[slce])

                loss.backward()
                optimizer.step()

                acc = accuracy(preds, self.y_train[slce])
                accuracies.append(acc)
                losses.append(loss.data.numpy())

            # --------------- validate the model --------------- #
            for batch in range(val_loader):

                if batch == (val_loader - 1):
                    val_slce = slice(batch * val_batch_size, -1)
                else:
                    val_slce = slice(batch * val_batch_size, (batch + 1) * val_batch_size)

                val_preds = net(self.X_val[val_slce])

                val_loss = cross_entropy(val_preds, self.y_val[val_slce])
                val_acc = accuracy(val_preds, self.y_val[val_slce])
                val_losses.append(val_loss.data.numpy())
                val_accuracies.append(val_acc)

            if early_stop:
                # EarlyStopping
                if e == 0:
                    es_old_val = float(val_acc)
                else:
                    es_new_val = float(val_acc)

                    if abs(es_old_val - es_new_val) <= es_range:
                        counter += 1
                        if counter == es_limit:
                            break
                    else:
                        counter = 0
                        es_old_val = float(val_acc)



            #if e % 10 == 0:
            #fsprint("Epoch %i: "
            #    "Train Accuracy: %0.3f"
            #    "\tVal Accuracy: %0.3f"
            #    "\tTrain Loss: %0.3f"
            #    "\tVal Loss: %0.3f"
            #    % (e, accuracies[-1], val_accuracies[-1], losses[-1], val_losses[-1]))


        # --------------- test the model --------------- #
        test_preds = net(self.X_test)
        test_loss = cross_entropy(test_preds, self.y_test)
        test_acc = accuracy(test_preds, self.y_test)

        #self.plotter(accuracies, losses, val_accuracies, val_losses)
        #print("Test Accuracy: %0.3f \t Test Loss: %0.3f" % (test_acc.data.numpy(), test_loss.data.numpy()))

        # return accuracies[-1], val_accuracies[-1], test_acc, losses[-1], val_losses[-1], test_loss
        return val_accuracies[-1]


test = False
if test:
    # test_string = get_function_from_LSTM

    params = {
        "num_epochs": 1000,
        "opt": "Adam",
        "lr": 0.01
    }
    data_set = "MNIST"
    train_m = Train_model(params)
    layers = []

    # Generate
    if data_set is "MOONS":
        test_string = ('10', 'ReLU', '5', 'Sigmoid', '6', 'ReLU')
        train_m.moon_data(1000, 0.2)
        network = Net_MOONS(string=test_string, in_features=2, num_classes=2, layers=layers)
    if data_set is "MNIST":
        # type of layer, number of neurons, kernel size, activation functions,
        test_string = ('10', 'ReLU', '5', 'Sigmoid', '6', 'ReLU')
        #test_string = ('Conv', '10', '5', 'ReLU', 'Linear', '6', '5', 'ReLU',)
        train_m.mnist_data(10)
        # network = Net_MNIST(string=test_string, in_features=784, num_classes=10, layers=layers)
        network = Net_MNIST(string=test_string, in_features=784, num_classes=10, layers=layers)

    # Defining Network
    net = nn.Sequential(*layers)
    #print(net)

    # Set get_variables used to train neural network
    # If we do not wish to use batches, set batch_size equals to the length
    # of the dataset

    train_batch_size = len(train_m.X_train)
    val_batch_size = len(train_m.X_val)

    plot = False

    val_accuracy = train_m.train(net, train_batch_size, val_batch_size, plot)
