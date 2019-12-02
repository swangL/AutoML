import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets
import math
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from helpers import get_variable

torch.manual_seed(0)

def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    ys = torch.argmax(ys,dim=-1)
    ts = torch.argmax(ts,dim=-1)
    return torch.mean(torch.eq(ys,ts).type(torch.FloatTensor)).cpu().data.numpy()

def conv_accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    ys = torch.argmax(ys,dim=-1)
    return torch.mean(torch.eq(ys,ts).type(torch.FloatTensor)).cpu().data.numpy()

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


def compute_conv_dim(dim_size, kernel_size, padding):
    return int(((dim_size - kernel_size + 2 * padding) / 1) + 1)

class Flatten(nn.Module):
    def forward(self, input):
        batch_size = len(input)
        return input.view(batch_size, -1)


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

# Network
class Net_CONV(nn.Module):

    # Constructor to build network
    def __init__(self, string, img_size, in_channels, num_classes, layers):

        # image = (28,28)
        image = (img_size, img_size)
        padding = 1
        stride = 1

        # Inherit from parent constructor
        super(Net_CONV, self).__init__()

        channels = in_channels

        # Break down string sent from Controller
        # and add layers in network based on this
        counter = 0

        for s in string:

            # If element in string is not a number (i.e. an activation function)
            if s is 'ReLU':
                layers.append(nn.ReLU())
            elif s is 'Tanh':
                layers.append(nn.Tanh())
            elif s is 'Sigmoid':
                layers.append(nn.Sigmoid())
            else:
                if counter % 2 == 0:
                    s_int = int(s)
                else:
                    kernel_size = int(s)
                    layers.append(nn.Conv2d(in_channels=channels, out_channels=s_int, kernel_size=kernel_size, stride=stride, padding=padding))
                    channels = s_int
                    self.conv_out_height = compute_conv_dim(image[0], kernel_size, padding)
                    self.conv_out_width = compute_conv_dim(image[1], kernel_size, padding)
                    image = (self.conv_out_height, self.conv_out_width)
                counter += 1

        # Last layer with output 2 representing the two classes

        if string == []:
            self.conv_out_height = image[0]
            self.conv_out_width = image[1]
        self.in_features = channels * self.conv_out_height * self.conv_out_width

        # With dropout:
        # Epoch 1: TrainAcc: 0.827        ValAcc: 0.922   TrainLoss: 1.634        ValLoss: 1.539
        # Epoch 2: TrainAcc: 0.913        ValAcc: 0.928   TrainLoss: 1.548        ValLoss: 1.533
        # Epoch 3: TrainAcc: 0.921        ValAcc: 0.954   TrainLoss: 1.540        ValLoss: 1.507
        # Epoch 4: TrainAcc: 0.931        ValAcc: 0.953   TrainLoss: 1.530        ValLoss: 1.508
        # Epoch 5: TrainAcc: 0.922        ValAcc: 0.953   TrainLoss: 1.539        ValLoss: 1.508
        # Epoch 6: TrainAcc: 0.927        ValAcc: 0.947   TrainLoss: 1.534        ValLoss: 1.515
        # Epoch 7: TrainAcc: 0.922        ValAcc: 0.946   TrainLoss: 1.539        ValLoss: 1.515

        # Without dropout:
        # Epoch 1: TrainAcc: 0.630        ValAcc: 0.733   TrainLoss: 1.830        ValLoss: 1.727
        # 

        layers.append(Flatten())
        #layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(self.in_features, num_classes))
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
        self.train_loader = 0
        self.val_loader = 0
        # self.test_loader = 0

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
        self.y_train = get_variable(torch.from_numpy(onehot(self.y_train,2)).float())

        self.X_val = get_variable(torch.from_numpy(self.X_val))
        self.y_val = get_variable(torch.from_numpy(onehot(self.y_val,2)).float())

        self.X_test = get_variable(torch.from_numpy(self.X_test))
        self.y_test = get_variable(torch.from_numpy(onehot(self.y_test,2)).float())

    def mnist_data(self, num_classes):

        # Import dataset
        data = np.load('mnist.npz')

        # Define interval used to split data into
        # train, val and test
        interval_1 = 8000
        interval_2 = 500

        # Define train, validation, and test sets
        self.X_train = data['X_train'][:interval_1].astype('float32')
        self.X_val = data['X_valid'][interval_1:interval_1+interval_2].astype('float32')
        self.X_test = data['X_test'][interval_1+interval_2:].astype('float32')

        # and labels
        self.y_train = data['y_train'][:interval_1].astype('int32')
        self.y_val = data['y_valid'][interval_1:interval_1+interval_2].astype('int32')
        self.y_test = data['y_test'][interval_1+interval_2:].astype('int32')

        self.X_train = get_variable(torch.from_numpy(self.X_train))
        self.y_train = get_variable(torch.from_numpy(onehot(self.y_train,num_classes))).float()

        self.X_val = get_variable(torch.from_numpy(self.X_val))
        self.y_val = get_variable(torch.from_numpy(onehot(self.y_val,num_classes))).float()

        self.X_test = get_variable(torch.from_numpy(self.X_test))
        self.y_test = get_variable(torch.from_numpy(onehot(self.y_test,num_classes))).float()

        print(len(self.X_train), len(self.X_val), len(self.X_test))

    def conv_data(self, data_set_name, batch_size_train, batch_size_val):
        
        if data_set_name == 'FASHION':
            train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            val_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        elif data_set_name == 'MNIST':
            train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            val_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        elif data_set_name == 'CIFAR':
            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
            val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

        # val_set, test_set = torch.utils.data.random_split(test_set, [int(0.9 * len(test_set)), int(0.1 * len(test_set))])

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)
        # self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_val, shuffle=True)

        print("Training dataset size: ", len(train_set))
        print("Validation dataset size: ", len(val_set))



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

        accuracies, losses, val_accuracies, val_losses = [], [], [], []

        train_loader = math.ceil(len(self.X_train)/train_batch_size)
        val_loader = math.ceil(len(self.X_val)/val_batch_size)

        # Variables used for EarlyStopping
        early_stop = True
        es_old_val, es_new_val, counter = 0, 0, 0
        es_range = 0.001
        es_limit = 30

        for e in range(self.params["num_epochs"]):
            
            batch_accuracies, batch_losses, batch_val_accuracies, batch_val_losses = [], [], [], []

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
                
                batch_accuracies.append(acc)
                batch_losses.append(loss.cpu().data.numpy())
                
                # accuracies.append(acc)
                # losses.append(loss.cpu().data.numpy())

            # --------------- validate the model --------------- #
            for batch in range(val_loader):

                if batch == (val_loader - 1):
                    val_slce = slice(batch * val_batch_size, -1)
                else:
                    val_slce = slice(batch * val_batch_size, (batch + 1) * val_batch_size)

                val_preds = net(self.X_val[val_slce])

                val_loss = cross_entropy(val_preds, self.y_val[val_slce])
                val_acc = accuracy(val_preds, self.y_val[val_slce])
                # val_losses.append(val_loss.cpu().data.numpy())
                # val_accuracies.append(val_acc)
                batch_val_accuracies.append(val_acc)
                batch_val_losses.append(val_loss.cpu().data.numpy())

            # Accuracy for each episode (a mean of the accuracies for the batches)
            accuracies.append(np.mean(batch_accuracies))
            val_accuracies.append(np.mean(batch_val_accuracies))
            losses.append(np.mean(batch_losses))
            val_losses.append(np.mean(batch_val_losses))

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
            print("Epoch %i: "
                "Train Accuracy: %0.3f"
                "\tVal Accuracy: %0.3f"
                "\tTrain Loss: %0.3f"
                "\tVal Loss: %0.3f"
                % (e, accuracies[-1], val_accuracies[-1], losses[-1], val_losses[-1]))


        # --------------- test the model --------------- #
        test_preds = net(self.X_test)
        test_loss = cross_entropy(test_preds, self.y_test)
        test_acc = accuracy(test_preds, self.y_test)

        #print("Test Accuracy: %0.3f \t Test Loss: %0.3f" % (test_acc.data.numpy(), test_loss.data.numpy()))

        # return accuracies[-1], val_accuracies[-1], test_acc, losses[-1], val_losses[-1], test_loss
        return val_accuracies[-1]

    def train_conv(self, net, plot):
        if self.params["opt"] is "Adam":
            optimizer = optim.Adam(net.parameters(), lr=self.params["lr"])

        elif self.params["opt"] is "SGD":
            optimizer = optim.SGD(net.parameters(), lr=self.params["lr"])

        criterion = nn.CrossEntropyLoss()
        accuracies, losses, val_accuracies, val_losses, test_accuracies, test_losses = [], [], [], [], [], []
        # plot_accuracies, plot_losses, plot_val_accuracies, plot_val_losses = [], [], [], []

        # Variables used for EarlyStopping
        early_stop = True
        es_old_val, es_new_val, counter = 0, 0, 0
        es_range = 0.001
        es_limit = 30

        for e in range(self.params["num_epochs"]):

            net.train()
            batch_accuracies, batch_losses, batch_val_accuracies, batch_val_losses = [], [], [], []

            # --------------- train the model --------------- #
            for itr, (image_train, label_train) in enumerate(self.train_loader):
                image_train, label_train = get_variable(image_train), get_variable(label_train)
                optimizer.zero_grad()
                preds = net(image_train)
                loss = criterion(preds, label_train)
                loss.backward()
                optimizer.step()
                acc = conv_accuracy(preds, label_train)
                # accuracies.append(acc)
                # losses.append(loss.data.numpy())
                batch_accuracies.append(acc)
                batch_losses.append(loss.cpu().data.numpy())

            net.eval()
            # --------------- validate the model --------------- #
            for itr, (image_val, label_val) in enumerate(self.val_loader):
                image_val, label_val = get_variable(image_val), get_variable(label_val)
                val_preds = net(image_val)
                val_loss = criterion(val_preds, label_val)
                val_acc = conv_accuracy(val_preds, label_val)
                # val_losses.append(val_loss.data.numpy())
                # val_accuracies.append(val_acc)
                batch_val_accuracies.append(val_acc)
                batch_val_losses.append(val_loss.cpu().data.numpy())

            # Accuracy for each episode (a mean of the accuracies for the batches)
            accuracies.append(np.mean(batch_accuracies))
            val_accuracies.append(np.mean(batch_val_accuracies))
            losses.append(np.mean(batch_losses))
            val_losses.append(np.mean(batch_val_losses))

            if early_stop:
                # EarlyStopping
                if e == 0:
                    es_old_val = float(val_accuracies[-1])
                else:
                    es_new_val = float(val_accuracies[-1])

                    if abs(es_old_val - es_new_val) <= es_range:
                        counter += 1
                        if counter == es_limit:
                            break
                    else:
                        counter = 0
                        es_old_val = float(val_accuracies[-1])


            #if e % 10 == 0:
            print("Epoch %i: "
            "TrainAcc: %0.3f"
            "\tValAcc: %0.3f"  
            "\tTrainLoss: %0.3f" 
            "\tValLoss: %0.3f" 
            % (e+1, accuracies[-1], val_accuracies[-1], losses[-1], val_losses[-1]), flush=False)

            '''
            plot_accuracies.append(accuracies[-1])
            plot_val_accuracies.append(val_accuracies[-1])
            plot_losses.append(losses[-1])
            plot_val_losses.append(val_losses[-1])
            '''

        '''
        # --------------- test the model --------------- #
        for itr, (image_test, label_test) in enumerate(self.test_loader):

            image_test, label_test = Variable(image_test), Variable(label_test)

            test_preds = net(image_test)
            test_loss = criterion(test_preds, label_test)
            test_acc = conv_accuracy(test_preds, label_test)
            test_losses.append(test_loss.detach().numpy())
            test_accuracies.append(test_acc)
        '''

        # Use first to take last batch for test, use last to take average of batches
        #print("Test Accuracy: %0.3f \t Test Loss: %0.3f" % (test_accuracies[-1], test_losses[-1]))
        # print("Test Accuracy: %0.3f \t Test Loss: %0.3f" % (np.mean(test_accuracies), np.mean(test_losses)))

        if plot:
            # self.plotter(plot_accuracies, plot_losses, plot_val_accuracies, plot_val_losses)
            self.plotter(accuracies, losses, val_accuracies, val_losses)

        # return accuracies[-1], val_accuracies[-1], test_acc, losses[-1], val_losses[-1], test_loss
        # return val_accuracies[-1]
        return val_accuracies[-1]


test = False
if test:
    # test_string = get_function_from_LSTM

    params = {
        "num_epochs": 10,
        "opt": "SGD",
        "lr": 0.01
    }

    data_set = "CONV"
    train_m = Train_model(params)
    layers = []
    # Set get_variables used to train neural network
    # If we do not wish to use batches, set batch_size equals to the length
    # of the dataset

    plot = True

    # Generate
    if data_set is "MOONS":
        test_string = ('10', 'ReLU', '5', 'Sigmoid', '6', 'ReLU')
        train_m.moon_data(1000, 0.2)
        network = Net_MOONS(string=test_string, in_features=2, num_classes=2, layers=layers)
        # Defining Network
        net = nn.Sequential(*layers)
        print(net)

        train_batch_size = len(train_m.X_train)
        val_batch_size = len(train_m.X_val)

        val_accuracy = train_m.train(net, train_batch_size, val_batch_size, plot)

    if data_set is "MNIST":
        # type of layer, number of neurons, kernel size, activation functions,
        test_string = ()
        #test_string = ('Conv', '10', '5', 'ReLU', 'Linear', '6', '5', 'ReLU',)
        train_m.mnist_data(10)
        # network = Net_MNIST(string=test_string, in_features=784, num_classes=10, layers=layers)
        network = Net_MNIST(string=test_string, in_features=784, num_classes=10, layers=layers)
        # Defining Network
        net = nn.Sequential(*layers)
        print(net)

        train_batch_size = len(train_m.X_train)
        val_batch_size = len(train_m.X_val)

        val_accuracy = train_m.train(net, train_batch_size, val_batch_size, plot)

    if data_set is "CONV":

        # this batch size is for all training and validation, dont know if its fine
        batch_size_train = 64
        batch_size_val = 32
        img_size = 28
        test_string = ['6','3', 'ReLU', '6','3', 'ReLU']
 
        train_m.conv_data(data_set_name="FASHION", batch_size_train=batch_size_train, batch_size_val=batch_size_val)

        network = Net_CONV(string=test_string, img_size=img_size, in_channels=1, num_classes=10, layers=layers)

        net = nn.Sequential(*layers)
        print(net)

        val_accuracy = train_m.train_conv(net, plot)

'''
import pickle

with open('/Users/mariana/AutoML/runs/151024-01122019/rewards_losses.pkl', 'rb') as f:
    data = pickle.load(f)


acc = data[0]
loss = data[1]

acc = np.array(acc)
loss = np.array(loss)

print(acc)
print(loss)

plt.figure()
plt.plot(range(len(loss)), loss, 'r', label='Train Loss')
plt.legend()
plt.xlabel('Updates')
plt.ylabel('Loss')

plt.figure()
plt.plot(range(len(acc)), acc, 'r', label='Train Acc')
plt.legend()
plt.xlabel('Updates')
plt.ylabel('Accuracy')
plt.show()

'''

# FASHIONMNIST results
# For test_string = ['6','3', 'ReLU', '6','3', 'ReLU']
'''
Epoch 1: TrainAcc: 0.827        ValAcc: 0.922   TrainLoss: 1.634        ValLoss: 1.539
Epoch 2: TrainAcc: 0.913        ValAcc: 0.928   TrainLoss: 1.548        ValLoss: 1.533
Epoch 3: TrainAcc: 0.921        ValAcc: 0.954   TrainLoss: 1.540        ValLoss: 1.507
Epoch 4: TrainAcc: 0.931        ValAcc: 0.953   TrainLoss: 1.530        ValLoss: 1.508
Epoch 5: TrainAcc: 0.922        ValAcc: 0.953   TrainLoss: 1.539        ValLoss: 1.508
Epoch 6: TrainAcc: 0.927        ValAcc: 0.947   TrainLoss: 1.534        ValLoss: 1.515
Epoch 7: TrainAcc: 0.922        ValAcc: 0.946   TrainLoss: 1.539        ValLoss: 1.515
Epoch 8: TrainAcc: 0.923        ValAcc: 0.942   TrainLoss: 1.538        ValLoss: 1.519
Epoch 9: TrainAcc: 0.929        ValAcc: 0.955   TrainLoss: 1.532        ValLoss: 1.506
Epoch 10: TrainAcc: 0.925       ValAcc: 0.942   TrainLoss: 1.536        ValLoss: 1.519
'''
# For test_string = [] with dropout
'''
Epoch 1: TrainAcc: 0.844        ValAcc: 0.908   TrainLoss: 1.632        ValLoss: 1.558
Epoch 2: TrainAcc: 0.870        ValAcc: 0.910   TrainLoss: 1.596        ValLoss: 1.554
Epoch 3: TrainAcc: 0.874        ValAcc: 0.913   TrainLoss: 1.589        ValLoss: 1.549
Epoch 4: TrainAcc: 0.874        ValAcc: 0.915   TrainLoss: 1.589        ValLoss: 1.547
Epoch 5: TrainAcc: 0.876        ValAcc: 0.912   TrainLoss: 1.586        ValLoss: 1.549
Epoch 6: TrainAcc: 0.876        ValAcc: 0.915   TrainLoss: 1.586        ValLoss: 1.547
Epoch 7: TrainAcc: 0.878        ValAcc: 0.915   TrainLoss: 1.584        ValLoss: 1.547
Epoch 8: TrainAcc: 0.879        ValAcc: 0.915   TrainLoss: 1.583        ValLoss: 1.546
Epoch 9: TrainAcc: 0.878        ValAcc: 0.916   TrainLoss: 1.583        ValLoss: 1.545
Epoch 10: TrainAcc: 0.879       ValAcc: 0.917   TrainLoss: 1.583        ValLoss: 1.543
'''
# For test_string = [] without dropout
'''
Epoch 1: TrainAcc: 0.890        ValAcc: 0.915   TrainLoss: 1.585        ValLoss: 1.550
Epoch 2: TrainAcc: 0.919        ValAcc: 0.921   TrainLoss: 1.547        ValLoss: 1.543
Epoch 3: TrainAcc: 0.924        ValAcc: 0.919   TrainLoss: 1.540        ValLoss: 1.544
Epoch 4: TrainAcc: 0.928        ValAcc: 0.916   TrainLoss: 1.536        ValLoss: 1.547
Epoch 5: TrainAcc: 0.930        ValAcc: 0.921   TrainLoss: 1.534        ValLoss: 1.541
Epoch 6: TrainAcc: 0.932        ValAcc: 0.921   TrainLoss: 1.531        ValLoss: 1.541
Epoch 7: TrainAcc: 0.934        ValAcc: 0.923   TrainLoss: 1.530        ValLoss: 1.539
Epoch 8: TrainAcc: 0.936        ValAcc: 0.928   TrainLoss: 1.528        ValLoss: 1.534
Epoch 9: TrainAcc: 0.936        ValAcc: 0.921   TrainLoss: 1.527        ValLoss: 1.541
Epoch 10: TrainAcc: 0.937       ValAcc: 0.925   TrainLoss: 1.526        ValLoss: 1.537
'''