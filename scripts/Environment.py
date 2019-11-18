import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets
import math

def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(torch.max(ys, 1)[1], torch.max(ts, 1)[1])
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())

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
class Net(nn.Module):
    
    # Constructor to build network
    def __init__(self, string, in_features, layers):
        
        # Inherit from parent constructor
        super(Net, self).__init__()
        
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
        layers.append(nn.Linear(num_input, 2))
        layers.append(nn.Softmax())
        
        print('layers', layers)

class Train_model():

    def __init__(self):
        self.X_train = 0
        self.y_train = 0
        self.X_val = 0
        self.y_val = 0
        self.X_test = 0
        self.y_test = 0
        
    def generate_data(self, num_samples):
        # num_samples should be divisable by 5

        # Import dataset
        X, y = make_moons(n_samples=num_samples, noise=0.1)

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

        self.X_train = Variable(torch.from_numpy(self.X_train))
        self.y_train = Variable(torch.from_numpy(onehot(self.y_train,2))).float()

        self.X_val = Variable(torch.from_numpy(self.X_val))
        self.y_val = Variable(torch.from_numpy(onehot(self.y_val,2))).float()

        self.X_test = Variable(torch.from_numpy(self.X_test))
        self.y_test = Variable(torch.from_numpy(onehot(self.y_test,2))).float()

    def plotter(self, accuracies, losses, val_accuracies, val_losses):
        plt.figure()
        epoch = np.arange(len(losses))
        plt.plot(epoch, losses, 'r', label='Train Loss')
        plt.plot(epoch, val_losses, 'b', label='Val Loss')
        plt.legend()
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.show()

        plt.figure()
        plt.plot(epoch, accuracies, 'r', label='Train Acc')
        plt.plot(epoch, val_accuracies, 'b', label='Val Acc')
        plt.legend()
        plt.xlabel('Updates')
        plt.ylabel('Accuracy')
        plt.show()

    def train(self):

        if opt is "Adam":
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        elif opt is "SGD":
            optimizer = optim.SGD(net.parameters(), lr=learning_rate)

        accuracies = []
        losses = []

        val_accuracies = []
        val_losses = []

        for e in range(num_epochs):

            train_loader = math.ceil(len(self.X_train)/train_batch_size)
            val_loader = math.ceil(len(self.X_val)/val_batch_size)

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
                val_accuracies.append(val_acc.data.numpy())

            if e % 10 == 0:
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

        print("Test Accuracy: %0.3f \t Test Loss: %0.3f" % (test_acc.data.numpy(), test_loss.data.numpy()))

        return accuracies[-1], val_accuracies[-1], test_acc, losses[-1], val_losses[-1], test_loss

# test_string = get_function_from_LSTM
test_string = ('10', 'ReLU', '5', 'Sigmoid', '6', 'ReLU')

# Generate 
train_m = Train_model()
train_m.generate_data(300)

# Defining Network
layers = []
network = Net(test_string, 2, layers)
net = nn.Sequential(*layers)
print(net)

# Set variables used to train neural network
# If we do not wish to use batches, set batch_size equals to the length
# of the dataset 
num_epochs = 200
train_batch_size = 50
val_batch_size = 50
opt = "Adam"
learning_rate = 0.01

train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = train_m.train()
