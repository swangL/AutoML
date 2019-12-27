import os
import re
from numpy import genfromtxt
import numpy as np
import cv2
import torch
import torch.nn as nn
import helpers
from helpers import get_variable
# from Environment import *
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn import metrics

txtfolder = "xyzTrainingTXTFloat"
imagefolder = "xyzTrainingImagesFloat"

def normalize_Val_Data(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = (x[i]/len(x)) * (-1)
        else:
            x[i] = x[i]/len(x)
    return x

def normalize(x):
    return x/255

def mse_loss(ys, ts):
    return torch.mean((ys-ts)**2)

def create_training_data(imagefolder):
    # NB: Image dimensions based on setting file used to generate data files
    IMG_SIZE = 200
    IMG_WIDTH = 45
    IMG_HEIGHT = 45

    training_data=[] # Empty list to fill up with training data from image folder
    pics=os.listdir(imagefolder) #Take files in corresponding folder, which should only be pictures!
    pics=sort_nice(pics)

    for img in range(0, len(pics)): #Create training data
        try:
            img_array = cv2.imread(os.path.join(imagefolder,pics[img]),cv2.IMREAD_GRAYSCALE) #adding pic name to folder directory
            new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
            training_data.append(new_array.reshape((1, IMG_WIDTH,IMG_HEIGHT)))
        except Exception as e: #pass destroyed files
            pass
    return torch.tensor(training_data)

def sort_nice(liste): #Sorting command, so data files are loaded in order 0,1,2,...,9,10,...,n.
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(liste, key = alphanum_key)

def settingLoad(txtfolder, filename):
    txt = genfromtxt(txtfolder+'/'+filename, delimiter=' ') #Genfromtxt create arrays from data
    return txt

def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    ys = torch.argmax(ys,dim=-1)
    ts = torch.argmax(ts,dim=-1)
    return torch.mean(torch.eq(ys,ts).type(torch.FloatTensor)).cpu().data.numpy()

def r2_score(ys, ts):
    SS_res =  torch.sum(math.pow((ts - ys),2))
    SS_tot = torch.sum(math.pow((ts - math.mean(ts),2)))
    return (1 - (SS_res/(SS_tot + torch.epsilon())))

class Flatten(nn.Module):
    def forward(self, input):
        batch_size = len(input)
        return input.view(batch_size, -1)
# Network
class Net_PARTICLE(nn.Module):

    # Constructor to build network
    def __init__(self, string, in_features, num_classes, layers):

        # Inherit from parent constructor
        super(Net_PARTICLE, self).__init__()

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

class Partivle_Net_CONV(nn.Module):

    # Constructor to build network
    def __init__(self, string, img_size, in_channels, num_classes, layers):

        image = (img_size, img_size)
        padding = 1
        stride = 1

        # Inherit from parent constructor
        super(Partivle_Net_CONV, self).__init__()

        channels = in_channels
        counter = 0

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
            else:
                if counter % 2 == 0:
                    s_int = int(s)
                else:
                    kernel_size = int(s)
                    padding = kernel_size - 1
                    layers.append(nn.Conv2d(in_channels=channels, out_channels=s_int, kernel_size=kernel_size, stride=stride, padding=padding))
                    channels = s_int
                    #image = (self.conv_out_height, self.conv_out_width)
                counter += 1

        self.conv_out_height = image[0]
        self.conv_out_width = image[1]

        if self.conv_out_height == 0:
            print('conv_out_height is zero')

        self.in_features = channels * self.conv_out_height * self.conv_out_width

        layers.append(Flatten())
        # Last layer with output 2 representing the two classes
        layers.append(nn.Linear(self.in_features, num_classes))

class Train_model_particle():

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

    def particle_data(self):
        # Create empty list to save x,y,z-coordinates as matrix
        train_target=[]
        #Read files in txtfolder
        dirCoordinates=os.listdir(txtfolder)
        dirCoordinates=os.listdir(txtfolder)[0:len(dirCoordinates)]
        dirCoordinates=sort_nice(dirCoordinates) #sort so 10 is after 9

        # Create a list of extracted x,y,z-values
        for i in range (0, len(dirCoordinates)):
            train_target.append([settingLoad(txtfolder, dirCoordinates[i])[0],settingLoad(txtfolder, dirCoordinates[i])[1],abs(settingLoad(txtfolder, dirCoordinates[i])[2])])

        train_target=torch.tensor(train_target)
        numImages = len(train_target) #Total number of loaded images
        train_data=create_training_data(imagefolder)
        train_data = torch.flatten(train_data, start_dim=1)

        validationfolder = "xyzPredictionImagesFloat"
        val_data = create_training_data(validationfolder)
        val_data = torch.flatten(val_data, start_dim=1)

        # Create empty list to save x,y,z-coordinates as matrix
        val_target=[]
        val_txtfolder = "xyzPredictionTXTFloat"

        #Read files in txtfolder
        dirCoordinates=os.listdir(val_txtfolder)
        dirCoordinates=os.listdir(val_txtfolder)[0:len(dirCoordinates)]
        dirCoordinates=sort_nice(dirCoordinates) #sort so 10 is after 9
        # Create a list of extracted x,y,z-values
        for i in range (0, len(dirCoordinates)):
            val_target.append([settingLoad(val_txtfolder, dirCoordinates[i])[0],settingLoad(val_txtfolder, dirCoordinates[i])[1],abs(settingLoad(val_txtfolder, dirCoordinates[i])[2])])

        # Load into CUDA
        val_data = get_variable(val_data).float()
        val_target = get_variable(torch.tensor(val_target)).float()
        train_data = get_variable(train_data).float()
        train_target = get_variable(train_target).float()

        transform = transforms.Compose([transforms.ToTensor,
                                        transforms.Normalize([0.5], [0.5])])

        val_data = normalize(val_data)
        val_target = normalize(val_target)
        train_target = normalize(train_target)
        train_data = normalize(train_data)

        train_data, train_target, val_data, val_target

        self.X_train = train_data
        self.X_val = val_data

        self.y_train = train_target
        self.y_val = val_target

    def particle_data_conv(self):
        # Create empty list to save x,y,z-coordinates as matrix
        train_target=[]
        #Read files in txtfolder
        dirCoordinates=os.listdir(txtfolder)
        dirCoordinates=os.listdir(txtfolder)[0:len(dirCoordinates)]
        dirCoordinates=sort_nice(dirCoordinates) #sort so 10 is after 9

        # Create a list of extracted x,y,z-values
        for i in range (0, len(dirCoordinates)):
            train_target.append([settingLoad(txtfolder, dirCoordinates[i])[0],settingLoad(txtfolder, dirCoordinates[i])[1],abs(settingLoad(txtfolder, dirCoordinates[i])[2])])

        train_target=torch.tensor(train_target)
        numImages = len(train_target) #Total number of loaded images
        train_data=create_training_data(imagefolder)

        validationfolder = "xyzPredictionImagesFloat"
        val_data = create_training_data(validationfolder)
        # Create empty list to save x,y,z-coordinates as matrix
        val_target=[]
        val_txtfolder = "xyzPredictionTXTFloat"

        #Read files in txtfolder
        dirCoordinates=os.listdir(val_txtfolder)
        dirCoordinates=os.listdir(val_txtfolder)[0:len(dirCoordinates)]
        dirCoordinates=sort_nice(dirCoordinates) #sort so 10 is after 9
        # Create a list of extracted x,y,z-values
        for i in range (0, len(dirCoordinates)):
            val_target.append([settingLoad(val_txtfolder, dirCoordinates[i])[0],settingLoad(val_txtfolder, dirCoordinates[i])[1],abs(settingLoad(val_txtfolder, dirCoordinates[i])[2])])

        # Load into CUDA
        val_data = get_variable(val_data).float()
        val_target = get_variable(torch.tensor(val_target)).float()
        train_data = get_variable(train_data).float()
        train_target = get_variable(train_target).float()

        transform = transforms.Compose([transforms.ToTensor,
                                        transforms.Normalize([0.5], [0.5])])

        val_data = normalize(val_data)
        val_target = normalize(val_target)
        train_target = normalize(train_target)
        train_data = normalize(train_data)

        train_data, train_target, val_data, val_target

        self.X_train = train_data
        self.X_val = val_data
        print(train_data.shape)
        self.y_train = train_target
        self.y_val = val_target

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

    def particle_train(self,net, plot):

        if self.params["opt"] is "Adam":
            optimizer = optim.Adam(net.parameters(), lr=self.params["lr"])

        elif self.params["opt"] is "SGD":
            optimizer = optim.SGD(net.parameters(), lr=self.params["lr"])

        early_stop = True

        accuracies, losses, val_accuracies, val_losses, r2_scores, val_r2_scores = [], [], [], [], [], []
        train_batch_size = self.X_train.shape[0]
        val_batch_size = self.X_val.shape[0]
        train_loader = math.ceil(len(self.X_train)/train_batch_size)
        val_loader = math.ceil(len(self.X_val)/val_batch_size)

        # Variables used for EarlyStopping
        early_stop = True
        es_old_val, es_new_val, counter = 0, 0, 0
        es_range = 0.001
        es_limit = 30

        for e in range(self.params["num_epochs"]):

            batch_accuracies, batch_losses, batch_val_accuracies, batch_val_losses, batch_r2_scores, batch_val_r2_scores = [], [], [], [], [], []

            # --------------- train the model --------------- #
            for batch in range(train_loader):

                optimizer.zero_grad()

                if batch == (train_loader - 1):
                    slce = slice(batch * train_batch_size, -1)
                else:
                    slce = slice(batch * train_batch_size, (batch + 1) * train_batch_size)

                preds = net(self.X_train[slce])
                loss = mse_loss(preds, self.y_train[slce])
                loss.backward()
                optimizer.step()
                acc = accuracy(preds, self.y_train[slce])
                r2_score = sklearn.metrics.r2_score(self.y_train[slce].cpu().data.numpy(), preds.cpu().data.numpy())
                # r2 = r2_score(preds, self.y_train[slce])
                batch_accuracies.append(acc)
                batch_r2_scores.append(r2_score)
                batch_losses.append(loss.cpu().data.numpy())

            # --------------- validate the model --------------- #
            for batch in range(val_loader):

                if batch == (val_loader - 1):
                    val_slce = slice(batch * val_batch_size, -1)
                else:
                    val_slce = slice(batch * val_batch_size, (batch + 1) * val_batch_size)

                val_preds = net(self.X_val[val_slce])
                val_loss = mse_loss(val_preds, self.y_val[val_slce])
                val_acc = accuracy(val_preds, self.y_val[val_slce])
                val_r2_score = sklearn.metrics.r2_score(self.y_val[val_slce].cpu().data.numpy(), val_preds.cpu().data.numpy())
                # val_r2 = r2_score(val_preds, self.y_val[val_slce])
                batch_val_accuracies.append(val_acc)
                batch_val_r2_scores.append(val_r2_score)
                batch_val_losses.append(val_loss.cpu().data.numpy())

            # Accuracy for each episode (a mean of the accuracies for the batches)
            accuracies.append(np.mean(batch_accuracies))
            val_accuracies.append(np.mean(batch_val_accuracies))
            losses.append(np.mean(batch_losses))
            val_losses.append(np.mean(batch_val_losses))

            r2_scores.append(np.mean(batch_r2_scores))

            for i in range(len(batch_val_r2_scores)):
                if batch_val_r2_scores[i] < 0:
                    batch_val_r2_scores[i] = 0


            # batch_val_r2_scores = normalizeValData(batch_val_r2_scores)

            val_r2_scores.append(np.mean(batch_val_r2_scores))

            # EarlyStopping
            if early_stop:
                if e == 0:
                    es_old_val = float(val_r2_score)
                else:
                    es_new_val = float(val_r2_score)

                    if abs(es_old_val - es_new_val) <= es_range:
                        counter += 1
                        if counter == es_limit:
                            break
                    else:
                        counter = 0
                        es_old_val = float(val_r2_score)

        if plot:
            self.plotter(r2_scores, losses, val_r2_scores, val_losses)
        return val_r2_scores[-1], val_losses[-1]

test = False

if test:
    plot = True

    params = {
        "num_epochs": 500,
        "opt": "Adam",
        "lr": 0.01
    }

    layers = []
    train_m = Train_model_particle(params)
    # type of layer, number of neurons, kernel size, activation functions,
    test_string = ('10', 'ReLU', '5', 'Sigmoid', '6', 'ReLU')
    network = Net_PARTICLE(string=test_string, in_features=2025, num_classes=3, layers=layers)
    # Defining Network
    net = nn.Sequential(*layers)
    print(net)

    train_m.particle_data()
    r2_score = train_m.particle_train(net, plot)
    print(r2_score)
