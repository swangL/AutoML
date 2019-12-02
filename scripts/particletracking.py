import os
import re
from numpy import genfromtxt
import numpy as np
import cv2
import torch
from helpers import get_variable
from Environment import *
from torchvision import transforms


txtfolder = "xyzTrainingTXTFloat"
imagefolder = "xyzTrainingImagesFloat"

def normalize(x): return x/255

def create_training_data(imagefolder):
    training_data=[] # Empty list to fill up with training data from image folder
    pics=os.listdir(imagefolder) #Take files in corresponding folder, which should only be pictures!
    pics=sort_nice(pics)

    for img in range(0, len(pics)): #Create training data
        try:
            img_array = cv2.imread(os.path.join(imagefolder,pics[img]),cv2.IMREAD_GRAYSCALE) #adding pic name to folder directory
            new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
            training_data.append(new_array)
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
#NB: Image dimensions based on setting file used to generate data files
IMG_SIZE=200
IMG_WIDTH=45
IMG_HEIGHT=45

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
# train_data = train_data.squeeze(1)
train_data = torch.flatten(train_data, start_dim=1)

validationfolder = "xyzPredictionImagesFloat"
val_data = create_training_data(validationfolder)
val_data = torch.flatten(val_data, start_dim=1)
# val_data = val_data.squeeze(1)



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

# train_target = torch.flatten(train_target, start_dim=0)
# val_target = torch.flatten(val_target, start_dim=0)

print(val_data.shape)
print(len(val_target))
print(train_data.shape)
print(len(train_target))

transform = transforms.Compose([transforms.ToTensor,
                                transforms.Normalize([0.5], [0.5])])

val_data = normalize(val_data)
val_target = normalize(val_target)
train_target = normalize(train_target)
train_data = normalize(train_data)

'''
val_data.transforms.Normalize([0.5], [0.5])

val_data = transform(val_data).float()
train_data = transform(train_data)
val_target = transform(val_target)
train_target = transform(train_target)

'''


print("val_target.shape:", val_target.shape)

plot = True

params = {
    "num_epochs": 500,
    "opt": "Adam",
    "lr": 0.01
}

layers = []

train_m = Train_model(params)

# type of layer, number of neurons, kernel size, activation functions,
test_string = ('10', 'ReLU', '5', 'Sigmoid', '6', 'ReLU')

network = Net_MNIST(string=test_string, in_features=2025, num_classes=3, layers=layers)
# Defining Network
net = nn.Sequential(*layers)
print(net)

train_batch_size = train_data.shape[0]
val_batch_size = val_data.shape[0]

print(train_batch_size)
print(val_batch_size)

train_m.particle_data(train_data, train_target, val_data, val_target)

val_accuracy = train_m.particle_train(net, train_batch_size, val_batch_size, plot)