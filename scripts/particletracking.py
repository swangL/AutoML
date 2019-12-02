import os
import re
from numpy import genfromtxt
import numpy as np
import cv2
import torch
from helpers import get_variable
txtfolder = "xyzTrainingTXTFloat"
imagefolder = "xyzTrainingImagesFloat"
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
print(train_data.shape)

print(len(train_target))
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
val_data = get_variable(val_data)
val_target = get_variable(torch.tensor(val_target))
train_data = get_variable(train_data)
train_target = get_variable(train_target)

print(val_data.shape)
print(len(val_target))