import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 
import cv2 as cv
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# References: 
# https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212 
# https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-2-max-pooling/ 
# https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model  
# https://stackoverflow.com/questions/54098364/understanding-channel-in-convolution-neural-network-cnn-input-shape-and-output 
# https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/ 
# https://deepai.org/machine-learning-glossary-and-terms/relu 
# https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7 

#Beginning of Notes
#
#
#
#Relu = Rectified Linear Unit
#1.)Important for convolutional process because
# it decreases the linearity of data. Images are nonlinear
# so it is important for the CNN to have this when doing image processing
# because there may be linearity imposed on the images when put through convolution operations
# Source: https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-1b-relu-layer

#Pooling: Taking different values from kernels in image to get rid of any extra information (Also called downsampling)
#This also allows for object detector to ignore spatial  variance
#An example would be one feature is value 5 while the pixel next to it is 1
#Max pooling would take this large number in kernal and group the data around that local maximum

#***** Impoortant****
#((W − F + 2P) / S) + 1 needs to be an integer
# P is zero padding
#W is the input size of images
#F is receptive field size 

#S is stride usually 1 or 2
#How much convolution kernel skips pixels. 2 would mean 
#convolution will be applied every 2 pixels. producing smaller output volume compared to 1.
#
#
#
#End of Notes

# dataset = ImageFolder('path/to/dataset/root')


object_types = ['book','plant','chair']
save_dir = 'C:/Users/dylan/Documents/Capstone/Home_Object_Detection_Dylan/Models/'

# define generic Transform each photo, resize, flip the image/rotate image
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

#need to calculate mean and std for normalize function, creating lists for means and standard deviations
means = []
stds = []

for object_type in object_types:
    dataset = datasets.ImageFolder('C:/Users/dylan/Documents/Capstone/Home_Object_Detection_Dylan/Data/' + object_type, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=1)
    data = next(iter(loader))
    means.append(data[0].mean().item())
    stds.append(data[0].std().item())
# Print the means and standard deviations for each object type
for i, object_type in enumerate(object_types):
    print('Object type:', object_type)
    print('Mean:', means[i])
    print('Standard deviation:', stds[i]) 

#count number of different objects that need to be detected
num_classes = len(dataset.classes)

#define new normalize function for object 
normalize_transform = transforms.Normalize(mean=means, std=stds)

#loop entire training code for each different object type 

for object_type in object_types:
    print('Training model:'+ str(object_type))
    #subdirectory for model weights
    sub_dir = os.path.join(save_dir, object_type)
    os.makedirs(sub_dir, exist_ok=True) 

    #load dataset with normalized transform and transform from before 
    dataset = datasets.ImageFolder('C:/Users/dylan/Documents/Capstone/Home_Object_Detection_Dylan/Data/' +object_type, transform=transforms.Compose([
        normalize_transform,
        transform 
    ]))


    #Split dataset into train and test sets (common practice for ML)
    #Will want to check if object labels are correct with test set later with model.eval()
    #80% of dataset will be used to train while 20% is used to test the model
    train_size = int(0.9*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #random split: takes data to be split as argu 1 and argu 2 is the size calculated above from int and len for each 
    
    #defining data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)




# Defining the convolutional neural network architecture 
#optimized to be with  512x512 pixels. Need to change parameters if using different image sizes
#code above resizes to 512x512 for now anyways so it should work with any folder of photos
#3 convolutional layers total, can use more but I would need more processing power

    class Net(nn.Module):
        def __init__(self, num_classes):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            #
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            #
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            #
            self.fc1 = nn.Linear(64 * 64 * 64, 500)
            self.relu4 = nn.ReLU()
            self.fc2 = nn.Linear(500, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool3(x)
            x = x.view(-1, 64 * 64 * 64)
            x = self.fc1(x)
            x = self.relu4(x)
            x = self.fc2(x)
            return x

    model = Net()


    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    #Note: that too many epochs will lead to overfitting, (bias in data and the model is only good for the data that I used. So any other images
    # will not work properly, only images in the dataset I trained the model in)
    #The amount of epochs is the number of times the CNN will go through the training data

    #for epoch in range(2)... include

    #instantiate the model