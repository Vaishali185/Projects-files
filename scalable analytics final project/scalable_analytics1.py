#Name: Vaishali Pawar.
#student number: 0559388
# VUB

#mpiexec -n 4 python -m mpi4py scalable_analytics_project.py
#mpiexec -n 2 python -m mpi4py scalable_analytics2.py
import numpy as np 
import pandas as pd
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Process
from mpi4py import MPI             # MPI functions in Python
import PIL.Image  
import sys
from collections import OrderedDict


BATCH = 10
EPOCHS = 2
LR = 0.01
IM_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_root = "C:/Users/kaila/3rd semester/information_retrival/scalable_analytics project/mini_herbarium/mini_herbarium/"
#dataset_root = "C:/Users/kaila/3rd semester/information_retrival/scalable_analytics project/kaggle_herbarium_1gb/kaggle_herbarium_1gb/"
train_images_dir = dataset_root + "train/"
#test_images_dir = dataset_root + "test/"
train_images_dir

with open(train_images_dir + 'metadata.json', "r", encoding="ISO-8859-1") as file:
     train = json.load(file)  
        
print("train dict", train.keys())

all_filenames=[]
for image in train["images"]:
    all_filenames.append(image["file_name"])
all_filenames


train_img = pd.DataFrame(train['images'])
train_ann = pd.DataFrame(train['annotations']).drop(columns='image_id')
train_df = train_img.merge(train_ann, on='id')
print(len(train_df))
train_df


NUM_CL = len(train_df['category_id'].value_counts())

X_Train, Y_Train = train_df['file_name'].values, train_df['category_id'].values

Transform = transforms.Compose(
    [transforms.RandomRotation(10),      # rotate +/- 10 degrees
     transforms.RandomHorizontalFlip(),  # reverse 50% of images
     transforms.Resize(224),             # resize shortest side to 224 pixels
     transforms.CenterCrop(224),  
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))

        if "train" in self.dir:             
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

trainset = GetData(train_images_dir, X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)

model = torchvision.models.resnet34()
model.fc = nn.Linear(512, NUM_CL, bias=True)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    tr_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):    
        pred = model(images)
              
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.detach().item()
        print('Epoch: %d | Minibatch: %d | Loss: %.4f'%(epoch,i, tr_loss))

#Defining a Convolutional Neural Network

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 44944)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train the network

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.detach().item()
        print('Epoch: %d | Minibatch: %d | Loss: %.4f'%(epoch,i, running_loss))
        #if i % 2000 == 1999:    # print every 2000 mini-batches
            #print('[%d, %5d] loss: %.3f' %
                  #(epoch + 1, i + 1, running_loss / 2000))
            #running_loss = 0.0

print('Finished Training')


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() # number workers

filenames_to_scatter = None

if rank == 0:
     
     
    # Split them among <size> workers (node 0 included)
    elements_per_worker = len(all_filenames) // size
    filenames_to_scatter = []
    
    for i in range(size):
        # fr and to: define a range of filenames to give to the i-th worker
        fr = i * elements_per_worker
        to = fr + elements_per_worker
        
        if i == size-1:
            # The last worker may have more images to process if <size> does not divide len(all_filenames)
            to = len(all_filenames)
        
        filenames_to_scatter.append(all_filenames[fr:to])
        
my_filenames = comm.scatter(filenames_to_scatter, root=0)
print('I am Node', rank, 'and I got', len(my_filenames), 'images to process')

for t in range(EPOCHS):
                #model = MNISTNetwork()
         
        for i, filename in enumerate(my_filenames):
            image = PIL.Image.open(train_images_dir + filename)        # Open the image
            image = image.resize((500, 500))                  # Resize the image to 500x500 pixels to be faster
            image = np.array(image)                           # Cast to a height,width,3 numpy array of uint8 values



        print('I am Node', rank, 'and I have opened all my images')

        model = Net()
        
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),lr=LR)
for k, (x, y) in enumerate(trainloader):
    opt.zero_grad()
    loss_fn(model(x), y).backward()
    opt.step()
m=comm.gather(model.state_dict(), root=0)
            
            
            
 
