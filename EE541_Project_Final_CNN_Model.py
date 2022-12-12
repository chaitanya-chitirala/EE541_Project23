#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

file = 'ASLdatasetGroup23.hdf5'
with h5py.File(file, 'r') as rd:
    x_train = np.array(rd['x_train'])
    YT = np.array(rd['y_train'])
    x_test = np.array(rd['x_test'])
    YTE = np.array(rd['y_test'])

XT = np.copy(x_train)
XT = XT.astype('float32')/255

XTE = np.copy(x_test)
XTE = XTE.astype('float32')/255

class_enum = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

print(XT.shape)
print(y_train.shape)
print(XTE.shape)
print(y_test.shape)

train_transforms = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomRotation(12),
                                      transforms.ColorJitter(brightness = 0.5, contrast = 0.3, saturation = 0.3, hue = 0.3),
                                      transforms.ToTensor()])

class ASLSetTRAIN(torch.utils.data.Dataset):
    def __init__(self, images, labels, forms):
        'Initialization'
        self.labels = labels
        self.images = images
        self.tform = forms
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = self.images[index]
        lab = self.labels[index]
        X = self.tform(torch.from_numpy(img))
        Y = lab
        
        return X, Y

class ASLSetTEST(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        'Initialization'
        self.labels = labels
        self.images = images
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = self.images[index]
        lab = self.labels[index]
        X = torch.from_numpy(img)
        Y = lab
        
        return X, Y

train_set = ASLSetTRAIN(XT, YT, train_transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100, shuffle = True)

test_set = ASLSetTEST(XTE, YTE)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 100, shuffle = False)

loss_func = nn.CrossEntropyLoss()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)    # 1st convolutional layer 
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)   # 2nd convolutional layer
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)  # 3rd convolutional layer
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1) # 4th convolutional layer
        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1) # 5th convolutional layer

        self.batchnorm1 =  nn.BatchNorm2d(32)
        self.batchnorm2 =  nn.BatchNorm2d(64)
        self.batchnorm3 =  nn.BatchNorm2d(128)
        self.batchnorm4 =  nn.BatchNorm2d(256)
        self.batchnorm5 =  nn.BatchNorm2d(512)

        # linear transformation operation: y = Wx + b
        self.fc1 = nn.Linear(4608, 3000) # 1st fully connected layer
        self.fc2 = nn.Linear(3000, 500)  # 2nd fully connected layer
        self.fc3 = nn.Linear(500,29)     # 3rd fully connected layer
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout = torch.nn.Dropout(p = 0.2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.batchnorm3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.batchnorm4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.batchnorm5(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

model1 = Net()
optimizer1 = torch.optim.SGD(model1.parameters(), lr = 0.05)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model1 = model1.to(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

train_loss_arr = []
train_acc_arr = []
test_loss_arr = []
test_acc_arr = []
num_epochs = 20
for epoch in range(num_epochs):
    train_correct = 0
    test_correct = 0
    train_loss = 0
    test_loss = 0
    model1.train()

    for images, labels in train_loader:
        labels = labels.to(device)
        images = images.to(device)

        #Forward Prop
        outputs = model1(images)
        loss = loss_func(outputs,labels)
        
        #Back Prop
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        predictions = torch.max(outputs.cpu(),1)[1]
        train_correct += (predictions == labels.cpu()).sum().numpy()
        train_loss += loss.cpu().data

    model1.eval()
    for images, labels in test_loader:
        images = images.to(device)

        #Forward Prop
        outputs = model1(images).cpu()
        loss = loss_func(outputs, labels)
        predictions = torch.max(outputs, 1)[1]
        test_correct += (predictions == labels).sum().numpy()
        test_loss += loss.cpu().data

    train_loss_arr.append(train_loss/len(train_loader.dataset))
    train_acc_arr.append(train_correct/len(train_loader.dataset))
    test_loss_arr.append(test_loss/len(test_loader.dataset))
    test_acc_arr.append(test_correct/len(test_loader.dataset))

    print(f'Epoch: {epoch+1:02d}: Train Loss: {(train_loss/len(train_loader.dataset)):.4f}, Train Accuracy: {(100*train_correct/len(train_loader.dataset)):2.3f}%')
    print("           ", f'Test Loss: {(test_loss/len(test_loader.dataset)):.4f}, Test Accuracy: {(100*test_correct/len(test_loader.dataset)):2.3f}%')
print(f'Final Train Accuracy: {100*train_acc_arr[-1]:2.3f}, Final Test Accuracy: {100*test_acc_arr[-1]:2.3f}')

a = plt.figure(1)
plt.plot(range(1, num_epochs + 1), 100*np.array(train_acc_arr), label = "Training Accuracy")
plt.plot(range(1, num_epochs + 1), 100*np.array(test_acc_arr), label = "Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy")
plt.legend()
plt.show()

b = plt.figure(2)
plt.plot(range(1, num_epochs + 1), 20*np.log10(np.array(train_loss_arr)), label = "Training Loss")
plt.plot(range(1, num_epochs + 1), 20*np.log10(np.array(test_loss_arr)), label = "Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (dB)")
plt.title("Loss")
plt.legend()
plt.show()

