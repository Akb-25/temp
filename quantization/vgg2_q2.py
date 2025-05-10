import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import csv
from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG
import warnings
from random import SystemRandom
import torch as th
import sys
import quantization as q
# torch.autograd.set_detect_anomaly(True)
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM
import pickle
from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU
# When to switch between Dendrite learning and neuron learning. 
PBG.switchMode = PBG.doingHistory 
# How many normal epochs to wait for before switching modes, make sure this is higher than your scheduler's patience.
PBG.nEpochsToSwitch = 30  
# Same as above for Dendrite epochs
PBG.pEpochsToSwitch = 30
# The default shape of input tensors
PBG.inputDimensions = [-1, 0, -1, -1]
PBG.capAtN = True

seed = 42
seed = 42

width = float(0.5)
def normalize_cifar10(train_data, test_data):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return transform_train, transform_test

class CIFAR10VGG(nn.Module):
    def __init__(self):
        super(CIFAR10VGG, self).__init__()
        self.num_classes = 47
        self.weight_decay = 0.0005

        self.conv1 = PBG.PBSequential([nn.Conv2d(1, int(64*width), kernel_size=3, padding=4, bias=False),
        nn.BatchNorm2d(int(64*width))])
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = PBG.PBSequential([nn.Conv2d(int(64*width), int(64*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(64*width))])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = PBG.PBSequential([nn.Conv2d(int(64*width), int(128*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(128*width))])
        self.dropout2 = nn.Dropout(0.4)
        self.conv4 = PBG.PBSequential([nn.Conv2d(int(128*width), int(128*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(128*width))])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = PBG.PBSequential([nn.Conv2d(int(128*width), int(256*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(256*width))])
        self.dropout3 = nn.Dropout(0.4)
        self.conv6 = PBG.PBSequential([nn.Conv2d(int(256*width), int(256*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(256*width))])
        self.dropout4 = nn.Dropout(0.4)
        self.conv7 = PBG.PBSequential([nn.Conv2d(int(256*width), int(256*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(256*width))])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = PBG.PBSequential([nn.Conv2d(int(256*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout5 = nn.Dropout(0.4)
        self.conv9 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout6 = nn.Dropout(0.4)
        self.conv10 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout7 = nn.Dropout(0.4)
        self.conv12 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.dropout8 = nn.Dropout(0.4)
        self.conv13 = PBG.PBSequential([nn.Conv2d(int(512*width), int(512*width), kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(int(512*width))])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout9 = nn.Dropout(0.5)

        self.fc1 = PBG.PBSequential([nn.Linear(int(512*width), int(512*width)),
        nn.BatchNorm1d(int(512*width))])
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(int(512*width), self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu((self.conv2(x)))
        x = self.pool1(x)

        x = F.relu((self.conv3(x)))
        x = self.dropout2(x)
        x = F.relu((self.conv4(x)))
        x = self.pool2(x)

        x = F.relu((self.conv5(x)))
        x = self.dropout3(x)
        x = F.relu((self.conv6(x)))
        x = self.dropout4(x)
        x = F.relu((self.conv7(x)))
        x = self.pool3(x)
        x = F.relu((self.conv8(x)))
        x = self.dropout5(x)
        x = F.relu((self.conv9(x)))
        x = self.dropout6(x)
        x = F.relu((self.conv10(x)))
        x = self.pool4(x)

        x = F.relu((self.conv11(x)))
        x = self.dropout7(x)
        x = F.relu((self.conv12(x)))
        x = self.dropout8(x)
        x = F.relu((self.conv13(x)))
        x = self.pool5(x)
        x = self.dropout9(x)

        x = x.view(x.size(0), -1)
        x = F.relu((self.fc1(x)))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x


random_seed=42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
def evaluate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = total_loss / len(valid_loader)
    accuracy = 100 * correct / total
    return loss, accuracy

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  transform_train, transform_test = normalize_cifar10(None, None)

  train_dataset = torchvision.datasets.EMNIST(root='../ResNet/data', split='balanced', train=True, download=True, transform=transform_train)
  test_dataset = torchvision.datasets.EMNIST(root='../ResNet/data', split='balanced', train=False, download=True, transform=transform_test)

  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
  model = CIFAR10VGG().to(device)
  model = PN.loadPAIModel2(model, 'HalfThreeDendrites.pt')
  
  model = q.load_quantized_model(model, "HalfThreeCompressedModel.pt")
  model = model.to(device)
    # new_model.load_state_dict(torch.load("compressedWeights.pt"))
    
    # # torch.save(new_model, "Q/small_model.pt")
    # torch.save(new_model.state_dict(), "small_model_dict.pt")
        
  print("-------------------------------------------")
    
  model.eval()
    
   


   
  criterion = nn.CrossEntropyLoss()

  loss, accuracy = evaluate(model,test_loader, criterion)
    
  print("Accuracy after the experiment is: ", accuracy)