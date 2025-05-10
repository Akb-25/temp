import torch
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
#import pandas as pd
import numpy as np
import logging
import csv
from time import localtime, strftime
import argparse
from torchvision import datasets, transforms
import os
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from itertools import zip_longest
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import resnet as models
import os
import quantization as q


from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM


# torch.autograd.set_detect_anomaly(True)
seed = 42
random_seed = 42

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
  norm_mean, norm_var = 0.0, 1.0

  th.manual_seed(seed)
  th.cuda.manual_seed(seed)
  th.cuda.manual_seed_all(seed)
  th.backends.cudnn.deterministic = True
  th.cuda.set_device(0)

  if th.cuda.is_available():
      th.cuda.manual_seed_all(random_seed)
      device=th.device("cuda")
  else:
      device=th.device("cpu")
  print(device)
  N = 1

  batch_size_tr = 128
  batch_size_te = 128

  

  epochs = 182
  custom_epochs=15
  new_epochs=80
  optim_lr=0.0001
  milestones_array=[100]
  lamda=0.001

  

  gpu = th.cuda.is_available()

  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--save-name', type=str, default='PB')
  parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                      help='input batch size for training (default: 128)')
  parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                      help='input batch size for testing (default: 128)')
  parser.add_argument('--epochs', type=int, default=128, metavar='N',
                      help='number of epochs to train (default: 14)')
  parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                      help='learning rate (default: 1.0)')
  parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--no-mps', action='store_true', default=False,
                      help='disables macOS GPU training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=False,
                      help='For Saving the current Model')
  args = parser.parse_args()
  use_cuda = torch.cuda.is_available()
  use_mps = not args.no_mps and torch.backends.mps.is_available()
  torch.manual_seed(args.seed)
  if use_cuda:
      device = torch.device("cuda")
  elif use_mps:
      device = torch.device("mps")
  else:
      device = torch.device("cpu")

  train_kwargs = {'batch_size': args.batch_size}
  test_kwargs = {'batch_size': args.test_batch_size}
  if use_cuda:
      cuda_kwargs = {'num_workers': 1,
                     'pin_memory': True,
                     'shuffle': True}
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)


  num_classes = 47
  image_size = 32
  #Define the data loaders
  transform_train = transforms.Compose(
          [ 
              #transforms.CenterCrop(26),
              transforms.Resize((image_size,image_size)),
              transforms.RandomRotation(10),      
              transforms.RandomAffine(5),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
              transforms.Normalize((0.1307,), (0.3081,)),
          ])
  transform_test = transforms.Compose(
          [ 
              transforms.Resize((image_size,image_size)),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
              transforms.Normalize((0.1307,), (0.3081,)),
          ])
  
  PBG.moduleNamesToConvert.append('BasicBlock')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #Dataset
  dataset1 = datasets.EMNIST(root='../ResNet/data', split='balanced', train=True, download=True, transform=transform_train)
  dataset2 = datasets.EMNIST(root='../ResNet/data',  split='balanced', train=False, download=True, transform=transform_test)
  train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
  total_step = len(train_loader)

  model = models.resnet18(num_classes == num_classes)
    
  model = PN.loadPAIModel(model, 'nets/net0.25_6.pt')
  
  model = q.load_quantized_model(model, "FullCompressedModel2.pt")
  model = model.to(device)
    # new_model.load_state_dict(torch.load("compressedWeights.pt"))
    
    # # torch.save(new_model, "Q/small_model.pt")
    # torch.save(new_model.state_dict(), "small_model_dict.pt")
        
  print("-------------------------------------------")
    
  model.eval()
    
   


   
  criterion = nn.CrossEntropyLoss()

  loss, accuracy = evaluate(model,test_loader, criterion)
    
  print("Accuracy after the experiment is: ", accuracy)