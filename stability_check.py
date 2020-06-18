# -*- coding: utf-8 -*-

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import numpy as np
import matplotlib.pyplot as plt
import os 
import collections
from sklearn.model_selection import StratifiedShuffleSplit
import copy
import math
import time
from datetime import datetime
from pathlib import Path
from glob import glob
import pickle as pkl
from scipy.io import loadmat
from PIL import Image
import zipfile

##################################
# Torch 
##################################

from torchvision import datasets, transforms, utils, models
import torchvision.transforms as transforms 
from torch.utils.data import Subset
from torchvision import utils
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import models
from torch import nn
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, StepLR, CosineAnnealingWarmRestarts
import torch.nn as nn
import torch.nn.functional as F
import shutil
import random
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, StepLR, CosineAnnealingWarmRestarts



import warnings
warnings.filterwarnings('ignore')

# load data 



def load_data():

    all_dir = os.getcwd() 
    dir_path1 = all_dir + '/data_new'
    dir_path2 = all_dir + '/data_new_2'
    dir_path3 = all_dir + '/__MACOSX'


    if os.path.exists(dir_path1) and os.path.isdir(dir_path1):    
        shutil.rmtree(dir_path1)
    if os.path.exists(dir_path2) and os.path.isdir(dir_path2):
        shutil.rmtree(dir_path2)
    if os.path.exists(dir_path3) and os.path.isdir(dir_path3):
        shutil.rmtree(dir_path3)


    # Load meta data
    zip_ref = zipfile.ZipFile('data_new.zip', 'r')
    zip_ref.extractall(all_dir)
    zip_ref.close()

    # split data 

    all_dir = Path(all_dir)
    data_new = "data_new" 
    p = "data_new_2"
    X = ["train","val"]
    train = X[0]
    val = X[1]

    if not os.path.exists(all_dir/p):
      os.makedirs(all_dir/p)


    for x in X:  
      for d in os.listdir(all_dir/data_new):
        if not os.path.exists(all_dir/p/x/d):
          os.makedirs(all_dir/p/x/d)
      

    for d in os.listdir(all_dir/data_new):
      if d == '.DS_Store':
         os.remove(all_dir/data_new/d)
         continue 

      l = os.listdir(all_dir/data_new/d)
      random.shuffle(l)

      for f in l:
        u = random.uniform(0, 1)
        if (u < 0.2):
          shutil.move(all_dir/data_new/d/f, all_dir/p/val/d/f)
        else:
          shutil.move(all_dir/data_new/d/f, all_dir/p/train/d/f)
        

         

#load_data()

def show_data_details():
  all_dir = os.getcwd() 
  all_dir = Path(all_dir)
  data_new = "data_new" 
  p = "data_new_2"
  X = ["train","val"]
  train = X[0]
  val = X[1]

  for x in X:
    for d in os.listdir(all_dir/data_new):
      print(x + " " +  " " + d  +  " :  " + str( len(os.listdir(all_dir/p/x/d)))) 

#show_data_details()





# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 7

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for 
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False

data_dir = os.getcwd() + "/data_new_2"

"""
----------------
Helper Functions
----------------
"""

def get_lr(opt):
  for param_group in opt.param_groups:
    return param_group['lr']





def train_model(model, dataloaders,lr_scheduler, criterion, optimizer, num_epochs, is_inception=False):
#def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        current_lr=get_lr(optimizer)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        ##print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #print(labels)
                labels = labels - 1 
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    if is_inception and phase == 'train':

                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:

                        outputs = model(inputs)
                        loss = F.cross_entropy(outputs, labels+1)

                        

                    _, preds = torch.max(outputs, 1)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                lr_scheduler.step()
        #print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model_ft, val_acc_history, val_loss_history, train_acc_history, train_loss_history
    #return model_ft, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    global model_ft
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        

      

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# Initialize the model for this run
#model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
#print(model_ft)

"""# Averaging"""

def single_simulation(num_epochs):

  load_data()
  show_data_details()
  global device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  
  # Initialize the model for this run
  model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
  data_transforms = {
    'train': transforms.Compose([
      transforms.Resize(input_size),
      transforms.RandomResizedCrop(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.5917261, 0.4514425, 0.41582367], [0.17611828, 0.1824943, 0.1785681])
  ]),
  'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.5917261, 0.4514425, 0.41582367], [0.17611828, 0.1824943, 0.1785681])
  ]),
  }


  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
  # Create training and validation dataloaders
  dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
  # Detect if we have a GPU available


  # Send the model to GPU
  model_ft = model_ft.to(device)

  # Gather the parameters to be optimized/updated in this run. 

  params_to_update = model_ft.parameters()
  if feature_extract:
      params_to_update = []
      for name,param in model_ft.named_parameters():
          if param.requires_grad == True:
              params_to_update.append(param)
              #print("\t",name)
  else:
      for name,param in model_ft.named_parameters():
          if param.requires_grad == True:
              #print("\t",name)
              continue

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)

  # Setup the loss fxn
  criterion = nn.CrossEntropyLoss()

    
  exp_lr_scheduler = CosineAnnealingWarmRestarts(optimizer_ft, T_0 = 1, T_mult = 1)

  # Train and evaluate
  model_ft, val_acc_history, val_loss_history, train_acc_history, train_loss_history  = train_model(model_ft, dataloaders_dict,exp_lr_scheduler, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)

  return val_acc_history, val_loss_history, train_acc_history, train_loss_history

def simulations(n_simu , num_epochs):
  t1 = time.time()
  Av_loss_val = []
  Av_loss_train = []
  k = 1
  for _ in range(n_simu):
      print(f" iteration {k}")
      val_acc_history, val_loss_history, train_acc_history, train_loss_history = single_simulation(num_epochs)

      Av_loss_val.append(val_loss_history)
      Av_loss_train.append(train_loss_history)
      k += 1

  av_loss_val = np.mean(np.array(Av_loss_val),axis =0)
  sd_loss_val = np.sqrt(np.var(np.array(Av_loss_val),axis =0))
 
  t2 = time.time()
  print(f"  time of all simulationsis {(t2-t1)/60} min")

  #plt.figure(figsize=(25,6))
  plt.title("Val Loss average")
  plt.plot(range(1,num_epochs+1),av_loss_val, 'or')
  plt.plot(range(1,num_epochs+1),av_loss_val, '-', color='gray')
  plt.fill_between(range(1,num_epochs+1), av_loss_val - sd_loss_val, av_loss_val + sd_loss_val,
                   color='gray', alpha=0.2)                
  plt.plot(range(1,num_epochs+1),av_loss_val,label="val loss")
  plt.ylabel("Loss")
  plt.xlabel("Training Epochs")
  plt.legend()
  plt.savefig("imx")
  plt.show()
  

if __name__ == '__main__':
    
    simulations(n_simu = 1, num_epochs = 100)




