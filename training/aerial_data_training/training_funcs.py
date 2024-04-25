from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
import copy


def initialize_inception_model(num_classes, feature_extract, use_pretrained=True, model=None):
  # Initialize these variables which will be set in this if statement. Each of these
  #   variables is model specific.
  model_ft = None
  input_size = 0

  """ Inception v3
  Be careful, expects (299,299) sized images and has auxiliary output
  """
  if model is None:
    if use_pretrained:
      model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
    else:
      model_ft = models.inception_v3('pytorch/vision:v0.10.0', 'inception_v3')
  else:
    model_ft = model
  #"switching off" training on the majority of the model weights if feature_extract
  set_parameter_requires_grad(model_ft, feature_extract)
  # Handle the auxilary net
  num_ftrs = model_ft.AuxLogits.fc.in_features
  model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
  # Handle the primary net
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, num_classes)
  input_size = 299
  #adding an initial 1x1 convolutional layer to handle 4 channel input and reduce it to 3 channels
  input_embedding_module = nn.Sequential(OrderedDict([('input_conv', nn.Conv2d(4, 3, kernel_size=1))]))#keeping this module seperate for future use
  #wrapping the pretrained modek and the input embedding module together for a combined model
  model_ft = nn.Sequential(OrderedDict([("pre_layer", input_embedding_module), ("inception_v3", model_ft)]))

  return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=True, tensorboard_writer=None, early_stopping=False, patience=5, stop_tol=1e-3, device='cpu'):
    early_stop = False

    if tensorboard_writer is not None:
        use_tensorboard = True
        writer = tensorboard_writer
    else:
        use_tensorboard = False
    
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    final_epoch = num_epochs

    for epoch in range(num_epochs):
        if early_stop:
            break
        else:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

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
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            if use_tensorboard:
                                writer.add_scalar('Training Loss', loss, epoch)
                                writer.add_scalar('Training Accuracy', torch.sum(preds == labels.data).double() / len(labels), epoch)
                            
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'train':
                    train_acc_history.append(epoch_acc.item())
                    train_loss_history.append(epoch_loss)
                else:
                    #deep copying the model if its performance has improved
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                
                    #saving validation performance
                    val_acc_history.append(epoch_acc.item())
                    val_loss_history.append(epoch_loss)
                    #early stopping
                    if early_stopping:
                        if len(val_loss_history) > patience:
                            if (np.mean(val_loss_history[-patience:]) < stop_tol) or (np.all(np.array(val_loss_history[-8:]) > min(val_loss_history))):
                                print("Early stopping activated")
                                early_stop = True
                                final_epoch = epoch
                            
                    if use_tensorboard:
                        writer.add_scalar('Validation Accuracy', epoch_acc, epoch)
                        writer.add_scalar('Validation Loss', epoch_loss, epoch)

            print("epoch {} complete".format(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if use_tensorboard:
        #making sure all data is written to disk
        writer.flush()
    
    # load best model weights and return best model
    model.load_state_dict(best_model_wts)
    convergence_dict = {'train_acc': train_acc_history, 
                        'train_loss': train_loss_history, 
                        'val_acc': val_acc_history, 
                        'val_loss': val_loss_history}
    
    if early_stopping:
        return model, convergence_dict, final_epoch
    else:
        return model, convergence_dict