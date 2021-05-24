

import glob
from PIL import Image
import os
import numpy as np

import pickle

# load lists from pickle file

image_list = []
image_labels = []

with open('image_list.pkl', 'rb') as f:
  image_list = pickle.load(f)

with open('image_labels.pkl', 'rb') as f:
  image_labels = pickle.load(f)

print(len(image_list))

three_channel_imgs = set()
remove_data = set()
for i, x in enumerate(image_list):
  if len(x.shape) > 2:
    three_channel_imgs.add(image_labels[i])
    remove_data.add(i)

x = []
y = []
for i, img in enumerate(image_list):
  if i not in remove_data:
    x.append(image_list[i])
    y.append(image_labels[i])

print(len(x), len(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

 X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

 print(len(X_train), len(X_val), len(X_test))

import torchvision.transforms as transforms

transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])  # [transforms.Resize((255,255)),transforms.ToTensor()]) 

X_train_np = np.zeros((len(X_train), 1, 32, 32))
for i, img in enumerate(X_train):
  pil_image = Image.fromarray(img)
  out = transform(pil_image)
  X_train_np[i, :, :, :] = out

X_val_np = np.zeros((len(X_val), 1, 32, 32))
for i, img in enumerate(X_val):
  pil_image = Image.fromarray(img)
  out = transform(pil_image)
  X_val_np[i, :, :, :] = out

X_test_np = np.zeros((len(X_test), 1, 32, 32))
for i, img in enumerate(X_test):
  pil_image = Image.fromarray(img)
  out = transform(pil_image)
  X_test_np[i, :, :, :] = out

print(X_train_np.shape)
print(X_val_np.shape)
print(X_test_np.shape)

y_train_np = np.zeros((len(y_train),))
for i, label in enumerate(y_train):
  y_train_np[i] = label

y_val_np = np.zeros((len(y_val),))
for i, label in enumerate(y_val):
  y_val_np[i] = label

y_test_np = np.zeros((len(y_test),))
for i, label in enumerate(y_test):
  y_test_np[i] = label

print(y_train_np.shape,y_val_np.shape, y_test_np.shape )

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)

import torch.optim as optim
import torch
import torch.nn as nn

import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from sklearn import metrics
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

batch_size=32

X_train_tensor = torch.Tensor(X_train_np) # transform to torch tensor
y_train_tensor = torch.Tensor(y_train_np)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # create your datset
dataloader_train = DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

X_val_tensor = torch.Tensor(X_val_np) # transform to torch tensor
y_val_tensor = torch.Tensor(y_val_np)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor) # create your datset
dataloader_val = DataLoader(val_dataset, batch_size=batch_size) # create your dataloader

X_test_tensor = torch.Tensor(X_test_np) # transform to torch tensor
y_test_tensor = torch.Tensor(y_test_np)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor) # create your datset
dataloader_test = DataLoader(test_dataset, batch_size=batch_size) # create your dataloader

# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def check_accuracy(loader, model):
  # print(loader.dataset)
  # if loader.dataset.train:
  #     print('Checking accuracy on validation set')
  # else:
  #     print('Checking accuracy on test set')   
  num_correct = 0
  num_samples = 0
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
      for x, y in loader:
          x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
          y = y.to(device=device, dtype=torch.long)
          scores = model(x)
          _, preds = scores.max(1)
          num_correct += (preds == y).sum()
          num_samples += preds.size(0)
      acc = float(num_correct) / num_samples
      print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

num_epochs = 10
batch_size = 32

class SVM_Loss(nn.modules.Module):    
    def __init__(self):
        super(SVM_Loss,self).__init__()
    def forward(self, outputs, labels):
         return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size

svm_loss_criteria = SVM_Loss()
losses = []
val_losses = []

def train(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(dataloader_train):
            model.train()  # put model to training mode

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            
            loss = F.multi_margin_loss(scores, y) 
            if t == 0:
              losses.append(loss.item())


            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(dataloader_val, model)
                print()
        # losses.append(loss.item())

def test(model, dataloader):
    check_accuracy(dataloader, model)

model = None
optimizer = None

learning_rate = 1e-3

model = nn.Sequential(
    nn.Conv2d(1, 32, 5, padding=2),
    nn.ReLU(),

    nn.BatchNorm2d(32),
    nn.Conv2d(32, 64, 5, padding=2),
    nn.ReLU(),

    Flatten(),
    nn.Dropout2d(0.2),
    nn.Linear(32 * 32 * 64, 32*32),

    # nn.BatchNorm1d(32*32),
    nn.Dropout2d(0.2),
    nn.Linear(32*32, 4)
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

model.eval()

train(model, optimizer, epochs=10)

test(model, dataloader_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.title("Loss for Simple Network")
plt.ylabel("loss")
plt.xlabel("iteration")

plt.plot(losses)

plt.savefig('losses.png')

# plt.plot(val_losses)

