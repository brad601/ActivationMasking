import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split

# Define the CNN model
class NIST_DIGIT_CNN(nn.Module):
    bottleneck_size: int = 128*10

    def __init__(self):
        super(NIST_DIGIT_CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,        # Input channels
            out_channels=32,      # Number of filters
            kernel_size=5,        # Change kernel size to 5x5
            stride=1,             # Keep stride as 1
            padding=2             # Adjust padding to maintain spatial dimensions
        )

        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 2)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)

        return x


# Define a fully connected (linear-only) model
class LinearMNIST(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, bottleneck_size=4, num_classes=10):
        super(LinearMNIST, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, bottleneck_size)  # Bottleneck layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(bottleneck_size, num_classes)  # Final classification layer
        self.s = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the 28x28 image into a vector
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))  # Bottleneck
        x = self.s(self.fc3(x))
        
        return x

class MaskRegistry:
    def __init__(self):
        self.registry = {}
    
    def register(self, instance, instance_id):
        self.registry[instance_id] = instance

    def get_instance(self, instance_id):
        return self.registry[instance_id]
    
    def get_managed_mask_ids(self):
        return self.registry.keys()

class Mask(nn.Module):
    def __init__(self, hidden_dim, registry, mask_id):
        super(Mask, self).__init__()
        self.size = hidden_dim
        self.mask = torch.ones(hidden_dim)
        registry.register(self, mask_id)
    
    def forward(self, x):
        return x * self.mask

    def set_mask(self, mask):
        if( mask.size() == self.mask.size()):
            self.mask = mask
        
        else:
             raise Exception(f"Size of mask provided: {mask.size()} is not valid for mask of size {self.mask.size()}.")

    def get_mask(self):
        return self.mask

    def reset_mask(self):
        self.mask = torch.ones(self.size)

class LearnableMask(nn.Module):
    def __init__(self, activation_shape, registry, mask_id, init_value=0.9):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.full(activation_shape, init_value))  # Soft mask (logits)
        #self.registry = mask_registry
        registry.register(self, mask_id)

    def forward(self, activations):
        soft_mask = torch.sigmoid(self.mask_logits)  # Convert logits to [0,1] range
        return activations * soft_mask  # Apply soft mask

    def binarize(self, threshold=0.5):
        """Convert soft mask to binary mask."""
        return (torch.sigmoid(self.mask_logits) > threshold).float()

    def overwrite_with_binary(self, high_value=1, low_value=0, threshold=0.5):
        """Overwrites mask_logits with discrete high/low values to enforce a hard mask."""
        with torch.no_grad():  # Prevent gradient updates
            binary_mask = self.binarize(threshold)
            self.mask_logits.copy_(binary_mask * high_value + (1 - binary_mask) * low_value)

    def get_mask(self):
        return self.mask_logits
    

class DeepLinearNN(nn.Module):
  def __init__(self, input_dim=784, hidden_dims=[51, 45, 30, 22, 64], output_dim=10):
    super(DeepLinearNN, self).__init__()
    self.mask_registry = MaskRegistry()

    layers = []
    prev_dim = input_dim
    mask_id = 0
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())  # Keeping some non-linearity, can be removed
        layers.append(nn.BatchNorm1d(hidden_dim))  # Optional, helps stabilize
        #layers.append(Mask(hidden_dim, registry=self.mask_registry, mask_id=mask_id))
        layers.append(LearnableMask([1, hidden_dim], self.mask_registry, mask_id=mask_id))
        mask_id += 1
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))  # Output layer
    
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

