import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
import seaborn as sns

class ActivationTracker:
    def __init__(self, model):
        self.activations = defaultdict(lambda: defaultdict(list))  # {layer_name: {epoch: [(activations, labels)]}}
        self.hooks = []
        self.current_activations = {}  # Store activations for all layers in one forward pass
        self.register_hooks(model)

    def register_hooks(self, model):
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):  # Track relevant layers
                hook = layer.register_forward_hook(self.get_hook(name))
                self.hooks.append(hook)

    def get_hook(self, layer_name):
        def hook(module, input, output):
            self.current_activations[layer_name] = output.detach().cpu()
        return hook

    def capture_activations(self, epoch, labels):
        if self.current_activations:  # Ensure activations exist before storing
            for layer, activation in self.current_activations.items():
                self.activations[layer][epoch].append((activation, labels.detach().cpu()))
        self.current_activations = {}  # Reset for the next forward pass

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
# Example model
def create_model():
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 5)
    )

# Example training loop
def train(model, dataloader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    tracker = ActivationTracker(model)
    

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

             # Compute regularization term
            loss += l1_lambda * L1_regularization(model) # Add regularization to the loss

            loss.backward()
            optimizer.step()
            
            #tracker.capture_activations(epoch, targets)
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


    tracker.remove_hooks()
    return tracker.activations  # Dictionary containing activations and labels
