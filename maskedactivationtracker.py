import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class MaskedActivationTracker:
    def __init__(self, model, mask_prob=0.5, num_masks=100):
        self.model = model
        self.mask_prob = mask_prob  # Probability of keeping an activation
        self.num_masks = num_masks  # Number of different masks to try per input
        self.activations = {}
        self.hooks = []
        self.successful_masks = defaultdict(lambda: torch.zeros_like(next(model.parameters())))  # Summed successful masks
        self.register_hooks()

    def register_hooks(self):
        """Registers hooks on all linear layers to capture activations."""
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):  # Track only fully connected layers
                hook = layer.register_forward_hook(self.get_hook(name))
                self.hooks.append(hook)

    def get_hook(self, layer_name):
        """Defines hook function to store activations."""
        def hook(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook

    def apply_mask(self, activations):
        """Applies a random binary mask with probability `mask_prob`."""
        mask = (torch.rand_like(activations) < self.mask_prob).float()  # Binary mask
        return activations * mask, mask

    def test_masks(self, dataloader):
        """Runs masked activations through the model and records which masks preserve classification."""
        self.model.eval()  # Set to eval mode
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)  # Run forward pass to get activations
                
                for _ in range(self.num_masks):
                    layer_masks = {}  # Stores masks per layer
                    masked_activations = {}  # Stores masked activations
                    
                    # Apply masking to all captured activations
                    for layer_name, activation in self.activations.items():
                        masked_act, mask = self.apply_mask(activation)
                        masked_activations[layer_name] = masked_act
                        layer_masks[layer_name] = mask
                    
                    # Run the model again with masked activations
                    masked_outputs = self.forward_with_masks(inputs, masked_activations)

                    for layer_name, mask in layer_masks.items():
                      if layer_name not in self.successful_masks:  
                          self.successful_masks[layer_name] = torch.zeros_like(mask[0])  # Use per-unit mask shape
                      self.successful_masks[layer_name] += mask.sum(dim=0)  

        return self.successful_masks  # Return summed masks per layer

    def forward_with_masks(self, inputs, masked_activations):
        """Manually forward-pass activations while injecting masked activations."""
        x = inputs
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in masked_activations:  # Replace activation with masked version
                x = masked_activations[name]
        return x  # Final model output

    def remove_hooks(self):
        """Removes registered hooks after analysis."""
        for hook in self.hooks:
            hook.remove()

