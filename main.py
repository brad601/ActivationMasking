import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
from mnistmodels import DeepLinearNN

model_file = "nist_digit.mdl"

# Hyperparameter for L1 regularization
l1_lambda = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Prepare the MNIST dataset for convolution
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
#])
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # Flatten

bulk_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_size = int(0.8 * len(bulk_dataset))
test_size = len(bulk_dataset) - train_size
train_dataset, test_dataset = random_split(bulk_dataset, [train_size, test_size])
print(f"Dataset sizes. Train: {len(train_dataset)}, Test: {len(test_dataset)}")

def L2_regularization(model):
    l2_norm = sum(param.pow(2).sum() for param in model.parameters())
    return l2_norm

def L1_regularization(model):
    l1_norm = sum(param.abs().sum() for param in model.parameters())
    return l1_norm

def load_or_train_model(model, training_dataset, epochs=5):
    
    if (os.path.isfile(model_file)):
        print("loading model from", model_file)
        model.load_state_dict(torch.load(model_file,weights_only=True))
    else:
        print("training on", device)
        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in training_dataset:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Compute regularization term
                loss += l1_lambda * L2_regularization(model) # Add regularization to the loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

        #save model file
        torch.save(model.state_dict(), model_file)
        print("Model saved to", model_file)

def digit_loader_eval(model, dataloader):
  model.to(device)
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
      for images, labels in dataloader:
          images, labels = images.to(device), labels.to(device)

          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()


  accuracy = 100 * correct / total
  print(f"Test Accuracy: {accuracy:.2f}%")

  return accuracy


from maskedactivationtracker import MaskedActivationTracker

def perform_masked_activation_task(model):
    
    tracker = MaskedActivationTracker(model, mask_prob=0.5, num_masks=100)

    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    successful_masks = tracker.test_masks(test_loader)

    tracker.remove_hooks()

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset

def split_dataset_by_label(dataset):
    """
    Splits a dataset into multiple subsets based on labels.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split. Assumes dataset[i] returns (data, label).

    Returns:
        dict: A dictionary where keys are labels and values are Subset datasets.
    """
    label_to_indices = defaultdict(list)

    # Collect indices for each label
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    # Create subsets for each label
    label_datasets = {label: Subset(dataset, indices) for label, indices in label_to_indices.items()}

    return label_datasets

def activation_disco(model, dataset, model_layer, mask_sparsity=0.2):
    
    accumulator = defaultdict()

    for item in dataset:
        loader = DataLoader(dataset=dataset[item], batch_size=1, shuffle=False)

        with torch.no_grad():
            mask_instance = model.mask_registry.get_instance(model_layer)

            for _, labels in loader:
                accumulator[int(labels[0])] = torch.zeros_like(mask_instance.get_mask())

            count = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                #accumulator[int(labels[0])] = torch.zeros_like(mask_instance.get_mask())

                outputs = model(images)
                if torch.max(outputs, 1)[1] == labels:
                    # at this point we have a sucessful classification
                    # re-run this sample now with permuted mask
                    new_mask  = (torch.ones_like(mask_instance.get_mask()))
                    new_mask_len = len(new_mask)
                    for iteration in range(new_mask_len):
                        new_mask[iteration] = 0
                        
                        mask_instance.set_mask(new_mask)
                        outputs = model(images)

                        if torch.max(outputs, 1)[1] != labels:
                            new_mask[iteration]  = 1

                    accumulator[int(labels[0])] += new_mask
                if count > 100:
                    break
                else:
                    count += 1
                mask_instance.reset_mask()
                    
    #print(accumulator)
    return accumulator               

def validate_mask(model, model_layer, mask, validation_dataset):
    
    model.mask_registry.get_instance(model_layer).set_mask(mask)
    ret = digit_loader_eval(model, DataLoader(dataset=validation_dataset, batch_size=64, shuffle=False))
    model.mask_registry.get_instance(model_layer).reset_mask()

    return ret

import torch

def top_n_mask(tensor, n):
    """
    Sets the N highest values in the tensor to 1 and the rest to 0.

    Args:
        tensor (torch.Tensor): Input tensor.
        n (int): Number of highest values to set to 1.

    Returns:
        torch.Tensor: A binary mask tensor with the N highest values set to 1.
    """
    flat_tensor = tensor.flatten()  # Flatten the tensor for easy indexing
    values, indices = torch.topk(flat_tensor, n)  # Get top N values and their indices
    mask = torch.zeros_like(flat_tensor)  # Initialize mask with zeros
    mask[indices] = 1  # Set top N positions to 1
    return mask.view(tensor.shape)  # Reshape back to original tensor shape

# Example usage:

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    mnist_model = DeepLinearNN().to(device)

    load_or_train_model(mnist_model, train_loader, epochs=10)
    digit_loader_eval(mnist_model, test_loader)
    
    mnist_datasets_by_label = split_dataset_by_label(bulk_dataset)
    non_interfering_activations_histogram = activation_disco(mnist_model, mnist_datasets_by_label, 1, 0.25)
    
    import math


    for index in mnist_datasets_by_label.keys():
        new_mask = top_n_mask(non_interfering_activations_histogram[index], math.ceil(len(non_interfering_activations_histogram[index]) * .99))
        err_rate = validate_mask(model=mnist_model, model_layer=1, mask=new_mask, validation_dataset=mnist_datasets_by_label[index])
        print("Label:", index, "Accuracy", err_rate, "mask", new_mask)


    #7: tensor               ([ 2., 0., 10., 1., 2., 3., 1., 1., 2., 0.]),
    #new_mask = top_n_mask(non_interfering_activations_histogram[7], 1)
    #new_mask = torch.IntTensor([0., 0., 1., 0., 0., 1., 0., 0., 0., 0.])  
    #validate_mask(model=mnist_model, model_layer=2, mask=new_mask, validation_dataset=mnist_datasets_by_label[7])

    #9: tensor                ([7., 9., 2., 3., 0., 0., 1., 2., 1., 1.]),
    #new_mask = torch.IntTensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]) 
    #validate_mask(model=mnist_model, model_layer=2, mask=new_mask, validation_dataset=mnist_datasets_by_label[9])

if __name__ == '__main__':
    main()


