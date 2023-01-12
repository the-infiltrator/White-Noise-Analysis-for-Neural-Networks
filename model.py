import torch.optim as optim
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, List



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(model_type, input_shape, num_classes):
  # Check if the specified model type is simple or complex
  if model_type == 'simple':
    # Define the simple CNN model
    model = nn.Sequential(
      nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(in_features=7*7*64, out_features=128),
      nn.Linear(in_features=128, out_features=num_classes)
    )
  elif model_type == 'complex':
    # Define the complex CNN model
    model = nn.Sequential(
      nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(num_features=32),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(in_features=7*7*64, out_features=128),
      nn.Linear(in_features=128, out_features=num_classes)
    )
  else:
    raise ValueError('Invalid model type')
  return model


def train(model, train_loader, valid_loader, num_epochs=100, early_stopping_patience=5, lr_min=1e-4, lr_max=1e-2, weight_decay=1e-4):
  """Trains a PyTorch model with cosine annealing learning rate scheduling and early stopping.

  Args:
    model: The model to be trained (a PyTorch nn.Module).
    train_loader: A PyTorch DataLoader for the training data.
    valid_loader: A PyTorch DataLoader for the validation data.
    num_epochs: The number of epochs to train for (default 100).
    early_stopping_patience: The number of epochs to wait for a improvement in validation accuracy before stopping training (default 5).
    lr_min: The minimum learning rate (default 1e-4).
    lr_max: The maximum learning rate (default 1e-2).
    weight_decay: The weight decay for the Adam optimizer (default 1e-4).

  Returns:
    The trained model (a PyTorch nn.Module).
  """
    # Load the trained models
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  # Define the Adam optimizer
  optimizer = optim.Adam(model.parameters(), lr=lr_max, weight_decay=weight_decay)

  # Set the initial learning rate and the learning rate scheduler
  lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, lr_min)

  # Set the criterion (loss function)
  criterion = nn.CrossEntropyLoss()

  # Set the number of epochs to train for and the early stopping patience
  num_epochs_to_train = num_epochs
  early_stopping_counter = 0

  # Set the best validation accuracy to 0
  best_valid_acc = 0.0

  # Set the model to training mode
  model.train()

  # Loop over the number of epochs
  for epoch in range(num_epochs):
    # Set the loss and accuracy for the epoch to 0
    epoch_loss = 0.0
    epoch_acc = 0.0
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model = model.to(device)

    # Loop over the training data
    for data, target in train_loader:
      # Move the data and target to the device
      data, target = data.to(device), target.to(device)

      # Zero the gradients
      optimizer.zero_grad()

      # Forward pass
      output = model(data)
      loss = criterion(output, target)

      # Backward pass
      loss.backward()
      optimizer.step()

      # Update the loss and accuracy for the epoch
      epoch_loss += loss.item()
      epoch_acc += (output.argmax(1) == target).float().mean().item()

    # Update the learning rate
    lr_scheduler.step()

    # Set the model to evaluation mode
    model.eval()

    # Set the validation loss and accuracy for the epoch to 0
    valid_loss = 0.0
    valid_acc = 0.0

    # Turn off gradients for validation
    with torch.no_grad():
      # Loop over the validation data
      for data, target in valid_loader:
        # Move the data and target to the device
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Update the validation loss and accuracy for the epoch
        valid_loss += loss.item()
        valid_acc += (output.argmax(1) == target).float().mean().item()

    # Calculate the average loss and accuracy for the epoch
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader)

    # Print the epoch
    # Print the training and validation results for the epoch
    print(f"Epoch: {epoch+1:2d} | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Train Acc: {epoch_acc*100:.1f}% | "
          f"Valid Loss: {valid_loss:.4f} | "
          f"Valid Acc: {valid_acc*100:.1f}%")

    # If the current model has the best validation accuracy, update the best validation accuracy and reset the early stopping counter
    if valid_acc > best_valid_acc:
      best_valid_acc = valid_acc
      early_stopping_counter = 0
    # If the current model does not have the best validation accuracy, increment the early stopping counter
    else:
      early_stopping_counter += 1

    # If the early stopping counter has reached the early stopping patience, stop training
    if early_stopping_counter >= early_stopping_patience:
      num_epochs_to_train = epoch + 1
      break

  # Return the trained model
  return model

def train_all_models(
    train_batch_size: int,
    num_epochs: int = 100,
    early_stopping_patience: int = 5,
    lr_min: float = 1e-4,
    lr_max: float = 1e-2,
    weight_decay: float = 1e-4,) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    # Load the MNIST and Fashion-MNIST datasets
    mnist_train, mnist_valid, mnist_test = get_data_loader("mnist", train_batch_size, shuffle=True, num_workers=4, pin_memory=False)
    fmnist_train, fmnist_valid, fmnist_test = get_data_loader("fashion-mnist", train_batch_size, shuffle=True, num_workers=4, pin_memory=False)
    
    # List of models to train
    models_to_train = [
        ("simple", "mnist", mnist_train, mnist_valid),
        ("complex", "mnist", mnist_train, mnist_valid),
        ("simple", "fashion-mnist", fmnist_train, fmnist_valid),
        ("complex", "fashion-mnist", fmnist_train, fmnist_valid),
    ]
    
    # Train all models
    trained_models = []
    for model_type, dataset, train_data, valid_data in models_to_train:
        print(f"Training {model_type} model on {dataset}...")
        # Get the appropriate model
        input_shape, num_classes = (1, 28, 28), 10
        model = get_model(model_type, input_shape, num_classes)
        # Train the model
        model = train(model, train_data, valid_data, num_epochs=num_epochs, early_stopping_patience=early_stopping_patience, lr_min=lr_min, lr_max=lr_max, weight_decay=weight_decay)
        trained_models.append(model)
    
    return tuple(trained_models)