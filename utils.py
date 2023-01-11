import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, List

import get_model from model



def get_data_loader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=False):
  if dataset == 'fashion-mnist':
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
  elif dataset == 'mnist':
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  else:
    raise ValueError('Invalid dataset name')
  
  num_train = len(trainset)
  indices = list(range(num_train))
  split = int(0.8 * num_train)
  train_idx, valid_idx = indices[:split], indices[split:]
  train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
  valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
  valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

  return train_loader, valid_loader, test_loader




def save_models(models: Tuple[nn.Module, nn.Module, nn.Module, nn.Module], directory: str) -> None:
    """Saves the given CNN models to the specified directory.

    Args:
      models: A tuple of the 4 trained CNN models (simple MNIST, complex MNIST, simple Fashion-MNIST, complex Fashion-MNIST).
      directory: The directory to save the models to.

    Returns:
      None. The models are saved to disk.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    simple_mnist_model, complex_mnist_model, simple_fmnist_model, complex_fmnist_model = models

    torch.save(simple_mnist_model.state_dict(), os.path.join(directory, "simple_mnist_model.pt"))
    torch.save(complex_mnist_model.state_dict(), os.path.join(directory, "complex_mnist_model.pt"))
    torch.save(simple_fmnist_model.state_dict(), os.path.join(directory, "simple_fmnist_model.pt"))
    torch.save(complex_fmnist_model.state_dict(), os.path.join(directory, "complex_fmnist_model.pt"))


def load_models(directory):
    simple_mnist_model = get_model('simple',(1, 28, 28), 10)
    complex_mnist_model= get_model('complex',(1, 28, 28), 10)
    simple_fmnist_model = get_model('simple',(1, 28, 28), 10)
    complex_fmnist_model= get_model('complex',(1, 28, 28), 10)
    simple_mnist_model.load_state_dict(torch.load(f'{directory}/simple_mnist_model.pt', map_location=device))
    complex_mnist_model.load_state_dict(torch.load(f'{directory}/complex_mnist_model.pt', map_location=device))
    simple_fmnist_model.load_state_dict(torch.load(f'{directory}/simple_fmnist_model.pt', map_location=device))
    complex_fmnist_model.load_state_dict(torch.load(f'{directory}/complex_fmnist_model.pt', map_location=device))
    return simple_mnist_model, complex_mnist_model, simple_fmnist_model, complex_fmnist_model


def visualise_preds(model, data_loader, class_names):
    # Set the model to evaluation mode
    model.eval()
    # Create a blank image for each class
    class_images = [np.zeros((28, 28)) for _ in range(len(class_names))]
    # Create a counter for each class
    class_counts = [0 for _ in range(len(class_names))]
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Turn off gradients
    with torch.no_grad():
        # Loop over the data
        for data, target in data_loader:
            # Move the data and target to the device
            data, target = data.to(device), target.to(device)
            # Forward pass
            output = model(data)
            # Get the predictions
            _, preds = torch.max(output, 1)
            # Loop over the predictions
            for i in range(len(preds)):
                # Add the image to the correct class image and increase the class count
                class_images[preds[i]] += data[i, 0].cpu().numpy()
                class_counts[preds[i]] += 1
    # Divide each class image by the class count to get the average image
    average_class_images = [image / count for image, count in zip(class_images, class_counts)]
    # Plot the average images
    fig, axs = plt.subplots(1, len(class_names), figsize=(20, 3))
    for i in range(len(class_names)):
        axs[i].imshow(average_class_images[i], cmap="viridis")
        axs[i].set_title(class_names[i])
        axs[i].axis("off")
    plt.show()