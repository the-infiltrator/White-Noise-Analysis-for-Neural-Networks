import torch.optim as optim
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, List


def calculate_noise_maps(model, class_names, noise_samples, sample_images, gamma):
    """Calculates classification images and average noise maps.
    Returns:
        A tuple containing the classification images and average noise maps.
    """
    num_classes = len(class_names)
    # Calculate the linear combination of the sample images and the white noise
    # classification_images = (gamma * sample_images[:,0,:,:,:]) + ((1 - gamma) * noise_samples[:num_classes].cpu().numpy())
    sample_images = [x[0] for x in sample_images]
    classification_images = (gamma * np.array(sample_images[:num_classes])) + ((1 - gamma) * noise_samples[:num_classes].cpu().numpy())

    # Calculate the average noise maps
    average_noise_maps = []
    for c in range(num_classes):
      # Select the classification images for the current class
      class_classification_images = torch.from_numpy(classification_images[c])

      # Calculate the average noise map for the current class
      average_noise_map = class_classification_images.mean(dim=0)
      average_noise_maps.append(average_noise_map)

    # Convert the average noise maps to a tensor
    # average_noise_maps = torch.stack(list(map(torch.from_numpy, average_noise_maps))).cpu()
    # Convert the average noise maps to a NumPy array
    average_noise_maps = torch.stack(average_noise_maps).cpu().numpy()
    return classification_images, average_noise_maps
    
def generate_samples(num_classes, test_set):
    sample_images = []
    labels = []
    for c in range(num_classes):
        image, label = list(test_set)[c]
        sample_images.append(image[None, :, :])
        labels.append(label)
    sample_images = torch.cat(sample_images).cpu().numpy()
    return sample_images, labels

def visualise_classification_images(model, class_names, test_set, gamma_values):
    """Visualizes the classification images for different gamma values.

    Args:
        model: The trained CNN model.
        class_names: A list of class names for the dataset.
        sample_images: A list of sample images for each class.
        gamma_values: A list of gamma values to generate classification images for.
    """
    num_classes = len(class_names)
    sample_images, labels = generate_samples(num_classes ,test_set)
    num_gamma_values = len(gamma_values)

    # Set the figure size
    fig = plt.figure(figsize=(25, 10))
    gs = fig.add_gridspec(num_gamma_values, num_classes+1)
    num_noise_samples = 5000
    noise_samples = torch.randn((num_noise_samples, 1, 28, 28), device=device)
    noise_labels = torch.randint(num_classes, (num_noise_samples,), device=device)

    for i, gamma in enumerate(gamma_values):
      classification_images, average_noise_maps = calculate_noise_maps(model, class_names, noise_samples, sample_images, gamma)
      # Feed the average noise maps back into the CNN

      # Add Gamma value label
      label_ax = fig.add_subplot(gs[i,0])
      label_ax.text(0.5, 0.5, f"Gamma = {round(gamma,2)}", horizontalalignment='center', verticalalignment='center', transform=label_ax.transAxes)
      label_ax.axis("off")

      for j in range(num_classes):
          img_ax = fig.add_subplot(gs[i, j+1])
          img_ax.imshow(classification_images[j][0], cmap="viridis")
          img_ax.set_title(f"{class_names[labels[j][0]]}")
          img_ax.axis("off")

    # Show the plot
    plt.show()

def visualise_average_noisemaps(model, class_names, test_set, gamma_values, title=None):
    """Visualizes the average noise maps for different gamma values.

    Args:
        model: The trained CNN model.
        class_names: A list of class names for the dataset.
        test_set: The test set for the dataset.
        gamma_values: A list of gamma values to generate average noise maps for.
        title: The title of the plot
    """
    num_classes = len(class_names)
    sample_images, labels = generate_samples(num_classes ,test_set)
    num_gamma_values = len(gamma_values)

    # Set the figure size
    fig = plt.figure(figsize=(30, 10))
    gs = fig.add_gridspec(num_gamma_values, num_classes+1)
    num_noise_samples = 5000
    noise_samples = torch.randn((num_noise_samples, 1, 28, 28), device=device)
    noise_labels = torch.randint(num_classes, (num_noise_samples,), device=device)

    for i, gamma in enumerate(gamma_values):
      classification_images, average_noise_maps = calculate_noise_maps(model, class_names, noise_samples, sample_images, gamma)
      average_noise_maps = torch.from_numpy(average_noise_maps).unsqueeze(0).permute(1,0,2,3)
      average_noise_maps_predictions = torch.argmax(model(average_noise_maps), dim=1)

      # Add Gamma value label
      label_ax = fig.add_subplot(gs[i,0])
      label_ax.text(0.5, 0.5, f"Gamma = {round(gamma,2)}", horizontalalignment='center', verticalalignment='center', transform=label_ax.transAxes)
      label_ax.axis("off")
 
      for j in range(num_classes):
          img_ax = fig.add_subplot(gs[i, j+1])
          img_ax.imshow(average_noise_maps[j][0], cmap="viridis")
          img_ax.set_title(f"Predicted: {class_names[average_noise_maps_predictions[j].item()]} (True: {class_names[labels[j][0]]})", fontsize=8)
          img_ax.axis("off")
    # Show the plot
    # plt.title(title)
    plt.show()

def visualize_activations(mean_first_conv_activation, mean_last_conv_activation, title=None):
    """Visualizes the mean activations of the first and last conv layers.

    Args:
        mean_first_conv_activation: The mean activations of the first conv layer.
        mean_last_conv_activation: The mean activations of the last conv layer.
        title: The title of the plot
    """
    num_classes = mean_first_conv_activation.shape[0]

    # Set the figure size
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(1, 2)

    # Visualize the mean activations of the first conv layer
    first_conv_ax = fig.add_subplot(gs[0, 0])
    first_conv_ax.imshow(mean_first_conv_activation, cmap="viridis")
    first_conv_ax.set_title("Mean Activations - First Conv Layer")
    first_conv_ax.axis("off")

    # Visualize the mean activations of the last conv layer
    last_conv_ax = fig.add_subplot(gs[0, 1])
    last_conv_ax.imshow(mean_last_conv_activation, cmap="viridis")
    last_conv_ax.set_title("Mean Activations - Last Conv Layer")
    last_conv_ax.axis("off")

    plt.title(title)
    # Show the plot
    plt.show()

def spike_triggered_analysis(model, test_set, class_names, gamma_values):
    num_classes = len(class_names)
    sample_images, labels = generate_samples(num_classes ,test_set)
    num_gamma_values = len(gamma_values)
    num_noise_samples = 5000
    noise_samples = torch.randn((num_noise_samples, 1, 28, 28), device=device)

    for i, gamma in enumerate(gamma_values):
        classification_images, average_noise_maps = calculate_noise_maps(model, class_names, noise_samples, sample_images, gamma)
        average_noise_maps = torch.from_numpy(average_noise_maps).unsqueeze(0).permute(1,0,2,3)
        average_noise_maps_predictions = torch.argmax(model(average_noise_maps), dim=1)

        #Make a forward pass through the entire model
        first_conv_activations = None
        last_conv_activations = None
        current_conv_layer = None
        for name, module in model.named_children():
            if type(module) == torch.nn.Conv2d:
                current_conv_layer = module
            if first_conv_activations is None and current_conv_layer is not None:
                first_conv_activations = current_conv_layer(average_noise_maps)
            if type(module) != torch.nn.Conv2d and current_conv_layer is not None:
                last_conv_activations = current_conv_layer(average_noise_maps)
                
        if first_conv_activations is None or last_conv_activations is None:
            raise ValueError("Model does not contain any 2D Convolutional layers")
            
        #collect the activation vector of the predicted classes
        first_conv_activation_vectors = [first_conv_activations[i] for i in range(num_classes) if average_noise_maps_predictions[i] == labels[i][0]]
        last_conv_activation_vectors = [last_conv_activations[i] for i in range(num_classes) if average_noise_maps_predictions[i] == labels[i][0]]

    # average the vectors across each class
    mean_first_conv_activation = sum(first_conv_activation_vectors) / len(first_conv_activation_vectors)
    mean_last_conv_activation = sum(last_conv_activation_vectors) / len(last_conv_activation_vectors)

    # Print the necessary information for each value of gamma
    print(f"Gamma: {round(gamma,2)}, Mean first conv layer activation: {mean_first_conv_activation}, Mean last conv layer activation: {mean_last_conv_activation}")

    # visualise the mean first and last conv activation as done in section 3.3
    visualize_activations(mean_first_conv_activation, mean_last_conv_activation)




