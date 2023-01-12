White Noise Analysis of Neural Networks
=======================================


![](https://github.com/the-infiltrator/White-Noise-Analysis-for-Neural-Networks/blob/master/fmnist_noisepreds.png?raw=true)

Average noise maps for varying levels of gamma and their corresponding predicted values for a  CNN trained on the `fmnist` dataset. 


-------------
This project implements a white noise analysis of modern deep neural networks in order to unveil their biases at the whole network level or the single neuron level. The analysis is based on two popular and related methods in psychophysics and neurophysiology namely classification images and spike triggered analysis. These methods have been widely used to understand the underlying mechanisms of sensory systems in humans and monkeys and are leveraged in this project to investigate the inherent biases of deep neural networks and to obtain a first-order approximation of their functionality.

The project focuses on CNNs as they are currently the state of the art methods in computer vision and are a decent model of human visual processing. In addition, the project also studies multi-layer perceptrons, logistic regression, and recurrent neural networks. Experiments are conducted over four classic datasets: MNIST, and Fashion-MNIST. The results show that the computed bias maps resemble the target classes and when used for classification lead to an over two-fold performance than the chance level.

The project also demonstrates that classification images can be used to attack a black-box classifier and to detect adversarial patch attacks. Furthermore, the project utilizes spike triggered averaging to derive the filters of CNNs and explores how the behavior of a network changes when neurons in different layers are modulated.

Run the Code: [![Open In Colab](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1CAFls1NWZkZEZYkvSXkYHqB1xSU5dJhz?usp=sharing)
=======================================
Dependencies
=======================================

Here is a list of dependencies for this project:

-   Python 3
-   PyTorch
-   torchvision
-   NumPy
-   Matplotlib
-   sklearn


Usage
=======================================

This project contains three main modules: `utils`, `model`, and `analysis` and provides several functions for performing white noise analysis of neural networks and understanding the inherent biases of deep neural networks. The main functions are contained in the `analysis.py` module, and are:

-   `calculate_noise_maps`: This function takes in a trained model, a list of class names, noise samples, and a list of gamma values. It then calculates the classification images and average noise maps for each value of gamma.
-   `visualise_classification_images`: This function takes in a trained model, a list of class names, a test set and a list of gamma values, and visualizes the classification images for different gamma values.
-   `visualise_average_noisemaps`: This function takes in a trained model, a list of class names, a test set, a list of gamma values, and visualizes the average noise maps for different gamma values.
-   `visualize_activations`: This function takes in the mean activations of the first and last convolutional layers, and visualizes them
-   `spike_triggered_analysis`: This function takes in a trained model, a test set, a list of class names, and a list of gamma values. It analyses the CNNs and derives the filters of CNNs and explore how the behavior of a network changes when neurons in different layers are modulated.

The utility functions are contained in the `utils.py` module, and are:

-   `get_data_loader`: This function takes in a dataset, batch size, shuffle, number of workers and pin_memory flag, and returns data loader for that dataset
-   `visualise_preds`: This function takes in a trained model, a dataloader and class_names and visualizes the predictions

The models functions are contained in the `models.py` module, and are:

-   `count_parameters`: This function returns the total number of parameters in a model.
-   `get_model`: This function returns the different models as per requirement (e.g. simple MNIST, complex MNIST, simple Fashion-MNIST, complex Fashion-MNIST)

    To train a model, you can use the `train` function in the `model` module. This function takes in the following arguments:

    -   `model`: The model to be trained (a PyTorch nn.Module)
    -   `train_loader`: A PyTorch DataLoader for the training data
    -   `valid_loader`: A PyTorch DataLoader for the validation data
    -   `num_epochs`: Number of training epochs (default=100)
    -   `early_stopping_patience`: Patience for early stopping (default=5)
    -   `lr_min`: Minimum learning rate for cosine annealing schedule (default=1e-4)
    -   `lr_max`: Maximum learning rate for cosine annealing schedule (default=1e-2)
    -   `weight_decay`: Weight decay coefficient (default=1e-4)


Example
=======================================
```python
# Import the necessary modules
from utils import get_data_loader
from model import get_model, train
from analysis import visualise_classification_images, spike_triggered_analysis

# Get the data loaders
train_loader, valid_loader, test_loader = get_data_loader(batch_size=64)

# Define the model and number of classes
model = get_model("complex_mnist", input_shape=(1, 28, 28), num_classes=10)
num_classes = 10

# Train the model
train(model, train_loader, valid_loader, num_epochs=30)

# Visualize the classification images
gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0]
class_names = [str(i) for i in range(10)]
visualise_classification_images(model, class_names, test_loader, gamma_values)

# Perform spike-triggered analysis
spike_triggered_analysis(model, test_loader, class_names, gamma_values)

```