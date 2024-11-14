import datetime
import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


def save_model_and_history(
    model: torch.nn.Module, 
    history: Dict[str, any],
    filename: str
) -> None:
    """
    Save the model parameters and training history to a JSON file.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        history (dict): Dictionary containing the training history.
        filename (str): The filename to save the model and history to.
            If the filename already has an extension, the date will be
            appended before the extension. If there is no extension,
            '.json' will be appended to the filename.

    Returns:
        None
    """
    # Convert tensors in model state dict to NumPy arrays
    model_parameters = {key: convert_to_json_serializable(value) for 
                        key, value in model.state_dict().items()}
    
    # Convert tensors in history to NumPy arrays
    history = {key: convert_to_json_serializable(value) for key, value in history.items()}

    # Create a dictionary to store both model parameters and training history
    model_and_history = {
        'model_parameters': model_parameters,
        'history': history
    }
    
    # Get the current date and time
    current_time = datetime.datetime.now()
    
    # Format the date and time as a string
    date_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Get the absolute path to the file 
    absolute_path = os.path.abspath(filename)
    
    # Remove extension from filename, if it is present
    if '.' in absolute_path:
        absolute_path, ext = absolute_path.rsplit('.', 1)
    else:
        ext = 'json'
    
    # Add date_string to filename and appena extension
    absolute_path = f"{absolute_path}_{date_string}.{ext}"
    
    # Save the combined dictionary to a JSON file
    with open(absolute_path, 'w') as file:
        json.dump(model_and_history, file, default=str) 
        # Use 'default=str' to handle non-serializable types
        
def convert_to_json_serializable(value):
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
    elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
        # Move each tensor in the list to CPU and convert to NumPy array
        value = [item.cpu().detach().numpy() for item in value]
    
    if isinstance(value, np.ndarray):
        return value.tolist()  # Convert NumPy array to nested Python list
    return value  # Keep other types unchanged

def plot_history(history, **kwargs):
    """
    Plot the training and validation losses for each stage.

    Args:
        history (dict): Dictionary containing the training and validation
            losses for each stage.
        **kwargs: Additional keyword arguments specifying which metrics to plot.
            Each keyword argument should be a boolean indicating whether to
            plot the corresponding metric.
    """

    plt.figure(figsize=(10, 6))

    for key, value in history.items():
        if kwargs.get(key, False):
            if isinstance(value, torch.Tensor):
                y = value.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
            elif isinstance(value, list) and all(
                isinstance(item, torch.Tensor) for item in value):
                # Move each tensor in the list to CPU and convert to NumPy array
                y = [item.cpu().detach().numpy() for item in value]  
            else:
                y = value  # Keep other types unchanged
            
            epochs = range(1, len(y) + 1) # Generate epoch nums starting from 1
            plt.plot(epochs, y, label=key.replace('_', ' ').title())

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_latent_space(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    use_tsne: bool = False, 
    label_column: int = 0, 
    figsize: Tuple[int, int] = (10, 10),
    **kwargs: Any
) -> None:
    """
    Plot the latent space variables of the two-stage VAE.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the
        dataset. 
        use_tsne (bool): Whether to use t-SNE to reduce dimensionality (default
        is False). 
        label_column (int): Index of the column to use as labels for coloring
        (default is 0). 
        figsize (Tuple[int, int]): Size of the figure (default is (10, 10)).
        **kwargs: Additional keyword arguments for scatter plot parameters
        (e.g., marker, cmap). 

    Returns:
        None
    """
    model.eval()  # Set the model to evaluation mode
    latent_space = []
    labels = []

    # Extract latent space variables for each sample in the dataset
    with torch.no_grad():
        for x in data_loader:
            x = x.to(model.device)
            mu_z, logvar_z = model.encoder(x)
            sigma_z = torch.exp(0.5 * logvar_z)
            z = mu_z + sigma_z * torch.randn_like(sigma_z)
            latent_space.append(z.cpu().numpy())
            
            # Extract labels from specified column
            labels.append(x[:, label_column].cpu().numpy()) 

    latent_space = np.concatenate(latent_space, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Apply dimensionality reduction
    if use_tsne:
        reducer = TSNE(n_components=2, random_state=0)
        reduction_method = 't-SNE'
    else:
        reducer = PCA(n_components=2, random_state=0)
        reduction_method = 'PCA'

    latent_space = reducer.fit_transform(latent_space)

    # Plot the latent space variables
    plt.figure(figsize=figsize)
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels, **kwargs)
    plt.colorbar(label=f'data_loader [:, {label_column}]')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Latent Space Variables ({reduction_method})')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_losses(history, training_epochs, start_epoch=1):
    """
    Plot training and optionally validation losses over epochs.

    Args:
    - history (dict): A dictionary containing training and optionally validation
      loss values. 
    - training_epochs (int): Total number of training epochs.
    - start_epoch (int, optional): The starting epoch for plotting. Defaults to
      1. 
    """
    plt.plot(range(start_epoch, training_epochs + 1), 
             history['train_loss'][start_epoch-1:], 
             label='Training Loss')
    
    if 'val_loss' in history and history['val_loss']:
        plt.plot(range(start_epoch, training_epochs + 1), 
                 history['val_loss'][start_epoch-1:], 
                 label='Validation Loss')
        
        plt.title('Training and Validation Loss Over Epochs')
    else:
        plt.title('Training Loss Over Epochs')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_test_vs_predicted_old(
    x_data, 
    test_outputs, 
    predicted_outputs, 
    x_feature,
    y_feature, 
    x_scaler=None,
    y_scaler=None,
    training_data=None, 
    **kwargs
):
    """
    Plot a scatter plot comparing test outputs and predicted outputs.

    Args:
      - x_data (pandas.DataFrame, numpy.ndarray, torch.Tensor): X values for the plot.
      - test_outputs (pandas.DataFrame, numpy.ndarray, torch.Tensor): Test outputs for the plot.
      - predicted_outputs (pandas.DataFrame, numpy.ndarray, torch.Tensor):
        Predicted outputs for the plot. 
      - x_feature (str): Name of the feature representing the X values.
      - y_feature (str): Name of the feature representing the Y values.
      - x_scaler (sklearn.preprocessing.StandardScaler, optional): Scaler object
        for scaling X Values. Defaults to None.
      - y_scaler (sklearn.preprocessing.StandardScaler, optional): Scaler object
        for scaling Y Values. Defaults to None.
      - training_data (pandas.DataFrame, numpy.ndarray, torch.Tensor, optional):
        Training data. 
      - **kwargs: Additional keyword arguments to customize the plot.
    """
    # Convert x_data, test_outputs, and predicted_outputs to NumPy arrays if
    # they are pandas DataFrames or torch Tensors 
    if isinstance(x_data, pd.DataFrame):
        x_data = x_data.values.flatten()
    elif isinstance(x_data, torch.Tensor):
        x_data = x_data.cpu().numpy().flatten()

    if isinstance(test_outputs, pd.DataFrame):
        test_outputs = test_outputs.values.flatten()
    elif isinstance(test_outputs, torch.Tensor):
        test_outputs = test_outputs.cpu().numpy().flatten()

    if isinstance(predicted_outputs, pd.DataFrame):
        predicted_outputs = predicted_outputs.values.flatten()
    elif isinstance(predicted_outputs, torch.Tensor):
        predicted_outputs = predicted_outputs.cpu().numpy().flatten()

    # If x_scaler is provided, apply scaling to x_data
    if x_scaler is not None:
        x_data = x_scaler.inverse_transform(x_data.reshape(-1, 1)).flatten()
        
    # If y_scaler is provided, apply scaling to test_outputs and predicted_outputs
    if y_scaler is not None:
        test_outputs = y_scaler.inverse_transform(test_outputs.reshape(-1, 1)).flatten()
        predicted_outputs = y_scaler.inverse_transform(predicted_outputs.reshape(-1, 1)).flatten()

    # Set default marker and color
    test_marker = kwargs.get('test_marker', 'o')
    test_color = kwargs.get('test_color', 'blue')
    predicted_marker = kwargs.get('predicted_marker', 's')
    predicted_color = kwargs.get('predicted_color', 'red')

    # Create plot
    plt.figure(figsize=kwargs.get('figsize', (8, 6)))
    
    if training_data is not None:
        # Convert training data to NumPy arrays if they are pandas DataFrames or torch Tensors 
        if isinstance(training_data, pd.DataFrame):
            training_data = training_data.values.flatten()
        elif isinstance(training_data, torch.Tensor):
            training_data = training_data.cpu().numpy().flatten()
        
        if x_scaler is not None:
            training_data = x_scaler.inverse_transform(training_data.reshape(-1, 1)).flatten()

        # Plot training data as a continuous line on plot background
        plt.plot(x_data, training_data, color='gray', linestyle='-', 
                 linewidth=0.5, alpha=0.5, label='Training Data')
    
    plt.scatter(x_data, test_outputs, color=test_color, label='Target Values', 
                marker=test_marker, alpha=0.5, s=8)
    plt.scatter(x_data, predicted_outputs, color=predicted_color, 
                label='Predicted Outputs', marker=predicted_marker, alpha=0.5, 
                s=8)

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title('Comparison of Target Values and Predicted Outputs')
    plt.legend()
    plt.grid(True)
    
    # Set x limit
    if 'xlim' in kwargs:
        plt.xlim(kwargs.get('xlim'))
    
    plt.show()
    
def plot_test_and_reconstructed(
    test_data: Union[np.ndarray, DataLoader], 
    recon_data: Union[np.ndarray, DataLoader], 
    scaler: Any, 
    train_data: Optional[Union[np.ndarray, DataLoader]] = None, 
    **kwargs
):
    """
    Plots the comparison between test data, reconstructed test data, and
    optionally train data.

    Parameters:
    - test_data (Union[np.ndarray, DataLoader]): The test data to plot.
    - recon_data (Union[np.ndarray, DataLoader]): The reconstructed test data to
      plot.
    - scaler (Any): The scaler used to inverse transform the data. It should
      have an `inverse_transform` method.
    - train_data (Optional[Union[np.ndarray, DataLoader]]): The train data to
      plot (if provided).
    - **kwargs: Additional keyword arguments for plot customization (e.g.,
      'figsize').

    Returns:
    None
    """
    
    # Helper function to check if input is DataLoader and extract data
    def get_data_from_input(data_input):
               
        # Check if input is a DataLoader and extract data from each batch
        if isinstance(data_input, DataLoader):
            # Initialize an empty list to store batches
            all_batches = []
            
            with torch.no_grad():
                for x in data_input:
                    x = x.cpu().detach().numpy()
                    all_batches.append(x)
            
            # Concatenate all batches along the first dimension
            all_data = np.concatenate(all_batches, axis=0)
            
        else:
            all_data = data_input
        
        # Apply inverse transformation to the concatenated data
        return scaler.inverse_transform(all_data)
    
    # Apply inverse transformation to test and reconstructed data
    X_test = get_data_from_input(test_data)
    X_recon = get_data_from_input(recon_data)
    
    # Extract x and y for test and reconstructed data
    x_test, y_test = X_test[:, 0], X_test[:, 1]
    x_recon, y_recon = X_recon[:, 0], X_recon[:, 1]
    
    # Create the figure object
    plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    
    # Apply inverse transform to train data, if provided
    if train_data is not None:
        X_train = get_data_from_input(train_data)
        
        # Extract x and y for train data and sort by x.Sorting is required to
        # plot the data as a line. 
        sorted_indices = np.argsort(X_train[:, 0])
        x_train, y_train = X_train[sorted_indices, 0], X_train[sorted_indices, 1]
        
        # Plot train data as a line with medium alpha
        plt.plot(x_train, y_train, label='Train Data', alpha=0.5)
        
    # Plot test data as scatter plot
    plt.scatter(x_test, y_test, label='Test Data', alpha=0.7, s=10, marker='x',
                c='C1')

    # Plot fake test data as scatter plot
    plt.scatter(x_recon, y_recon, label='Reconstructed Test Data', alpha=0.7,
                s=10, marker='1', c='C2')
    
    # Customize plot
    if train_data is not None:
        title_str = 'Comparison between Train, Test, and Reconstructed Test Data'
    else:
        title_str = 'Comparison Test and Reconstructed Test Data'
        
    plt.title(title_str)
    plt.xlabel('x [mm]')
    plt.ylabel('LTT [keV $\mu$m$^{{-1}}$]')
    plt.legend()
    plt.show()


def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Size: {param.size()}, "
                  f"Requires gradient: {param.requires_grad}")