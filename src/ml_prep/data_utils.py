from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset as _Dataset


def train_val_test_split(
    *arrays: Union[np.ndarray, pd.DataFrame], 
    train_size: float=0.6, 
    val_size: float=0.2, 
    test_size: float=0.2, 
    random_state: int=None,
    shuffle: bool=True
) -> tuple:
    """
    Split a dataset into train, validation, and test subsets.

    Parameters:
    - *arrays (sequence of indexables with same length / shape[0]): Allowed
      inputs are lists, numpy arrays, scipy-sparse matrices, or pandas
      dataframes.  
    - train_size (float, optional): Proportion of the dataset to include in the
      train split (default is 0.6). 
    - val_size (float, optional): Proportion of the dataset to include in the
      validation split (default is 0.2). 
    - test_size (float, optional): Proportion of the dataset to include in the
      test split (default is 0.2). 
    - random_state (int, optional): Random seed for reproducibility (default is
      None).
    - shuffle (bool, optional): Whether to shuffle the dataset before splitting
      (default is True). 


    Returns:
    - tuple: A tuple containing four elements:
        - X_train (array-like): Training data.
        - X_val (array-like): Validation data.
        - X_test (array-like): Testing data.
        - y_train (array-like): Labels for the training data.
        - y_val (array-like): Labels for the validation data.
        - y_test (array-like): Labels for the testing data.
    """
    # Convert DataFrames to numpy arrays
    arrays = [df.values if isinstance(df, pd.DataFrame) else df for df in arrays]
    
     # Check if at least one array is provided
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    
    # Check if all arrays have the same first dimension
    first_size = len(arrays[0])
    if not all(len(arr) == first_size for arr in arrays):
        raise ValueError("All input arrays must have the same size.")
    
    if not np.isclose(train_size + val_size + test_size, 1.0, atol=1e-10):
        raise ValueError("The sum of train_size, val_size, and test_size must "
                         "equal 1.0.")
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle the indices if required
    if shuffle:
        indices = np.random.permutation(first_size)
    else:
        indices = np.arange(first_size)
    
    # Calculate split points
    train_end = int(train_size * first_size)
    val_end = train_end + int(val_size * first_size)
    
    # Split data
    splits = []
    for arr in arrays:
        splits.append(arr[indices[:train_end]])
        splits.append(arr[indices[train_end:val_end]])
        splits.append(arr[indices[val_end:]])
    
    return tuple(splits)

def train_val_test_scale(
    training_data: np.ndarray, 
    validation_data: np.ndarray,
    test_data: np.ndarray, 
    scaler_type: str='standard',
    single_scaler: bool=True,
    **kwargs # support output_distribution argument for power and quantile
) -> tuple:
    """
    Apply a scaler to the train, validation, and test splits.

    Parameters:
    - training_data (array-like): Training data.
    - validation_data (array-like): Validation data.
    - test_data (array-like): Testing data.
    - scaler_type (str, optional): Type of scaler to be used. Options are
      'standard', 'minmax', 'robust', 'power', or 'quantile' (default is
      'standard'). 
    - single_scaler (bool, optional): Whether to create a single scaler for all
      columns (True) or separate scalers for each column (False) (default is
      True).
    - **kwargs: Additional keyword arguments to be passed to the scaler

    Returns:
    - tuple: A tuple containing three elements:
        - training_data_scaled (array-like): Scaled training data.
        - validation_data_scaled (array-like): Scaled validation data.
        - test_data_scaled (array-like): Scaled testing data.
        - scalers (list or scaler): List of scaler objects used for each column
          if single_scaler is False, otherwise a single scaler object.
    """
    # Choose the scaler based on the scaler_type
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'power':
        scaler = PowerTransformer(
            method = kwargs.get('method', 'yeo-johnson'),
            standardize= kwargs.get('standardize', True),
        )
    elif scaler_type == 'quantile':
        scaler = QuantileTransformer(
            n_quantiles= kwargs.get('n_quantiles', 1000),
            output_distribution = kwargs.get('output_distribution', 'uniform')
        )
    else:
        raise ValueError("Invalid scaler_type. Use 'standard', 'minmax', or 'robust'.")
    
    if single_scaler:
        # Fit the scaler to the training data
        scaler.fit(training_data)
    else:
        # Create a list of scaler objects for each column
        scalers = [scaler for _ in range(training_data.shape[1])]
        # Fit each scaler to the corresponding column of the training data
        for i, scaler in enumerate(scalers):
            scaler.fit(training_data[:, i].reshape(-1, 1))
    
    
     # Scale the data using the fitted scaler(s)
    if single_scaler:
        training_data_scaled = scaler.transform(training_data)
        validation_data_scaled = scaler.transform(validation_data)
        test_data_scaled = scaler.transform(test_data)
    else:
        training_data_scaled = np.hstack(
            [scaler.transform(training_data[:, i].reshape(-1, 1)) 
             for i, scaler in enumerate(scalers)]
        )
        validation_data_scaled = np.hstack(
            [scaler.transform(validation_data[:, i].reshape(-1, 1)) 
             for i, scaler in enumerate(scalers)]
        )
        test_data_scaled = np.hstack(
            [scaler.transform(test_data[:, i].reshape(-1, 1)) 
            for i, scaler in enumerate(scalers)]
        )
    
    return (
        training_data_scaled, 
        validation_data_scaled, 
        test_data_scaled, 
        scaler if single_scaler else scalers
    )
    
    
def inverse_transform_with_scalers(scaled_data, scalers):
    """
    Apply inverse_transform with a list of scalers.

    Parameters:
    - scaled_data (array-like): Scaled data.
    - scalers (list): List of scaler objects used for scaling.

    Returns:
    - array-like: Inverse transformed data.
    """
    
    n_scalers = len(scalers) if isinstance(scalers, list) else 1
    
    # Check if the number of columns in scaled_data matches the number of
    # scalers
    if n_scalers != scaled_data.shape[1] and n_scalers != 1:
        raise ValueError("Number of columns in scaled data must match the number "
                         "of scalers or there should be exactly one scaler.")
    elif n_scalers == 1 and scalers.n_features_in_ != scaled_data.shape[1]:
        raise ValueError("Number of features in the scaler must match the number "
                         "of columns in scaled data.")
    
    if n_scalers == 1:
        inverse_transformed_data = scalers.inverse_transform(scaled_data)
    else:
        # Apply inverse_transform for each scaler on the corresponding columns
        inverse_transformed_data = np.hstack([
            scaler.inverse_transform(scaled_data[:, i].reshape(-1, 1))
            for i, scaler in enumerate(scalers)
        ])
    
    return inverse_transformed_data

def inverse_transform_tensor(tensor, scalers):
    """
    Apply inverse_transform to a PyTorch tensor.

    Parameters:
    - tensor (torch.Tensor): Input PyTorch tensor.
    - scalers (list): List of scaler objects used for scaling.

    Returns:
    - torch.Tensor: Inverse transformed tensor.
    """
    
    n_scalers = len(scalers) if isinstance(scalers, list) else 1
    
    # Check if the number of columns in the tensor matches the number of scalers
    # or if there's only one scaler
    if tensor.size(1) != n_scalers and n_scalers != 1:
        raise ValueError("Number of columns in the tensor must match the number "
                         "of scalers.")
    elif n_scalers == 1 and scalers.n_features_in != tensor.size(1):
        raise ValueError("Number of features in the scaler must match the number "
                         "of columns in the tensor.")
    
    # Convert tensor to numpy array
    tensor_array = tensor.detach().cpu().numpy()
    
    # Apply inverse_transform for each scaler on the corresponding columns
    inverse_transformed_array = np.hstack([
        scaler.inverse_transform(tensor_array[:, i].reshape(-1, 1))
        for i, scaler in enumerate(scalers)
    ])
    
    # Convert the result back to a PyTorch tensor
    inverse_transformed_tensor = torch.tensor(inverse_transformed_array, dtype=torch.float32)
    
    return inverse_transformed_tensor

def inverse_transform_data_loader(data_loader, scalers):
    """
    Apply inverse_transform to a DataLoader object.

    Parameters:
    - data_loader (DataLoader): PyTorch DataLoader object.
    - scalers (list): List of scaler objects used for scaling.

    Returns:
    - torch.Tensor: Inverse transformed data tensor.
    """
    
    n_scalers = len(scalers) if isinstance(scalers, list) else 1
    
    # Check if the number of columns in data_loader.data matches the number of
    # scalers or if there's only one scaler
    if data_loader.dataset.data.shape[1] != n_scalers and n_scalers != 1:
        raise ValueError("Number of columns in data_loader.data must match the "
                         "number of scalers.")
    elif n_scalers == 1 and data_loader.dataset.data.shape[1] != scalers.n_features_in:
        raise ValueError("Nnumber of features in the scaler must match the number "
                         "of columns in data_loader.data.")
        
    
    # Concatenate all batches along the batch dimension
    data = torch.cat([batch for batch in data_loader], dim=0)
    
    # Convert data to numpy array
    data_array = data.numpy()
    
    # Apply inverse_transform for each scaler on the corresponding columns
    inverse_transformed_data = np.hstack([
        scaler.inverse_transform(data_array[:, i].reshape(-1, 1))
        for i, scaler in enumerate(scalers)
    ])
    
    # Convert the result back to a PyTorch tensor
    inverse_transformed_tensor = torch.tensor(inverse_transformed_data, dtype=torch.float32)
    
    return inverse_transformed_tensor

class DataBuilder(_Dataset):
    """
    PyTorch Dataset class for X_train, X_val, or X_test.

    Args:
    - data (array-like): The input data.
    - data_type (str): Type of the input data. Options are 'train', 'val', or
      'test'. 

    Attributes:
    - data (Tensor): The data tensor for the selected dataset.
    - length (int): The length of the dataset.
    """
    def __init__(self, data, data_type):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.length = len(data)
        self.data_type = data_type

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length
    
def create_data_loader(
    data: Any, 
    data_type: str, 
    batch_size: int,
    shuffle: bool = True,
) -> _DataLoader:
    """
    Create a DataLoader object for the given data.

    Parameters:
    - data (Any): The input data.
    - data_type (str): Type of the input data. Options are 'train',
      'validation', or 'test'. 
    - batch_size (int): The batch size for the DataLoader.
    - shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: DataLoader object for the given data.
    """
    databuilder = DataBuilder(data, data_type=data_type)
    loader = _DataLoader(
        dataset=databuilder,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader