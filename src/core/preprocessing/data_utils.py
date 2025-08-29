# src/ml_prep/data_utils.py

import logging
from typing import Any, Tuple, Union

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def train_val_test_split(
    *arrays: Union[np.ndarray, pd.DataFrame], 
    train_size: float=0.6, 
    val_size: float=0.2, 
    test_size: float=0.2, 
    random_state: int=None,
    shuffle: bool=True,
    preserve_df: bool = True 
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
    - preserve_df (bool, optional): Whether to preserve pandas DataFrames in the
      output splits (default is True).

    Returns:
      If a single input array is provided, returns a tuple:
        (train_split, val_split, test_split), each being a DataFrame (if 
        preserve_df is True) or an array.
        
      If multiple arrays are provided, returns a tuple of three lists:
        (train_splits, val_splits, test_splits), where each list contains the 
        corresponding split for each input array.
    """
    # If preserve_df is True, do not convert DataFrames to numpy arrays.
    if not preserve_df:
        arrays = [df.values if isinstance(df, pd.DataFrame) else df for df in arrays]
    
    # Check if at least one array is provided
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    
    # Check if all arrays have the same first dimension
    first_size = len(arrays[0])
    if not all(len(arr) == first_size for arr in arrays):
        raise ValueError("All input arrays must have the same size.")
    
    actual_sum = train_size + val_size + test_size
    if not np.isclose(actual_sum, 1.0, atol=1e-10):
        raise ValueError(f"The sum of train_size, val_size, and test_size must "
                         f"equal 1.0. Got {actual_sum:.10f}.")
    
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
    
    # If only one array is provided, return DataFrames/arrays directly.
    if len(arrays) == 1:
        arr = arrays[0]
        if preserve_df and isinstance(arr, pd.DataFrame):
            train_split = arr.iloc[indices[:train_end]]
            val_split = arr.iloc[indices[train_end:val_end]]
            test_split = arr.iloc[indices[val_end:]]
        else:
            train_split = arr[indices[:train_end]]
            val_split = arr[indices[train_end:val_end]]
            test_split = arr[indices[val_end:]]
        return train_split, val_split, test_split
    
    # If multiple arrays are provided, return lists.
    train_splits = []
    val_splits = []
    test_splits = []
    for arr in arrays:
        if preserve_df and isinstance(arr, pd.DataFrame):
            train_split = arr.iloc[indices[:train_end]]
            val_split = arr.iloc[indices[train_end:val_end]]
            test_split = arr.iloc[indices[val_end:]]
        else:
            train_split = arr[indices[:train_end]]
            val_split = arr[indices[train_end:val_end]]
            test_split = arr[indices[val_end:]]
        train_splits.append(train_split)
        val_splits.append(val_split)
        test_splits.append(test_split)
    
    return train_splits, val_splits, test_splits

def train_val_test_scale(
    training_data: Union[np.ndarray, pd.DataFrame], 
    validation_data: Union[np.ndarray, pd.DataFrame],
    test_data: Union[np.ndarray, pd.DataFrame], 
    scaler_type: str='standard',
    single_scaler: bool=True,
    **kwargs # support output_distribution argument for power and quantile
) -> tuple:
    """
    Apply a scaler to the train, validation, and test splits.

    Parameters:
    - training_data (array-like or pd.DataFrame): Training data.
    - validation_data (array-like or pd.DataFrame): Validation data.
    - test_data (array-like or pd.DataFrame): Testing data.
    - scaler_type (str, optional): Type of scaler to be used. Options are
      'standard', 'minmax', 'robust', 'power', or 'quantile' (default is
      'standard'). 
    - single_scaler (bool, optional): Whether to create a single scaler for all
      columns (True) or separate scalers for each column (False) (default is
      True).
    - **kwargs: Additional keyword arguments to be passed to the scaler

    Returns:
      A tuple containing:
        - training_data_scaled: Scaled training data, as a DataFrame if the input
          was a DataFrame.
        - validation_data_scaled: Scaled validation data, as a DataFrame if the input
          was a DataFrame.
        - test_data_scaled: Scaled test data, as a DataFrame if the input was a DataFrame.
        - scaler: A single scaler object if single_scaler is True, otherwise a list 
          of scaler objects.
    """
    # Determine if we should preserve DataFrame structure.
    preserve_df = isinstance(training_data, pd.DataFrame)
    if preserve_df:
        train_index = training_data.index
        train_columns = training_data.columns
        training_data_values = training_data.values
        validation_data_values = validation_data.values if isinstance(
            validation_data, pd.DataFrame) else validation_data 
        test_data_values = test_data.values if isinstance(
            test_data, pd.DataFrame) else test_data
    else:
        training_data_values = training_data
        validation_data_values = validation_data
        test_data_values = test_data
        
    # Choose the scaler based on the scaler_type
    if scaler_type == 'standard':
        scaler = StandardScaler(
            with_mean=kwargs.get('with_mean', True),
            with_std=kwargs.get('with_std',True)
        )
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(
            feature_range=kwargs.get('feature_range', (0, 1))
        )
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
        scaler.fit(training_data_values)
    else:
        # Create a list of scaler objects for each column
        scalers = [scaler for _ in range(training_data_values.shape[1])]
        # Fit each scaler to the corresponding column of the training data
        for i, scaler in enumerate(scalers):
            scaler.fit(training_data_values[:, i].reshape(-1, 1))
    
    
     # Scale the data.
    if single_scaler:
        training_scaled = scaler.transform(training_data_values)
        # Only transform if the array is non-empty.
        if validation_data_values.shape[0] > 0:
            validation_scaled = scaler.transform(validation_data_values)
        else: 
            validation_scaled = validation_data_values
        if test_data_values.shape[0] > 0:
            test_scaled = scaler.transform(test_data_values)
        else: 
            test_scaled = test_data_values
    else:
        training_scaled = np.hstack(
            [scaler.transform(training_data_values[:, i].reshape(-1, 1)) 
             for i, scaler in enumerate(scalers)]
        )
        validation_scaled = np.hstack(
            [scaler.transform(validation_data_values[:, i].reshape(-1, 1)) 
             for i, scaler in enumerate(scalers)]
        )
        test_scaled = np.hstack(
            [scaler.transform(test_data_values[:, i].reshape(-1, 1)) 
            for i, scaler in enumerate(scalers)]
        )
        
    # If the input was a DataFrame, convert the scaled arrays back to DataFrames.
    if preserve_df:
        training_data_scaled = pd.DataFrame(training_scaled, index=train_index,
                                            columns=train_columns)
        if isinstance(validation_data, pd.DataFrame):
            validation_data_scaled = pd.DataFrame(validation_scaled,
                                                  index=validation_data.index,
                                                  columns=validation_data.columns)
        else:
            validation_data_scaled = validation_scaled
        if isinstance(test_data, pd.DataFrame):
            test_data_scaled = pd.DataFrame(test_scaled,
                                            index=test_data.index,
                                            columns=test_data.columns)
        else:
            test_data_scaled = test_scaled
    else:
        training_data_scaled = training_scaled
        validation_data_scaled = validation_scaled
        test_data_scaled = test_scaled
    
    return training_data_scaled, validation_data_scaled, test_data_scaled, \
           (scaler if single_scaler else scalers)


def split_and_scale_dataset(
    let_df: pd.DataFrame,
    random_seed: int = None,
    preserve_df: bool = True,
    shuffle: bool = True,
    **kwargs: Any
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """
    Splits the dataset into training, validation, and test subsets and scales them.

    Parameters:
    - let_df (pd.DataFrame): Input dataset.
    - random_seed (int, optional): Random seed for reproducibility.
    - preserve_df (bool, optional): Whether to preserve pandas DataFrame format
      in output splits. 
    - shuffle (bool, optional): Whether to shuffle the dataset before splitting.
      Default is True. 

    Keyword Args:
      train_size (float): Proportion of data to use for training. Default is 0.7.
      val_size (float): Proportion of data to use for validation. Default is 0.2.
      test_size (float): Proportion of data to use for testing. Default is computed as 
                         1 - train_size - val_size.
      single_scaler (bool): Whether to use a single scaler for all splits.
                         Default is True. 

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, scaler]: Scaled training,
      validation, and test datasets, along with the scaler used. 
    """
    # Default training, validation, and test sizes
    train_size = kwargs.get('train_size', 0.7)
    val_size = kwargs.get('val_size', 0.2)
    test_size = kwargs.get('test_size', 1 - train_size - val_size)
    
    # Validate that train, val, and test sizes sum approximately to 1
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"Train, validation, and test sizes must sum to 1. "
                         f"Got {total}")
        
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Split the dataset into train, validation, and test subsets
    X_train, X_val, X_test = train_val_test_split(
        let_df,             # Input dataset
        train_size=train_size,     
        val_size=val_size,       
        test_size=test_size,      
        random_state=random_seed,   # Random seed for reproducibility
        shuffle=shuffle,        # Whether to shuffle the dataset before splitting
        preserve_df=preserve_df    # Preserve DataFrame if specified
    )
    
    # Default single scaler set (multiple scalers if set to False).
    single_scaler = kwargs.pop('single_scaler', True)  
    
    # Scale the datasets
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = \
        train_val_test_scale(
            training_data=X_train,
            validation_data=X_val,
            test_data=X_test,
            single_scaler=single_scaler,
            scaler_type=kwargs.pop('scaler_type', 'standard'),
            **kwargs
        )
        
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    
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
        raise ValueError("Number of features in the scaler must match the number "
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
    - data (array-like or pd.DataFrame): The input data.
    - data_type (str): Type of the input data. Options are 'train', 'val', or
      'test'. 

    Attributes:
    - data (Tensor): The data tensor for the selected dataset.
    - length (int): The length of the dataset.
    - col_mapping (dict, optional): Mapping from column names to indices, if the
      input data was a DataFrame.
    """
    def __init__(self, data, data_type):
        if isinstance(data, pd.DataFrame):
            self.data = torch.tensor(data.values, dtype=torch.float32)
            self.col_mapping = {col: idx for idx, col in enumerate(data.columns)}
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
            self.col_mapping = None
        self.length = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.length
    
def create_data_loader(
    data: Any, 
    data_type: str, 
    batch_size: int,
    shuffle: bool = True,
) -> _DataLoader:
    """
    Create a DataLoader object for the given data and attach the column mapping.

    Parameters:
      - data (Any): The input data.
      - data_type (str): Type of the input data. Options are 'train',
        'validation', or 'test'. 
      - batch_size (int): The batch size for the DataLoader.
      - shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: A DataLoader object for the given data with an attached 
        'col_mapping' attribute if available.
    """
    databuilder = DataBuilder(data, data_type=data_type)
    loader = _DataLoader(
        dataset=databuilder,
        batch_size=batch_size,
        shuffle=shuffle
    )
    # Attach column mapping from DataBuilder to DataLoader for downstream access.
    loader.col_mapping = databuilder.col_mapping
    return loader