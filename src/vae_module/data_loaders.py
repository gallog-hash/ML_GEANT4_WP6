import sys

sys.path.append("../src")

import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def split_and_standardize_data(data_df, scaler_type='standard', random_state=42,
                               create_validation=True):
    """
    Split and standardize the input DataFrame.

    Parameters:
    - data_df (DataFrame): The input DataFrame to be split and standardized.
    - scaler_type (str, optional): Type of scaler to be used. Options are
      'standard' (default), 'minmax', or 'robust'.
    - random_state (int, optional): Random seed for reproducibility.
    - create_validation (bool, optional): Whether to create a validation dataset
      or not. Default is True.

    Returns:
    - tuple: A tuple containing three elements:
        - X_train (numpy.ndarray): Standardized training data.
        - X_val (numpy.ndarray, optional): Standardized validation data if 
          create_validation is True.
        - X_test (numpy.ndarray): Standardized testing data.
        - scaler (StandardScaler, MinMaxScaler, or RobustScaler): The selected
          scaler used for standardization.
    """

    # Convert the DataFrame into a 2D NumPy array
    data_np = data_df.values.reshape(-1, data_df.shape[1]).astype('float32')
    
    # Split the data into train, validation, and testing sets
    X_train, X_test = train_test_split(data_np, test_size=0.2, shuffle=True,
                                       random_state=random_state)
    if create_validation:
        X_val, X_test = train_test_split(X_test, test_size=0.5, shuffle=True,
                                         random_state=random_state)    
    
    # Convert scaler_type to lowercase for case-insensitivity
    scaler_type_lower = scaler_type.lower()
    
    # Choose the scaler based on the user's selection
    if scaler_type_lower == 'standard':
        scaler = preprocessing.StandardScaler()
    elif scaler_type_lower == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif scaler_type_lower == 'robust':
        scaler = preprocessing.RobustScaler()
    else:
        raise ValueError("Invalid scaler_type. Use 'standard', 'minmax', or 'robust'.")
    
    # Fit the scaler on the training data and transform it
    X_train = scaler.fit_transform(X_train)
    
    # Transform the validation and test data using the same scaler
    X_test = scaler.transform(X_test)
    if create_validation:
        X_val = scaler.transform(X_val)
    
    # Return the standardized training data, testing data, and the selected
    # scaler
    if create_validation:
        return X_train, X_val, X_test, scaler
    else:
        return X_train, X_test, scaler


class DataBuilder(Dataset):
    """
    PyTorch Dataset class for building train, validation, or test datasets.

    Args:
        - df (DataFrame): The input DataFrame containing the data.
        - train (bool, optional): Whether to build the train dataset (default is
          True). 
        - scaler (str, optional): Type of scaler to be used. Options are
          'standard',  'minmax', or 'robust' (default is 'standard').
        - random_state (int, optional): Random seed for reproducibility (default
          is 42). 
        - create_validation (bool, optional): Whether to create a validation
          dataset (default is True). 

    Attributes:
        - data (Tensor): The data tensor for the selected dataset.
        - length (int): The length of the dataset.

    """
    def __init__(self, df, train=True, scaler='standard', random_state=42,
                 create_validation=False):
        if create_validation:
            X_train, X_val, X_test, self.standardizer = \
                split_and_standardize_data(df, scaler_type=scaler,
                                            random_state=random_state,
                                            create_validation=create_validation)
        else:
            X_train, X_test, self.standardizer = \
                split_and_standardize_data(df, scaler_type=scaler,
                                            random_state=random_state,
                                            create_validation=create_validation)

        self.data = torch.from_numpy(X_train if train else 
                                  (X_test if not create_validation else X_val))
        
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


def create_data_loaders(df, batch_size, random_state=42, create_validation=False):
    """
    Create DataLoader objects for the train, test, and optionally validation datasets.

    Args:
        - df (DataFrame): The input DataFrame containing the data.
        - batch_size (int): The batch size for DataLoader objects.
        - random_state (int, optional): Random seed for reproducibility (default
          is 42). 
        - create_val_dataset (bool, optional): Whether to create a validation
          dataset (default is True). 

    Returns:
       -  tuple: A tuple containing three DataLoader objects (train_loader,
          test_loader, val_loader). The val_loader is None if create_val_dataset
          is False. 

    """
    train_data_set = DataBuilder(df, train=True, random_state=random_state,
                                 create_validation=create_validation)
    test_data_set = DataBuilder(df, train=False, random_state=random_state,
                                create_validation=create_validation)
    
    val_data_set = None
    if create_validation:
        val_data_set = DataBuilder(df, train=False, random_state=random_state,
                                   create_validation=create_validation)

    train_loader = DataLoader(dataset=train_data_set, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_data_set, batch_size=batch_size)
    val_loader = (DataLoader(dataset=val_data_set, batch_size=batch_size) if 
                  val_data_set else None)

    return train_loader, test_loader, val_loader