import sys as _sys
from typing import List, Optional, Tuple, Union

_sys.path.append("../../src")

import numpy as _np
import pandas as _pd
from my_plot_pkg.plot_utils import plot_let_outliers_replacements
from scipy.stats import zscore as _zscore
from sklearn.cluster import DBSCAN as _DBSCAN
from sklearn.neighbors import KDTree as _KDTree
from sklearn.neighbors import KNeighborsRegressor as _KNeighborsRegressor
from sklearn.neighbors import LocalOutlierFactor as _LocalOutlierFactor

# ==============================================
# === OUTLIER IDENTIFICATION AND REPLACEMENT ===
# ==============================================

def search_neighbors(
    query_dataset: Union[_pd.DataFrame, _np.ndarray],
    reference_dataset: Union[_pd.DataFrame, _np.ndarray],
    n_neighbors: int,
) -> list:
    """
    Find the indices of the nearest neighbors in a reference dataset
    for each point in a query dataset.

    Parameters:
    - query_dataset (pd.DataFrame or np.ndarray): The dataset containing
      points for which nearest neighbors are to be found.
    - reference_dataset (pd.DataFrame or np.ndarray): The dataset containing
      points to search for nearest neighbors in.
    - n_neighbors (int): The number of nearest neighbors to find.

    Returns:
    - list: A list where each element corresponds to a point in the query
      dataset and contains the indices of its nearest neighbors in the reference
      dataset.
    """
    # Check if n_neighbors is a positive integer
    if not isinstance(n_neighbors, int) or n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer.")
    
    # If needed, convert DataFrames to NumPy arrays for compatibility with
    # KDTree
    if isinstance(query_dataset, _pd.DataFrame):
        query_df = query_dataset.copy()
        query_dataset = query_dataset.values
    if isinstance(reference_dataset, _pd.DataFrame):
        reference_points = reference_dataset.values
        reference_df_indices = reference_dataset.index
        
    # Remove query points from reference dataset
    query_set = set(map(tuple, query_dataset))
    reference_points = [row for row in reference_points if tuple(row) not in query_set]
    if isinstance(reference_dataset, _pd.DataFrame):
        reference_dataset = reference_dataset.loc[~reference_dataset.index.isin(
            query_df.index)]
        reference_df_indices = reference_dataset.index
            
    # Build the K-D Tree from the reference dataset
    kdtree = _KDTree(reference_points)
    
    # Query the K-D Tree for each point in the query dataset
    nn_indices = []
    for i, row in enumerate(query_dataset):
        # Query the K-D Tree to find the indices of the N nearest neighbors for
        # the current point
        _, indices = kdtree.query([row], k=n_neighbors)
        
        # Convert indices to indices of original DataFrame if reference_dataset
        # is a DataFrame 
        if isinstance(reference_dataset, _pd.DataFrame):
            indices = [reference_df_indices[idx] for idx in indices[0]]
            indices = _np.array(indices).reshape(1, -1)
            
        nn_indices.append(indices)
        
    return nn_indices

def nearest_neighbor_replacement(
    outliers_df: _pd.DataFrame, 
    data: _pd.DataFrame, 
) -> List[float]:
    """
    Replace outlier values in outliers_df with the LET value of their nearest
    neighbor in data. 

    Parameters:
    - outliers_df (pd.DataFrame): DataFrame containing outlier data.
    - data (pd.DataFrame): DataFrame containing reference data.
    
    Returns:
    - List[float]: List of replacement values for outliers.
    """
    # Use search_neighbors to find index of nearest neighbor for each outlier
    nn_index = search_neighbors(
        query_dataset=outliers_df,
        reference_dataset=data,
        n_neighbors=1,
    )
    
    replacement_values = []
    
    # Iterate over each nearest neighbor index
    for outlier_neighbor_index in nn_index:
        outlier_neighbor_index = outlier_neighbor_index[0]
        
        # Get the LET value of the nearest neighbor and append it to
        # replacement_values 
        nn_let_value = data.loc[outlier_neighbor_index].iloc[0, 1]
        replacement_values.append(nn_let_value)
    
    return replacement_values

def local_statistic_outlier_replacement(
    outliers_df: _pd.DataFrame, 
    data: _pd.DataFrame, 
    n_neighbors: int,
    replace_method: str = 'mean',
) -> List[float]:
    """
    Replace outlier values in a DataFrame with the mean of their nearest
    neighbors. 

    Parameters:
    - outliers_df (pd.DataFrame): DataFrame containing the outliers to be
      replaced. 
    - data (pd.DataFrame): DataFrame containing the data points used for
      neighbor search. 
    - n_neighbors (int): Number of nearest neighbors to consider for
      replacement. 
    - replace_method: (str, optional): The statistical index to be used for
      calculating replacement value. Defaults to 'mean'.

    Returns:
    - List[float]: List of replacement values for each outlier.
    """
    # Validate replace_method
    if replace_method not in ['median', 'mean']:
        raise ValueError("Invalid replace_method. Use 'median', or 'mean'.")
    
    # Use search_neighbors to find indices of nearest neighbors for each outlier
    nn_indices = search_neighbors(
        query_dataset=outliers_df,
        reference_dataset=data,
        n_neighbors=n_neighbors,
    )
        
    replacement_values = []
    
    # Iterate over the list of neighbor indices for each outlier
    for outlier_neighbors_indices in nn_indices:
        outlier_neighbors_indices = outlier_neighbors_indices[0]
        
        # Calculate the replacement value for the current outlier
        if replace_method == 'median':
            replace_value = data.loc[outlier_neighbors_indices].iloc[:, 1].median()
        elif replace_method == 'mean':
            replace_value = data.loc[outlier_neighbors_indices].iloc[:, 1].mean()

        # Append the current value to the list of replacement values
        replacement_values.append(replace_value)
    
    return replacement_values

def knn_regressor_replacement(
    outliers_data: _pd. DataFrame, 
    data: _pd.DataFrame, 
    n_neighbors: int=5
) -> List[float]:
    """
    Use K-nearest neighbors regression to replace outlier values.

    Parameters:
    - outliers_data (pd.DataFrame): DataFrame containing outlier values.
    - data (pd.DataFrame): DataFrame containing the entire data.
    - n_neighbors (int, optional): Number of neighbors for KNN regressor.
      Defaults to 5.

    Returns:
    - List[float]: List of replacement values.
    """
    # Remove outliers_data indices from data
    data = data[~data.index.isin(outliers_data.index)]
    
    # Extract x and LET columns
    x_outliers = outliers_data.iloc[:, 0].values.reshape(-1, 1)

    # Fit KNN regressor
    knn_regressor = _KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(data.iloc[:, 0].values.reshape(-1, 1), data.iloc[:, 1].values)

    # Predict replacements
    replacement_values = knn_regressor.predict(x_outliers)

    return replacement_values

# main methods called in the following function are defined after it
def identify_let_outliers(
    df: _pd.DataFrame, 
    column_type: str = 'track', 
    x_bounds: tuple = None,
    outliers_method: str = 'upper_limit',
    replace_method: str = 'median',
    let_upper_limit: float = 100.0,
    lof_neighbors: int = 20,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    knn_neighbors: int = 5,  # Number of neighbors for KNN regressor
    cbar_label: str = 'Outlier Scores',
    verbose: bool = False,
    **kwargs
) -> _pd.DataFrame:
    """
    Identify outliers in LET profiles.

    Parameters:
    - df (DataFrame): The input DataFrame containing LET profiles.
    - column_type (str, optional): Type of columns to analyze ('track' or 'dose').
      Defaults to 'track'.
    - x_bounds (tuple, optional): Tuple specifying the lower and upper bounds of
      'x' values for which to evaluate z-scores and statistics. Defaults to
      None, meaning the entire range of 'x' values is considered.
    - outliers_method (str, optional): Method for identifying outliers ('upper_limit'
      or 'lof'). Defaults to 'upper_limit'.
    - replace_method (str, optional): Method for replacing outliers ('median',
      'mean', 'nearest_neighbor', or 'local_median', 'local_mean', or 
      'nearest_neighbor_regressor'). Defaults to 'median'. 
    - let_upper_limit (float, optional): Upper limit for LET values. Values
      greater than this limit are considered outliers. Defaults to 100.0.
    - lof_neighbors (int, optional): Number of neighbors for LOF calculation.
      Defaults to 20.
    - dbscan_eps (float, optional): The maximum distance between two samples
      for one to be considered as in the neighborhood of the other in DBSCAN.
      Defaults to 0.5.
    - dbscan_min_samples (int, optional): The number of samples (or total weight)
      in a neighborhood for a point to be considered as a core point in DBSCAN.
      Defaults to 5.
    - knn_neighbors (int, optional): Number of neighbors for KNN regressor.
      Defaults to 5.
    - cbar_label (str, optional): Label for the cbar. Defaults to 'Outlier
      Scores'. 
    - verbose (bool, optional): Whether to print verbose output. Defaults to
      False.
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title.
      
    Returns:
    - DataFrame: DataFrame containing the outliers identified based on the LET
      profiles and their LET replacement values. 
    """
    # Convert input parameters to lowercase for case insensitivity
    column_type = column_type.lower()
    outliers_method = outliers_method.lower()
    replace_method = replace_method.lower()

    # Validate column_type
    if column_type not in ['track', 'dose']:
        raise ValueError("Invalid column_type. Use 'track' or 'dose'.")
    
    # Validate outliers_method
    if outliers_method not in ['upper_limit', 'lof', 'dbscan']:
        raise ValueError("Invalid outliers_method. Use 'upper_limit', 'lof', "
                         "or 'dbscan'.")
    
    # Validate replace_method
    replace_method_valid = ['median', 'mean']
    replace_method_valid.extend(['local_' + method for method in replace_method_valid])
    replace_method_valid.extend(['nearest_neighbor', 'nearest_neighbor_regressor'])
    if replace_method not in replace_method_valid:
        raise ValueError("Invalid replace_method. Use: '" + 
                         "', '".join(replace_method_valid) + "'.")

    # Define search pattern based on column type to select LET profile columns
    search_pattern = 'LTT' if column_type.lower() == 'track' else 'LDT'

    # Select LET profile column based on column type
    let_column = df.filter(like=search_pattern)
    
    # Filter let_column and 'x' based on x_bounds if provided
    if x_bounds:
        x_min, x_max = x_bounds
        let_column_filtered = let_column[df['x'].between(x_min, x_max)]
        x_data_filtered = df['x'][df['x'].between(x_min, x_max)]
    else:
        let_column_filtered = let_column
        x_data_filtered = df['x']
    
    outliers_df, outlier_scores = outlier_detection(
        let_column_filtered, x_data_filtered, outliers_method, let_upper_limit,
        lof_neighbors, dbscan_eps, dbscan_min_samples
    )
    
    if outliers_df.empty:
        # Handle case where no outliers are found
        print(f"\nNo outliers were found by '{outliers_method}' method.")
        return outliers_df
    
    if outliers_df.size == let_column_filtered.size:
        pattern = outliers_method + '_'
        collected_vars = {k: v for k, v in locals().items() if k.startswith(pattern)}
        # Handle the case in which all let values are marked as outliers
        print(f"\n'{outliers_method.upper()}' method with current parameters "
              "inconsistently marked all LET values as outliers.")
        print(f"[{collected_vars}]")
        print("---> Please, check the parameters and try again.")
        return _pd.Series() # return an empty Series
    
    replacement_values = replace_outliers(
        outliers_df, let_column_filtered, x_data_filtered, replace_method, 
        knn_neighbors
    )
        
    # Create replacement values DataFrame
    outliers_and_replacements_df = _pd.DataFrame(
        replacement_values, 
        columns= ['replacements'],
        index=outliers_df.index
    )
    
    # Concatenate replacement values to outliers_df
    outliers_and_replacements_df = _pd.concat(
        [outliers_df, outliers_and_replacements_df], axis=1)
    
    if verbose: 
        # Print number of outliers and their values alongside x
        print(f"\n{outliers_method.upper()} - number of outliers "
              f"identified: {len(outliers_df.dropna())}")
        print(f"[Replacement method: {replace_method.upper()}]")
        print("Outlier and Replacement alongside corresponding x:")
        for idx in outliers_and_replacements_df.dropna().index:
            x_value = df.loc[idx, 'x']
            let_value = outliers_and_replacements_df.loc[idx, 'outliers']
            rep_value = outliers_and_replacements_df.loc[idx, 'replacements']
            print(f"x: {x_value:.4f}, LET: {let_value:.4f}, rep_value: {rep_value:.4f}")
            
    # Plot LET profile with outliers and replacements   
    radius_scaling = 'inverted' if outliers_method == 'lof' else 'direct'
    plot_let_outliers_replacements(
        x_data_filtered = x_data_filtered,
        let_column_filtered = let_column_filtered,
        outliers_and_replacements_df = outliers_and_replacements_df,
        outlier_scores= outlier_scores,
        radius_scaling = radius_scaling,
        cbar_label = cbar_label,
        **kwargs
    )

    return outliers_and_replacements_df

# identify_let_outliers modularization: method #1
def outlier_detection(
    let_column_filtered: _pd.Series, 
    x_data_filtered: _pd.Series, 
    outliers_method: str, 
    let_upper_limit: float,
    lof_neighbors: int, 
    dbscan_eps: float, 
    dbscan_min_samples: int
) -> Tuple[_pd.Series, Optional[_pd.Series], Optional[_pd.Series]]:
    """
    Detect outliers in LET profiles based on the specified method.

    Parameters:
    - let_column_filtered (pd.Series): The filtered LET profile column.
    - x_data_filtered (pd.Series): The filtered 'x' values corresponding to LET
      profile. 
    - outliers_method (str): Method for identifying outliers ('upper_limit',
      'lof', 'dbscan'). 
    - let_upper_limit (float): Upper limit for LET values, used if
      outliers_method is 'upper_limit'. 
    - lof_neighbors (int): Number of neighbors for LOF calculation, used if
      outliers_method is 'lof'. 
    - dbscan_eps (float): The maximum distance between two samples for DBSCAN,
      used if outliers_method is 'dbscan'. 
    - dbscan_min_samples (int): The number of samples in a neighborhood for
      DBSCAN, used if outliers_method is 'dbscan'. 

    Returns:
    - tuple: A tuple containing the DataFrame of outliers, outlier scores, and
      z-scores. 
    """
    # Implement outlier detection based on the specified method
    if outliers_method == 'upper_limit':
        outliers_df = let_column_filtered[
            let_column_filtered > let_upper_limit].dropna()
        outlier_scores = None  # No outlier scores for this method

    elif outliers_method == 'lof':
        # Calculate LOF
        lof = _LocalOutlierFactor(n_neighbors=lof_neighbors, contamination='auto')
        lof_predict = lof.fit_predict(
            _pd.concat([x_data_filtered, let_column_filtered], axis=1).values)
        lof_scores = lof.negative_outlier_factor_
        outliers_df = let_column_filtered[(lof_predict == -1)]
        outlier_scores = lof_scores[lof_predict == -1]

    elif outliers_method == 'dbscan':
        # Calculate DBSCAN
        dbscan = _DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan_predict = dbscan.fit_predict(
            _pd.concat([x_data_filtered, let_column_filtered], axis=1).values)
        outlier_mask = dbscan_predict == -1
        outliers_df = let_column_filtered[outlier_mask]
        outlier_scores = None  # No outlier scores for DBSCAN
        
    if outlier_scores is None:
        # Calculate z-score for LET column
        z_scores = _zscore(let_column_filtered)
        outlier_scores = z_scores.loc[outliers_df.index].values
        
    # Rename outliers_df column
    outliers_df.columns = ['outliers']
       
    # In the case where no outliers are found, 'outliers_df' is an empty Series.
    return outliers_df, outlier_scores

# identify_let_outliers modularization: method #2
def replace_outliers(
    outliers_df: _pd.Series, 
    let_column_filtered: _pd.Series, 
    x_data_filtered: _pd.Series, 
    replace_method: str,
    knn_neighbors: int
) -> List:
    """
    Replace outliers in LET profiles based on the specified method.

    Parameters:
    - outliers_df (pd.Series): DataFrame containing outliers in LET profiles.
    - let_column_filtered (pd.Series): The filtered LET profile column.
    - x_data_filtered (pd.Series): The filtered 'x' values corresponding to LET
      profile. 
    - replace_method (str): Method for replacing outliers.
    - knn_neighbors (int): Number of neighbors for KNN regressor, used if
      replace_method is 'nearest_neighbor_regressor'. 

    Returns:
    - list: A list of replacement values for outliers.
    """
    non_outliers_df = let_column_filtered[
        ~let_column_filtered.index.isin(outliers_df.index)]
        
    # Implement outlier replacement based on the specified method
    if replace_method == 'median':
        replacement_values = [non_outliers_df.median().values[0]] * len(outliers_df)
        
    elif replace_method == 'mean':
        replacement_values = [non_outliers_df.mean().values[0]] * len(outliers_df)
        
    elif 'local' in replace_method:
        # Split the string by '_' and get the part after 'local_'
        local_replacement_method = replace_method.split('_')[1] 
        replacement_values = local_statistic_outlier_replacement(
            _pd.concat([x_data_filtered.loc[outliers_df.index], outliers_df], axis=1), 
            _pd.concat([x_data_filtered, let_column_filtered], axis=1),
            n_neighbors = 20,
            replace_method = local_replacement_method,
        )

    elif replace_method == 'nearest_neighbor':
        replacement_values = nearest_neighbor_replacement(
            _pd.concat([x_data_filtered.loc[outliers_df.index], outliers_df], axis=1), 
            _pd.concat([x_data_filtered, let_column_filtered], axis=1)
        )

    elif replace_method == 'nearest_neighbor_regressor':
        replacement_values = knn_regressor_replacement(
            _pd.concat([x_data_filtered.loc[outliers_df.index], outliers_df], axis=1),
            _pd.concat([x_data_filtered, let_column_filtered], axis=1), n_neighbors=knn_neighbors
        )

    # Implement other replacement methods if needed
    
    return replacement_values
  
  
def replace_outliers_in_df(
    df: _pd.DataFrame,
    outliers_and_replacements_df: _pd.DataFrame,
    column_name: str,
) -> _pd.DataFrame:
    """
    Replace outliers in the selected column of the input DataFrame with
    replacement values from the 'replacements' column in the outliers DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - outliers_and_replacements_df (pd.DataFrame): DataFrame containing the
      outliers and replacement values. 
    - column_name (str): The name of the column in which outliers are to be
      replaced.

    Returns:
    - pd.DataFrame: A new DataFrame with outliers replaced in the selected
      column. 
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    # DataFrame 
    new_df = df.copy()
    
    # Check if 'outliers_and_replacements_df' is empty
    if outliers_and_replacements_df.empty:
        # Add a boolean column indicating no replacement was made
        new_df[column_name + '_is_replacement'] = False
        return new_df
      
    # Check if 'column_name' is in df.columns
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' is not a column in the input "
                         "DataFrame.")

    # Check if values in 'outliers' column of 'outliers_and_replacements_df'
    # match those in 'new_df' 
    if not new_df.loc[outliers_and_replacements_df.index, column_name].equals(
        outliers_and_replacements_df['outliers']):
        print("Values in 'outliers' column of 'outliers_and_replacements_df' "
              "do not match those in 'new_df' at corresponding indices:")
        comparison_df = _pd.concat(
            [outliers_and_replacements_df['outliers'], 
             new_df.loc[outliers_and_replacements_df.index, column_name]], 
            axis=1
        )
        comparison_df.columns = ['outliers_and_replacements_df', 'new_df']
        print(comparison_df)

    # Replace outliers in the selected column with replacement values
    new_df.loc[outliers_and_replacements_df.index, column_name] = \
        outliers_and_replacements_df['replacements']
        
    # Add boolean column to new_df indicating whether a replacement was made
    new_df[column_name + '_is_replacement'] = new_df.index.isin(
        outliers_and_replacements_df.index)

    return new_df