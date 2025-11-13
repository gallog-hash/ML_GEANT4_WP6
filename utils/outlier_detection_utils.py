"""
Outlier Detection and Replacement Utilities

This module provides functions for identifying and replacing outliers in 
LET (Linear Energy Transfer) data, extracted from the data_eng_pkg package.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor, NearestNeighbors

logger = logging.getLogger(__name__)


def identify_let_outliers(
    df: pd.DataFrame,
    column_type: str = 'track',
    outliers_method: str = 'lof',
    let_upper_limit: float = 100.0,
    lof_neighbors: int = 20,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    knn_neighbors: int = 5,
    replace_method: str = 'median',
    x_bounds: Tuple[float, float] = (32.5, 40.0),
    cbar_label: str = "LOF Outliers Factor",
    cut_in_um: int = 1000,
    voxel_in_um: int = 100,
    verbose: bool = False,
    auto_save: bool = False,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Identify outliers in LET data using various methods and compute 
    replacement values.
    
    Parameters:
        df: DataFrame containing LET data
        column_type: Type of LET column ('track' or 'dose')  
        outliers_method: Method for outlier detection ('lof', 'dbscan', 
                        'upper_limit')
        let_upper_limit: Upper threshold for LET values
        lof_neighbors: Number of neighbors for LOF
        dbscan_eps: Maximum distance between samples for DBSCAN
        dbscan_min_samples: Minimum samples for DBSCAN core point
        knn_neighbors: Number of neighbors for KNN regressor
        replace_method: Method for computing replacement values
        x_bounds: Tuple of (min_x, max_x) bounds for outlier search
        cbar_label: Label for colorbar in plots
        cut_in_um: Cut value in micrometers
        voxel_in_um: Voxel size in micrometers
        verbose: Whether to print verbose output
        auto_save: Whether to automatically save plots
        save_dir: Directory to save plots
        
    Returns:
        DataFrame with outlier indices and replacement values
    """
    let_prefix = 'LTT' if column_type.lower() == 'track' else 'LDT'
    let_columns = df.filter(like=let_prefix).columns
    
    if let_columns.empty:
        logger.warning(f"No {let_prefix} columns found in DataFrame")
        return pd.DataFrame(columns=['outliers', 'replacements'])
    
    # Use first LET column for outlier detection
    let_column = let_columns[0]
    let_series = df[let_column]
    
    # Filter data within x_bounds
    x_mask = (df['x'] >= x_bounds[0]) & (df['x'] <= x_bounds[1])
    filtered_let = let_series[x_mask]
    
    if len(filtered_let) == 0:
        logger.warning("No data points within specified x_bounds")
        return pd.DataFrame(columns=['outliers', 'replacements'])
    
    # Identify outliers based on method
    outlier_mask = _identify_outliers_by_method(
        filtered_let, 
        outliers_method, 
        let_upper_limit,
        lof_neighbors,
        dbscan_eps, 
        dbscan_min_samples,
        verbose
    )
    
    if not outlier_mask.any():
        if verbose:
            logger.info(f"No outliers found using {outliers_method} method")
        return pd.DataFrame(columns=['outliers', 'replacements'])
    
    # Get outlier indices and values
    outlier_indices = filtered_let[outlier_mask].index
    outlier_values = filtered_let[outlier_mask].values
    
    # Compute replacement values
    replacement_values = _compute_replacement_values(
        filtered_let,
        outlier_mask, 
        replace_method,
        knn_neighbors,
        verbose
    )
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'outliers': outlier_values,
        'replacements': replacement_values
    }, index=outlier_indices)
    
    if verbose:
        logger.info(f"Found {len(result_df)} outliers using {outliers_method} "
                    f"method with {replace_method} replacement")
    
    # Generate visualization if requested
    if auto_save and save_dir is not None and not result_df.empty:
        try:
            from .outlier_visualization_utils import plot_let_outliers_replacement
            
            plot_let_outliers_replacement(
                df=df,
                outliers_df=result_df,
                let_column=let_column,
                outliers_method=outliers_method,
                replace_method=replace_method,
                x_bounds=x_bounds,
                cbar_label=cbar_label,
                cut_in_um=cut_in_um,
                voxel_in_um=voxel_in_um,
                save_fig=True,
                save_dir=save_dir
            )
        except ImportError:
            logger.warning("Could not import plot_let_outliers_replacement for visualization")
    
    return result_df


def _identify_outliers_by_method(
    let_series: pd.Series,
    method: str,
    let_upper_limit: float,
    lof_neighbors: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    verbose: bool
) -> np.ndarray:
    """
    Identify outliers using the specified method.
    
    Returns:
        Boolean mask indicating outlier positions
    """
    if method.lower() == 'upper_limit':
        return let_series > let_upper_limit
    
    elif method.lower() == 'lof':
        # Local Outlier Factor
        lof = LocalOutlierFactor(
            n_neighbors=lof_neighbors, 
            contamination='auto'
        )
        outlier_labels = lof.fit_predict(let_series.values.reshape(-1, 1))
        return outlier_labels == -1
    
    elif method.lower() == 'dbscan':
        # DBSCAN clustering
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(let_series.values.reshape(-1, 1))
        # Points labeled as -1 are noise/outliers
        return cluster_labels == -1
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def _compute_replacement_values(
    let_series: pd.Series,
    outlier_mask: np.ndarray,
    method: str,
    knn_neighbors: int,
    verbose: bool
) -> np.ndarray:
    """
    Compute replacement values for outliers using the specified method.
    
    Returns:
        Array of replacement values
    """
    non_outlier_values = let_series[~outlier_mask]
    n_outliers = outlier_mask.sum()
    
    if method.lower() == 'median':
        replacement = np.median(non_outlier_values)
        return np.full(n_outliers, replacement)
    
    elif method.lower() == 'mean':
        replacement = np.mean(non_outlier_values)
        return np.full(n_outliers, replacement)
    
    elif method.lower() == 'local_mean':
        # Use KNN to find local mean for each outlier
        return _compute_local_mean_replacements(
            let_series, outlier_mask, knn_neighbors
        )
    
    elif method.lower() == 'nearest_neighbor':
        # Simple nearest neighbor replacement
        return _compute_nearest_neighbor_replacements(
            let_series, outlier_mask
        )
    
    elif method.lower() == 'nearest_neighbor_regressor':
        # KNN regressor based replacement
        return _compute_knn_regressor_replacements(
            let_series, outlier_mask, knn_neighbors
        )
    
    else:
        raise ValueError(f"Unknown replacement method: {method}")


def _compute_local_mean_replacements(
    let_series: pd.Series,
    outlier_mask: np.ndarray,
    k_neighbors: int
) -> np.ndarray:
    """Compute local mean replacements using k-nearest neighbors."""
    non_outlier_indices = let_series.index[~outlier_mask]
    non_outlier_values = let_series[~outlier_mask]
    outlier_indices = let_series.index[outlier_mask]
    
    if len(non_outlier_values) < k_neighbors:
        # Not enough non-outliers, use global mean
        return np.full(len(outlier_indices), np.mean(non_outlier_values))
    
    # Fit KNN model
    nbrs = NearestNeighbors(n_neighbors=k_neighbors)
    nbrs.fit(non_outlier_indices.values.reshape(-1, 1))
    
    replacements = []
    for outlier_idx in outlier_indices:
        _, indices = nbrs.kneighbors([[outlier_idx]])
        neighbor_values = non_outlier_values.iloc[indices[0]]
        replacements.append(np.mean(neighbor_values))
    
    return np.array(replacements)


def _compute_nearest_neighbor_replacements(
    let_series: pd.Series,
    outlier_mask: np.ndarray
) -> np.ndarray:
    """Compute nearest neighbor replacements."""
    non_outlier_indices = let_series.index[~outlier_mask]
    non_outlier_values = let_series[~outlier_mask]
    outlier_indices = let_series.index[outlier_mask]
    
    replacements = []
    for outlier_idx in outlier_indices:
        # Find closest non-outlier index
        distances = np.abs(non_outlier_indices - outlier_idx)
        closest_idx = non_outlier_indices[np.argmin(distances)]
        replacements.append(non_outlier_values.loc[closest_idx])
    
    return np.array(replacements)


def _compute_knn_regressor_replacements(
    let_series: pd.Series,
    outlier_mask: np.ndarray,
    k_neighbors: int
) -> np.ndarray:
    """Compute KNN regressor based replacements."""
    non_outlier_indices = let_series.index[~outlier_mask]
    non_outlier_values = let_series[~outlier_mask]
    outlier_indices = let_series.index[outlier_mask]
    
    if len(non_outlier_values) < k_neighbors:
        k_neighbors = len(non_outlier_values)
    
    # Train KNN regressor
    knn = KNeighborsRegressor(n_neighbors=k_neighbors)
    knn.fit(non_outlier_indices.values.reshape(-1, 1), non_outlier_values)
    
    # Predict replacements
    replacements = knn.predict(outlier_indices.values.reshape(-1, 1))
    
    return replacements


def replace_outliers_in_df(
    df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    column_prefix: str
) -> pd.DataFrame:
    """
    Replace outliers in the DataFrame with their computed replacement values.
    
    Parameters:
        df: Original DataFrame
        outliers_df: DataFrame with outlier indices and replacement values
        column_prefix: Prefix for columns to apply replacements ('LTT' or 'LDT')
        
    Returns:
        DataFrame with outliers replaced
    """
    if outliers_df.empty:
        logger.info("No outliers to replace")
        return df.copy()
    
    df_replaced = df.copy()
    let_columns = df.filter(like=column_prefix).columns
    
    for col in let_columns:
        for idx, replacement_value in outliers_df['replacements'].items():
            if idx in df_replaced.index:
                df_replaced.loc[idx, col] = replacement_value
    
    logger.info(f"Replaced {len(outliers_df)} outliers in {len(let_columns)} "
                f"columns with prefix '{column_prefix}'")
    
    return df_replaced