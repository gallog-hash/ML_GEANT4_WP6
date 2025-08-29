# src/core/preprocessing/preprocessing_utils.py
import os
import random
import re as _re
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch

from utils.logger import VAELogger

logger = VAELogger(__name__).get_logger()

def change_default_settings(random_seed: int):
    """
    Set random seeds for reproducibility across numpy, random, and torch.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_and_clean_data(data_dir: str, data_file: str, primary_particle: str,
                          let_type: str, cut_with_primary: bool,
                          drop_zero_cols: bool = False,
                          drop_zero_thr: float = 100.0,
                          verbose: int = 0) -> tuple:
    """
    Import and clean LET data for VAE training.

    This function imports a LET data file from the specified directory,
    applies essential preprocessing using the mandatory_let_df_manipulation
    function, and returns the processed DataFrame along with zero-value metrics.

    Parameters:
      - data_dir (str): Directory containing the data file.
      - data_file (str): Data file name (e.g., 'Let_1000-100.out').
      - primary_particle (str): Primary particle identifier (e.g., 'proton').
      - let_type (str): Type of LET data to extract ('track' or 'dose').
      - cut_with_primary (bool): Whether to cut with primary particle.
      - drop_zero_cols (bool, optional): Whether to drop columns with zero
        values. Default is False. 
      - drop_zero_thr (float, optional): Threshold for dropping zero values.
        Default is 100.0. 
      - verbose (int, optional): Verbosity level for logging. Default is 0.

    Returns:
      - df_processed (pd.DataFrame): The processed DataFrame after
        preprocessing. 
      - cut_in_um(float): Cut value in micrometers.
      - voxel_in_um (float): Voxel size in micrometers.
    """
    file_path = Path(data_dir) / data_file
    try:
        df = import_out_file(file_path)
    except Exception as e:
        logger.error("Data import failed from %s: %s", file_path, e)
        raise e

    logger.info("DataFrame successfully loaded from: %s", file_path)
    if verbose > 0:
        logger.debug("First 10 rows of the DataFrame:\n%s", df.head(10))
        
    # Try to extract cut_in_um and voxel_in_um from the full file path.
    cut_in_um, voxel_in_um = extract_cut_voxel(file_path, verbose=verbose)
    
    # Use mandatory cleaning.
    df_non_zero, _, _ = \
        mandatory_let_df_manipulation(
            let_df=df,
            vsize_um=voxel_in_um,
            let_type=let_type,
            primary_particle=primary_particle,
            drop_zero_thr=drop_zero_thr,
            drop_zero_cols=drop_zero_cols,
            verbose=verbose
        )
        
    if cut_with_primary:
        df_processed = cut_dataframe_with_primary(
            df_non_zero, column_type=let_type,                             
            primary_particle=primary_particle
            )
        logger.info("Truncated DataFrame based on primary particle.")
    else:
        df_processed = df_non_zero.copy()
        logger.info("Unaltered copy of non-zero data.")
    
    return (df_processed, cut_in_um, voxel_in_um)

def reorder_identity_features(df: pd.DataFrame, identity_feature_names: list
                              ) -> pd.DataFrame:
    """
    Reorder the columns of the DataFrame so that the identity features appear at
    the right end.

    Parameters:
      - df (pd.DataFrame): The input DataFrame.
      - identity_feature_names (list of str): List of column names that are
        designated as identity features and should bypass the encoding.

    Returns:
      - pd.DataFrame: The DataFrame with columns reordered such that all columns
        not in identity_feature_names come first, followed by the identity features.
    """
    # Identify columns not designated as identity features.
    other_columns = [col for col in df.columns if col not in identity_feature_names]
    # Define the new column order: non-identity columns first, then identity columns.
    new_order = other_columns + identity_feature_names
    return df[new_order]

def import_out_file(
    file_path: str, 
    separator: str = '\t'
) -> pd.DataFrame:
    """
    Import data from a .out file using pandas.read_csv.
    
    Parameters:
    - file_path (str): The path to the .out file.
    - separator (str, optional): The field separator used in the .out file.
      Default to '\t' (tab).
    
    Returns:
    - DataFrame: A pandas DataFrame containing the data froom the .out file.
    """
    try:
        # Use pandas to read the .out file
        df = pd.read_csv(file_path, sep=separator)
        
        # Drop specific columns (j, k) if needed
        if 'j' in df.columns and 'k' in df.columns:
            df.drop(columns=['j', 'k'], inplace=True)
        
        return df
    
    except FileNotFoundError:
        error_message = f"File '{file_path}' not found."
        raise OSError(error_message)
    
    except Exception as e:
        error_message = f"{str(e)}"
        raise OSError(error_message)
    
def extract_cut_voxel(full_path: str, verbose: bool = False) -> Tuple[float, float]:
    """
    Extract the cut and voxel values from a full file path.

    This function first attempts to extract the cut and voxel values from the
    filename (without extension). The expected format is:
      'cut_<cut_value>-voxel_<voxel_value>.extension'
    If extraction from the filename fails, it falls back to extracting these
    values from the last directory name in the path using regex.

    Parameters:
      - full_path (str): The full path to the file (directory and filename).
      - verbose (bool, optional): If True, logs detailed info messages.

    Returns:
      - tuple: (cut_in_um, voxel_in_um) where both values are in micrometers.

    Raises:
      - ValueError: If the values cannot be extracted from either the filename or
        the directory.
    """
    # Attempt extraction from filename
    file_path = Path(full_path)
    filename = file_path.name
    file_root = file_path.stem
    try:
        # Expected format: "cut_<cut_value>-voxel_<voxel_value>"
        cut_part, voxel_part = file_root.split('-')
        # Remove prefix "cut_" from cut_part.
        _, cut_str = cut_part.split('_')
        # Remove prefix "voxel_" from voxel_part if present.
        if _re.search(r'(?i)voxel', voxel_part):
            _, voxel_str = _re.split(r'(?i)voxel[_-]', voxel_part)
        else:
            voxel_str = voxel_part
        cut_in_um = float(cut_str)
        voxel_in_um = float(voxel_str)
        if verbose:
            logger.info("Extracted from filename - Cut: %.1f µm, Voxel: %.1f µm", 
                        cut_in_um, voxel_in_um)
        return cut_in_um, voxel_in_um
    except ValueError:
        logger.warning("Failed to extract cut and voxel from filename: %s", filename)
    
    # Fallback: attempt extraction from the directory name.
    last_dir = Path(full_path).parent.name
    if verbose:
        logger.info("Attempting to extract cut and voxel values from directory "
                    "name: %s", last_dir)
    
    cut_match = _re.search(r'(?:cut|c)(\d+)(mm|um)', last_dir, _re.IGNORECASE)
    voxel_match = _re.search(r'(?:voxel|v)(\d+)(mm|um)?', last_dir, _re.IGNORECASE)
    
    cut_in_um = None
    voxel_in_um = None
    
    if cut_match:
        cut_in_um = float(cut_match.group(1))
        if cut_match.group(2) and cut_match.group(2).lower() == 'mm':
            cut_in_um *= 1000.0
        else:
            if verbose:
                logger.info("Assuming cut value is in micrometers.")
    else:
        logger.warning("Could not extract cut value from directory name.")
    
    if voxel_match:
        voxel_in_um = float(voxel_match.group(1))
        if voxel_match.group(2) and voxel_match.group(2).lower() == 'mm':
            voxel_in_um *= 1000.0
        else:
            if verbose:
                logger.info("Assuming voxel value is in micrometers.")
    else:
        logger.warning("Could not extract voxel value from directory name.")
    
    if cut_in_um is None or voxel_in_um is None:
        logger.error("Could not extract cut and voxel values from filename or directory.")
        raise ValueError("Cut and voxel values must be provided.")
    else:
        if verbose:
            logger.info("Extracted Cut: %d µm, Voxel: %d µm", cut_in_um, voxel_in_um)
    
    return cut_in_um, voxel_in_um

def mandatory_let_df_manipulation(
    let_df: pd.DataFrame,
    vsize_um: float,
    let_type: str,
    primary_particle: str,
    drop_zero_thr: float = 100.0,
    drop_zero_cols: bool = True,
    verbose: int = 0
) -> tuple:
    """
    Apply essential preprocessing steps to the LET (Linear Energy Transfer)
    DataFrame, including feature transformation, filtering, and extraction of
    key data columns. 

    Args:
    - let_df (pd.DataFrame): Input DataFrame containing LET values for analysis.
    - vsize_um (float): Voxel size in micrometers for converting index values to
        depth. 
    - let_type (str): Type of LET data to extract ('track' or 'dose').
    - primary_pattern (str): Regex pattern matching primary particle
        identifiers.
    - drop_zero_thr (float): Threshold percentage for dropping columns with high
        zero percentages. Defaults to 100.0.
    - drop_zero_cols (bool): Flag to indicate whether to drop columns with high
        zero percentages.
    - verbose (int): Verbosity level for logging information. Defaults to 0.

    Returns:
    - tuple: Processed DataFrame, zero percentages for primary particle columns,
    and overall zero percentages. 
    """

    # Transform the index feature 'i' in the df DataFrame into a depth feature
    # 'x'.  
    df = convert_index_feature_to_depth(
        let_df, vsize_um, 
        input_size='um', 
        output_size='mm'
    )
    
    if not primary_particle.endswith('_1'):
        primary_search_pattern = primary_particle + '_1'
    else:
        primary_search_pattern = primary_particle
        
    # Extract columns related to the specified quantity
    df = extract_track_or_dose_cols(df, column_type=let_type)
    
    # Get the percentage of zero values per column in the full Dataframe. 
    percentage_eq_to_zero_df = calculate_zero_values_percentage(
        df, verbose=verbose)
    
    # Get the percentage of zero values in the primary particle columns.
    primary_zero_perc = percentage_eq_to_zero_df[
        percentage_eq_to_zero_df.index.str.contains(primary_search_pattern)
    ]
    
    if drop_zero_cols:
        # Remove columns with a percentage of values equal to 0 higher than
        # threshold.
        df = drop_zero_columns(df, threshold=drop_zero_thr, 
                                   apply_rename=True, verbose=verbose,
                                   zero_percentage=percentage_eq_to_zero_df)
    
    return df, primary_zero_perc, percentage_eq_to_zero_df

def convert_index_feature_to_depth(
    df: pd.DataFrame, 
    voxel_size: float, 
    input_size: str = 'um', 
    output_size: str = 'mm'
) -> pd.DataFrame: 
    """
    Convert an index feature in a DataFrame to depth in a specified output unit.

    Parameters:
    - df (DataFrame): The input DataFrame containing the index feature 'i'.
    - voxel_size (float): Size of the voxel in the input unit.
    - input_size (str, optional): Unit of the input voxel_size. Defaults to 'um'
    (micrometers). 
    - output_size (str, optional): Desired unit for the output depth. Defaults
    to 'mm' (millimeters). 

    Returns:
    - DataFrame: A new DataFrame with the 'x' column representing the depth in
    the specified output unit. 
    """
    # Conversion factors for different units to millimeters
    unit_to_mm = {'nm': 1e-6, 'um': 1e-3, 'mm': 1., 'cm': 10., 'm': 1e3}

    # Check if input_size is valid, default to 'um' if not
    if input_size not in unit_to_mm:
        print("Warning: Invalid input_size '{}'. Defaulting to 'um'.".format(
            input_size))
        input_size = 'um'
    
    # Check if output_size is valid, default to 'mm' if not
    if output_size not in unit_to_mm:
        print("Warning: Invalid output_size '{}'. Defaulting to 'mm'.".format(
            output_size))
        output_size = 'mm'
    
    # Convert voxel size to millimeters
    voxel_size_mm = voxel_size * unit_to_mm[input_size]
    
    # Get the index feature values as a NumPy array
    i = df['i'].values
    
    # Convert 'x' to the desired output_size using NumPy array operations
    x = voxel_size_mm * i / unit_to_mm[output_size]

    # Drop the column 'i' from the dataframe
    df = df.drop(columns='i')
    
    # Create a new DataFrame with the 'x' column
    x_df = pd.DataFrame(x, columns=['x'])
    
    # Concatenate the DataFrames along the columns (axis=1)
    x_df = pd.concat([x_df, df], axis=1)
    
    return x_df

def extract_track_or_dose_cols(df, column_type='track'):
    """
    Filter and keep specific columns in a DataFrame based on column type.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_type (str, optional): Type of columns to keep ('track' or 'dose').
      Defaults to 'track'.

    Returns:
    - DataFrame: A new DataFrame without the specified columns.
    """
    # Define search patterns based on column type to keep columns
    if column_type == 'track':
        search_patterns = ['LTT', '_T']
    elif column_type == 'dose':
        search_patterns = ['LDT', '_D']
    else:
        raise ValueError("Invalid column_type. Use 'track' or 'dose'.")
    
    # Check if any of the search patterns are present in DataFrame columns
    if not any(pattern in col for col in df.columns for pattern in search_patterns):
        raise ValueError("No columns matching the search patterns "
                         f"{search_patterns} found in the DataFrame.")

    # Filter column names containing any of the specified patterns
    filtered_cols = [col for col in df.columns if 
                     any(pattern in col for pattern in search_patterns)]

    # Ensure 'i' or 'x' column is included
    if 'i' in df.columns:
        filtered_cols = ['i'] + filtered_cols
    elif 'x' in df.columns:
        filtered_cols = ['x'] + filtered_cols
    else:
        raise ValueError("No index or depth column ('i' or 'x') found in input "
                         "DataFrame.")


    # Select only the specified columns
    df_filtered = df[filtered_cols]

    return df_filtered

def calculate_zero_values_percentage(
    df: pd.DataFrame,
    x_threshold: float = 0.0,
    condition: str = "above",
    verbose: bool = False,
    ) -> pd.DataFrame:
    """
    Calculate the percentage of values equal to 0 in each column of a DataFrame,
    filtered by a threshold condition on column 'x'.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - x_threshold (float): The threshold value for the 'x' column.
    - condition (str, optional): Condition to apply for the threshold ('above'
    or 'below'). Defaults to 'above'. 
    - verbose (bool, optional): If True, logs detailed column statistics.
    
    Returns:
    - DataFrame: A DataFrame containing column names as index and the
      corresponding percentage of values equal to zero as values.
    """
    if 'x' not in df.columns:
        raise ValueError("DataFrame must contain a column named 'x' "
                         "for threshold filtering.")
    
    if condition not in ["above", "below"]:
        raise ValueError("Condition must be 'above' or 'below'.")
    
    if condition == "above":
        filtered_df = df[df['x'] > x_threshold]
    else:
        filtered_df = df[df['x'] <= x_threshold]

    percentage_eq_zero_df = ((filtered_df == 0).mean()) * 100
    
    if verbose:
        for col, perc in percentage_eq_zero_df.items():
            logger.info("Column '%s' has %.2f%% zero values.", col, perc)
    
    return percentage_eq_zero_df

def drop_zero_columns(
    df: pd.DataFrame, 
    threshold: float = 100.0,
    apply_rename: bool = False,
    verbose: int = 0,
    zero_percentage: pd.Series = None
) -> pd.DataFrame:
    """
    Drop columns with all zeros from a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - threshold (float, optional): The threshold percentage for dropping columns.
      Defaults to 100.0.
    - apply_rename (bool, optional): Whether to apply renaming to columns.
      Defaults to False. 
    - verbose (int, optional): Set the verbosity level. Default is 0, which 
      means no output. 
    - zero_percentage (pd.Series, optional): Precomputed percentage of zero values 
      for each column. If not provided, it will be calculated.

    Returns:
    - DataFrame: DataFrame with zero columns removed.
    """  
    # Calculate the percentage of zero values in each column if not provided
    if zero_percentage is None:
        zero_percentage = calculate_zero_values_percentage(df, verbose=verbose)
    
    # Find columns with a percentage of zero values greater than or equal to
    # threshold 
    columns_to_drop = zero_percentage[zero_percentage > threshold].index
          
    # Drop columns with all zeros
    df = df.drop(columns=columns_to_drop)
    
    if verbose > 0:
        logger.info("The following columns with a percentage of zero values "
                f"greater than {threshold:.4f}% have been removed:")
        # Log each column name in a new line
        for col in columns_to_drop:
            logger.info(col)
    
    if apply_rename:
        df = rename_columns(df, columns_to_drop, verbose=verbose)

    return df

def rename_columns(
    df: pd.DataFrame, 
    zero_columns: pd.Index,
    verbose: int = 0
) -> pd.DataFrame:
    """
    Rename columns with all zeros by adding progressive numbering.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - zero_columns (Index): Index of columns with all zeros.
    - verbose (int, optional): Set the verbosity level. Default is 0, which 
      means no output. 

    Returns:
    - DataFrame: DataFrame with renamed columns.
    """
    unique_zero_columns_prefix = [col.split('.')[0] for col in zero_columns]
    unique_zero_columns_prefix = np.unique(unique_zero_columns_prefix)
    
    # Get columns for renaming
    rename_columns_dict = {prefix: [col for col in df.columns if 
                                    col.split('.')[0] == prefix]
                           for prefix in unique_zero_columns_prefix}
    
    # Rename remaining columns with progressive numbering
    old_column_names = list(df.columns)
    new_column_names = list(df.columns)
    
    for prefix, columns in rename_columns_dict.items():
        for i, column in enumerate(columns, start=0):
            new_column_name = f"{prefix}.{i}"
            index_to_replace = new_column_names.index(column)
            new_column_names[index_to_replace] = new_column_name
    
    # Remove '.0' suffix from column names
    new_column_names = [col.split('.')[0] if col.endswith('.0') else col for 
                  col in new_column_names]
    
    # Assign the new column names back to the DataFrame
    df.columns = new_column_names
    
    if verbose > 0:
        for old_name, new_name in zip(old_column_names, new_column_names):
            if old_name != new_name:
                print(f"Renaming column: {old_name} -> {new_name}")
        print("\nColumn renaming completed.\n\n")

    return df

def cut_dataframe_with_primary(
    df: pd.DataFrame, 
    column_type: str = 'track', 
    primary_particle: str = 'proton'
) -> pd.DataFrame:
    """
    Cut a DataFrame up to the first minimum value after the peak in the column
    with the most non-zero values among the primary particle columns.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_type (str, optional): Type of columns to consider ('track' or
      'dose'). Defaults to 'track'.
    - primary_particle (str, optional): Name of the primary particle. Defaults
      to 'proton'. 

    Returns:
    - DataFrame: DataFrame containing data up to the first minimum value after
      the peak. 
    """
    df_by_column_type = extract_track_or_dose_cols(df, column_type)
    
    if not primary_particle.endswith('_1'):
        primary_search_pattern = primary_particle + '_1'
    else:
        primary_search_pattern = primary_particle
    
    # Find the primary particle column with the fewest zeros
    primary_column = find_primary_column_with_fewest_zeros(
        df_by_column_type, primary_search_pattern
    )
    
    # Find the index of the minimum value after the peak
    min_after_max_index = find_min_after_peak_index(
        df_by_column_type[primary_column]
    )
    
    # Return the DataFrame up to the primary minimum value after the peak
    return df.iloc[:min_after_max_index + 1]

def find_primary_column_with_fewest_zeros(
    df: pd.DataFrame, 
    primary_search_pattern: str
) -> str:
    """
    Find the primary particle column with the fewest zero values.

    Parameters:
    - df (DataFrame): The input DataFrame to analyze.
    - primary_search_pattern (str): The pattern to identify primary particle
      columns. 

    Returns:
    - str: The name of the primary particle column with the fewest zeros.
    """
    if not primary_search_pattern.endswith('_1'):
        primary_search_pattern = primary_search_pattern + '_1'
    else:
        primary_search_pattern = primary_search_pattern
        
    # Extract primary particle columns
    primary_columns = [col for col in df.columns if primary_search_pattern in col]

    # Find the column with the fewest zero values
    return max(primary_columns, key=lambda col: df[col].astype(bool).sum())

def find_min_after_peak_index(series: pd.Series) -> int:
    """
    Find the index of the first minimum value after the peak in a Series.

    Parameters:
    - series (Series): The input Series to analyze.

    Returns:
    - int: The index of the first minimum value after the peak.
    """
    # Find the index of the maximum value in the Series
    max_index = series.idxmax()

    # Find the index of the minimum value after the maximum
    min_after_max_index = series[max_index:].idxmin()

    return min_after_max_index

def find_bragg_peak_start(
    values: Union[np.ndarray, pd.Series], 
    threshold_ratio: float = 0.001
) -> int:
    """
    Find the index where the pronounced Bragg peak begins.
    
    This function computes the first derivative of the input array (or Series)
    (i.e., the differences between successive values) and then finds the first
    index where the derivative exceeds a fraction (threshold_ratio) of the
    maximum derivative. This index is taken as the beginning of the pronounced
    peak. 
    
    Parameters:
      - values (np.ndarray or pd.Series): Array or Series of LET values
        approximating a Bragg curve. 
      - threshold_ratio (float, optional): Fraction of the maximum derivative to 
        determine the peak start. Defaults to 0.001.
        
    Returns:
      - int: The index in the input where the pronounced peak begins.
    
    Raises:
      - ValueError: If the input has fewer than 2 values.
    """
    # Convert Series to numpy array if necessary.
    if isinstance(values, pd.Series):
        values = values.values

    if len(values) < 2:
        raise ValueError("Input must contain at least two values.")
    
    differences = np.diff(values)
    max_difference = np.max(differences)
    threshold = threshold_ratio * max_difference
    
    for i, diff_value in enumerate(differences):
        if diff_value >= threshold:
            return i + 1  # Return the corresponding index in the original array.
    
    # If no derivative exceeds the threshold, return the last index.
    return len(values) - 1

def import_and_process_dose_fluence(
    data_dir: str, dose_file: str, voxel_in_um: int,
    zero_thr_percent: int, drop_zero_cols: bool = False, verbose: int = 0, 
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Import dose data and convert its index to a depth feature.
    
    Returns:
    - dose_profile (DataFrame): DataFrame with 'x' and 'Dose(Gy)' columns.
    - fluence_df (DataFrame): Dataframe with 'x' and '*_f' columns.
    """
    dose_df = import_out_file(os.path.join(data_dir, dose_file))
    dose_df = convert_index_feature_to_depth(
        dose_df, voxel_in_um, input_size='um', output_size='mm'
    )
    dose_profile = dose_df[['x', 'Dose(Gy)']].copy()
    
    fluence_columns = [col for col in dose_df.columns if "_f" in col]
    fluence_df = dose_df[['x'] + fluence_columns].copy()
    
    # Remove columns with a percentage of values equal to 0 higher than
    # threshold from fluence DataFrame if drop_zero_cols is True.
    if drop_zero_cols:
        fluence_df = drop_zero_columns(
            fluence_df, threshold=zero_thr_percent,
            apply_rename=True,
            verbose=verbose
        )
    
    if verbose > 0: 
        # Print statement to introduce the dose and fluence DataFrames
        logger.info("Dose DataFrame: %s", dose_df.head())
        logger.info("Fluence DataFrame: %s", fluence_df.head())
    
    return dose_profile, fluence_df