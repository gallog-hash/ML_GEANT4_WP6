import os as _os

import numpy as _np
import pandas as _pd
from pandas import DataFrame as _DataFrame

# ===========================
# === DATA IMPORT SECTION ===
# ===========================

def import_out_file(
    file_path: str, 
    separator: str = '\t'
) -> _DataFrame:
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
        df = _pd.read_csv(file_path, sep=separator)
        
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
    
def get_cut_voxel_from_filename(filename: str) -> tuple:
    """
    Extract cut and voxel values from a filename formatted as
    'cut_<cut_value>-voxel_<voxel_value>.extension'. 

    Parameters:
    - filename (str): The filename containing cut and voxel values.

    Returns:
    - tuple: A tuple containing the cut value and voxel value extracted from the
    filename. 
    """
    # Remove file extension from filename
    data_file = _os.path.splitext(filename)[0]

    # Split filename at specified delimiter
    cut_str, voxel_str = data_file.split('-')

    # Remove filename root
    _, cut_str = cut_str.split('_')

    # Convert strings to floats
    return float(cut_str), float(voxel_str)

def convert_index_feature_to_depth(
    df: _pd.DataFrame, 
    voxel_size: float, 
    input_size: str = 'um', 
    output_size: str = 'mm'
) -> _pd.DataFrame: 
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
    x_df = _pd.DataFrame(x, columns=['x'])
    
    # Concatenate the DataFrames along the columns (axis=1)
    x_df = _pd.concat([x_df, df], axis=1)
    
    return x_df

def print_columns_with_pattern(
    df: _DataFrame, 
    pattern: str
) -> None:
    """
    Print column names in a DataFrame containing a specified pattern.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - pattern (str): The pattern to search for in column names.
    """
    columns_with_pattern = [col for col in df.columns if pattern in col]
    print("Columns with pattern '{}':".format(pattern))
    
    for index, col in enumerate(columns_with_pattern):
        print(f"{index}: {col}")
        
# =============================
# === DATA ANALYSIS SECTION ===
# =============================

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

def cut_dataframe_at_abrupt_variation(
    df: _pd.DataFrame, 
    column_type: str = 'track', 
    std_threshold: float = 3
) -> _pd.DataFrame:
    """
    Cut a DataFrame at the point where abrupt variations occur in the LET total
    column. 

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_type (str, optional): Type of columns to include ('track' or
      'dose'). Defaults to 'track'. 
    - std_threshold (float, optional): Standard deviation threshold for detecting
      abrupt variations. Defaults to 3. 

    Returns:
    - DataFrame: DataFrame cut at the point of the first abrupt variation in the
      LET total column. 
    """
    # Define search patterns based on column type to drop columns
    if column_type == 'track':
        let_total_pattern = 'LTT'
    elif column_type == 'dose':
        let_total_pattern = 'LDT'
    else:
        raise ValueError("Invalid column_type. Use 'track' or 'dose'.")
    
    # Check if the LET total column exists in the DataFrame
    if let_total_pattern not in df.columns:
        raise ValueError("The DataFrame does not contain the column "
                         f"'{let_total_pattern}'.")
    
    # Extract the LET total values from the DataFrame
    y_values = df[let_total_pattern]
    
    # Compute the gradients of LET total values
    diff = _np.diff(y_values)
    
    # Compute mean and standard deviation of the gradients
    diff_mean = _np.mean(diff)
    diff_std = _np.std(diff)
    
    # Set a threshold for detecting abrupt variations in LET total values
    diff_thr = diff_mean + std_threshold * diff_std
    
    # Detect indices where abrupt variations occur
    abrupt_variation_indices = _np.where(_np.abs(diff) > diff_thr)[0]

    # Check if there are abrupt variations
    if any(abrupt_variation_indices):
        # Get the index of the first abrupt variation
        first_abrupt_variation_index = abrupt_variation_indices[0]
    else:
        # If no abrupt variations are found, return the original DataFrame
        print("No abrupt variation detected. Returning the original DataFrame.")
        return df
    
    # Return the DataFrame up to the first abrupt variation
    return df.iloc[:first_abrupt_variation_index + 1]

def cut_dataframe_with_primary(
    df: _pd.DataFrame, 
    column_type: str = 'track', 
    primary_particle: str = 'proton'
) -> _pd.DataFrame:
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
    
    # Compose the primary particle name with the suffix '_1'
    primary_search_pattern = f'{primary_particle}_1'
    
    # Extract primary columns
    primary_columns = [col for col in df_by_column_type.columns if 
                       primary_search_pattern in col]
    
    # Select the column with the most non-zero values
    max_non_zero_column = max(primary_columns, 
                              key=lambda col: df[col].astype(bool).sum())
    
    # Find the index of the maximum value in the array
    max_index = df_by_column_type[max_non_zero_column].idxmax()

    # Find the index of the minimum after the maximum value
    min_after_max_index = \
        df_by_column_type[max_non_zero_column][max_index:].idxmin()
        
    # Return the DataFrame up to the primary minimum value after the peak
    return df.iloc[:min_after_max_index + 1]

def calculate_zero_values_percentage(
    df: _pd.DataFrame,
    x_threshold: float = 0.0,
    condition: str = "above",
    ) -> _pd.DataFrame:
    """
    Calculate the percentage of values equal to 0 in each column of a DataFrame,
    filtered by a threshold condition on column 'x'.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - x_threshold (float): The threshold value for the 'x' column.
    - condition (str, optional): Condition to apply for the threshold ('above'
    or 'below'). Defaults to 'above'. 
    
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
    
    return percentage_eq_zero_df

def drop_zero_columns(
    df: _pd.DataFrame, 
    threshold: float = 100.0,
    apply_rename: bool = False,
    verbose: int = 0,
) -> _pd.DataFrame:
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

    Returns:
    - DataFrame: DataFrame with zero columns removed.
    """  
    # Calculate the percentage of zero values in each column
    zero_percentage = calculate_zero_values_percentage(df)
    
    # Find columns with a percentage of zero values greater than or equal to
    # threshold 
    columns_to_drop = zero_percentage[zero_percentage >= threshold].index
          
    # Drop columns with all zeros
    df = df.drop(columns=columns_to_drop)
    
    if verbose > 0:
        print("The following columns with a percentage of zero values greater "
              f"than {threshold}% have been removed:")
        # Print each column name in a new line
        print('\n'.join([str(col) for col in columns_to_drop]))  
        print('\n')
    
    if apply_rename:
        df = rename_columns(df, columns_to_drop, verbose=verbose)

    return df

def rename_columns(
    df: _pd.DataFrame, 
    zero_columns: _pd.Index,
    verbose: int = 0
) -> _pd.DataFrame:
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
    unique_zero_columns_prefix = _np.unique(unique_zero_columns_prefix)
    
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

def get_correlation_df(
    df: _pd.DataFrame,
    col_type: str
) -> _pd.DataFrame:
    """
    Get the correlation DataFrame for a specific column type in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - col_type (str): Type of LET-total column to include ('track' or 'dose').

    Returns:
    - DataFrame: A DataFrame containing correlation coefficients with the LET total 
            feature for each column in the DataFrame.
    """
    # Create the LET total feature name
    let_total_feat_name = f"L{col_type[0].upper()}T"
    
    # Calculate the correlation matrix for numeric columns
    let_total_corr_df = df.corr(numeric_only=True)[
        [let_total_feat_name]].sort_values(let_total_feat_name, ascending=False)
    
    return let_total_corr_df

def squeeze_and_rename_data_series(
    data_series: _pd.Series, 
    drop_value: float = 100.0, 
    verbose: int = 0,
) -> _pd.Series:
    """
    Squeeze a data series by dropping columns with values equal to a specified
    value and renaming the remaining columns. 

    Parameters:
    - data_series (pd.Series): Series containing the data to be squeezed.
    - drop_value (float, optional): Value used to determine which columns to
      drop. Columns with values greater than or equal this value will be
      dropped. Defaults to 100.0. 
    - verbose (int, optional): Set the verbosity level. Default is 0, which 
      means no output. 

    Returns:
    - pd.Series: Squeezed and renamed data series.
    """
    # Drop columns with values greater than or equal to the drop_value
    drop_columns = data_series.index[data_series >= drop_value]
    squeezed_series = data_series.drop(drop_columns)
    
    # Rename remaining columns
    squeezed_series = rename_columns(
        squeezed_series.to_frame().T, 
        drop_columns, 
        verbose=verbose
    ).squeeze()
    
    return squeezed_series
    