# src/utils/data_analysis_utils.py

import pandas as pd


def get_correlation_df(
    df: pd.DataFrame,
    col_type: str
) -> pd.DataFrame:
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