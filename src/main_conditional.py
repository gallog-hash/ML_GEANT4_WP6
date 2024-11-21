# %% [markdown]
# # Variational Autoencoder (VAE) for data augmentation
#
# **Citation:**
#
# - Author: Giuseppe Gallo
# - Title: Variational_Autoencoder_data_augmentation
# - Date: October, 2024
#

# %% [markdown]
# ## Introduce the work here

# Step 1: Setup and Configuration
# In this step, we set up the necessary configurations and import libraries.

# %%
import sys

import torch.utils

sys.path.append("../src")

import os
from typing import List, Union

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import display
from torchinfo import summary
from torchviz import make_dot

import autoenc as my_autoenc
import data_eng_pkg as data_eng
import eda_module as eda
import ml_prep as my_ml_prep
import my_plot_pkg as myplt
import vae_module as my_vae

# %%
# Define the global variables at the top of the module
random_seed = 42
device = 'cpu'

# %%
def change_default_settings(seed):
    # Declare it as global inside the function if you want to change it 
    global random_seed, device
    random_seed = seed
    
    # Change the default dpi setting in matplotlib and seaborn
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

    sns.set_theme(rc={'figure.dpi':100, 'savefig.dpi':300})
    sns.set_context('notebook')
    sns.set_style('ticks')

    matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    display(device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)

# %% [markdown]    
# Step 2: Define Helper Functions
# Here, we define the helper functions for data exploration and VAE training.

def mandatory_let_df_manipulation(
    let_df: pd.DataFrame,
    vsize_um: float,
    let_type: str,
    primary_pattern: str,
) -> tuple:
    """
    Apply essential preprocessing steps to the LET (Linear Energy Transfer)
    DataFrame, including feature transformation, filtering, and extraction of
    key data columns. 

    Args:
    ----
    - let_df (pd.DataFrame): Input DataFrame containing LET values for analysis. 
    - vsize_um (float): Voxel size in micrometers for converting index values to
      depth. 
    - let_type (str): Type of LET data to extract, either 'track' or 'dose',
      which determines the subset of columns selected.
    - primary_pattern (str): Regex pattern matching primary particle identifiers
      in column names, used for zero-value analysis.

    Returns:
    -------
        tuple: A tuple containing:
            - pd.DataFrame: Processed LET DataFrame with non-essential
              zero-value columns removed. 
            - float: Zero threshold percentage used to drop columns with
              excessive zero values. 
            - pd.DataFrame: DataFrame showing the percentage of zero values in
              columns corresponding to the primary particle.
            - pd.DataFrame: DataFrame showing the percentage of zero values for
              each column in the full dataset. 

    Process:
    -------
        1. Index Transformation: The index column `i` in `let_df` is converted
           to a depth feature `x`, allowing voxel indexing to reflect depth in
           millimeters. 
        2. Column Extraction: Columns specific to the `let_type` ('track' or
           'dose') are extracted for further analysis.
        3. Zero Value Analysis: Computes the percentage of zero values per
           column, identifying sparsity across features.
        4. Primary Particle Analysis: Isolates columns matching
           `primary_pattern` to compute zero-value percentages specifically for
           primary particle data. 
        5. Zero Value Filtering: Filters out columns where the percentage of
           zero values exceeds a threshold, set by the maximum zero-value
           percentage in primary particle columns (capped at 99.9%). 

    Notes:
    -----
        - This function prepares LET data for exploratory and statistical
          analysis by reducing sparsity, ensuring relevant depth features, and
          extracting meaningful track or dose data.
        - The transformation from `i` to `x` depth values supports visualization
          and analysis by converting voxel-based indices to metric depth.

    """

    # Transform the index feature 'i' in the df DataFrame into a depth feature
    # 'x'.  
    df = eda.convert_index_feature_to_depth(
        let_df, vsize_um, 
        input_size='um', 
        output_size='mm'
    )
        
    # Extract columns related to the specified quantity
    df = eda.extract_track_or_dose_cols(df, column_type=let_type)
    
    # Get the percentage of zero values per column in the full Dataframe. 
    percentage_eq_to_zero_df = eda.calculate_zero_values_percentage(df)
    
    # Get the percentage of zero values in the primary particle columns.
    primary_zero_perc = percentage_eq_to_zero_df[
        percentage_eq_to_zero_df.index.str.contains(primary_pattern)
    ]
    
    # Remove columns with a percentage of values equal to 0 higher than
    # threshold.
    zero_thr_percent = np.min([primary_zero_perc.max(), 99.9])
    df = eda.drop_zero_columns(df, threshold=zero_thr_percent, 
                               apply_rename=True, 
                               verbose=True)
    
    return df, zero_thr_percent, primary_zero_perc, percentage_eq_to_zero_df

def plot_and_export(
    main_df: pd.DataFrame,
    dose_profile: pd.DataFrame,
    fluence_df: pd.DataFrame,
    let_type: str,
    feature_on_single_plot: str,
    element_list: List[str], 
    n_features_per_plot: int,
    primary_x_stop_mm: float,
    cut_in_um: int, 
    voxel_in_um: int,
    twin_ylabel: str,
    twin_plotlabel: str,
    title_fontsize: int,
    legend_fontsize: int,
    xlim: List[float], 
    xlabel_fontsize: int,
    xticks_fontsize: int,
    ylabel_fontsize: int,
    yticks_fontsize: int,
    save_dir: str,
    more_feat_figsize: tuple[float]=(12,14),
    more_df_figsize: tuple[float]=(17,10),
    subplot_specs: Union[dict, None]=None,
    secondary_let_df: Union[List[pd.DataFrame], None]=None,
    secondary_dose_df: Union[List[pd.DataFrame], None]=None,
    secondary_plot_suffix: Union[List[str], None]=None,
    export_plots: bool=False,
):
    """
    Plot and export various figures for data exploration and visualization.

    Parameters:
    - main_df (DataFrame): The main DataFrame containing the LET data.
    - dose_profile (DataFrame): DataFrame containing the dose profile.
    - fluence_df (DataFrame): DataFrame containing the fluence distributions.
    - let_type (str): The type of LET to be plotted ('track' or 'dose').
    - feature_on_single_plot (str): Feature to be plotted in the first figure.
    - element_list (list of str): List of elements to be plotted.
    - n_features_per_plot (int): Number of features per plot.
    - primary_x_stop_mm (float): The depth at which the primary particle stops
    (in mm). 
    - cut_in_um (int): Cut in micrometers.
    - voxel_in_um (int): Voxel size in micrometers.
    - twin_ylabel (str): Y-axis label for the twin plot.
    - twin_plotlabel (str): Plot label for the twin plot.
    - title_fontsize (int): Font size for the plot title.
    - legend_fontsize (int): Font size for the legend.
    - xlim (list of float): X-axis limits.
    - xlabel_fontsize (int): Font size for the X-axis label.
    - xticks_fontsize (int): Font size for the X-axis ticks.
    - ylabel_fontsize (int): Font size for the Y-axis label.
    - yticks_fontsize (int): Font size for the Y-axis ticks.
    - save_dir (str): Directory where the plots will be saved.
    - more_feat_figsize (tuple(float,float), optional): Size of single
    elements plot. Default to (12, 14).
    - more_df_figsize (tuple(float,float), optional): Size of more dataframe
    plot. Default to (17, 10).
    - subplot_specs (dict, optional): Specification for optional subplots
    (location, x_range). Default to None.
    - secondary_let_df (list of DataFrame, optional): List of DataFrames
    containing secondary LET data for comparison. Default to None.
    - secondary_dose_df (list od DataFrame, optional): List of DataFrames
    containing secondary dose data for comparison. Default to None. 
    - secondary_plot_suffix (list of str, optional): List of suffixes for
    labeling secondary plots. Default to None. 
    - export_plots (bool, optional): Whether to export plots. Defals to False.

    Returns:
    - None
    """
    # Plot the distribution of the primary particle to show where it stops.
    cfig = myplt.plot_feature_distribution(
        main_df, 
        dose_profile=dose_profile,
        feature=feature_on_single_plot,
        subplot_location=subplot_specs.get('feat_dist_location', None),
        subplot_x_range=subplot_specs.get('feat_dist_x_range', None),
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        figsize=(12, 7),
        twin_ylabel=twin_ylabel,
        twin_plotlabel=twin_plotlabel,
        title_fontsize=title_fontsize, 
        legend_fontsize=legend_fontsize,
        xlim=xlim, 
        xlabel_fontsize=xlabel_fontsize, 
        xticks_fontsize=xticks_fontsize, 
        ylabel_fontsize=ylabel_fontsize, 
        yticks_fontsize=yticks_fontsize
    )

    if export_plots:
        filename = f"{feature_on_single_plot}_distribution_c{int(cut_in_um)}"
        filename += f"_v{int(voxel_in_um)}_d{int(xlim[-1])}.png"
        myplt.save_figure_to_file(cfig, filename, save_dir)
    
    # Visualize zero values in the main DataFrame
    cfig = myplt.visualize_zero_values(
        main_df, 
        column_as_index='x', 
        cmap='rocket',
        invert_cmap=True, 
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        primary_x_stop=primary_x_stop_mm * 1e3 / voxel_in_um
    )
    
    if export_plots:
            filename = f"zeros_heatmap_c{int(cut_in_um)}_v{int(voxel_in_um)}_"
            filename += f"d{int(xlim[-1])}.png"
            myplt.save_figure_to_file(cfig, filename, save_dir)
    
    # Plot the LET distribution of selected elements
    let_type_suffix = '_T' if let_type == 'track' else '_D'
    feature_list = [element + let_type_suffix for element in element_list]
    ylabel_single_element = "Let single elements [keV $\mu$m$^{-1}$]"
    
    fig_list = myplt.plot_more_features(
        main_df, 
        dose_profile=dose_profile,
        feature_list=feature_list,
        ylabel=ylabel_single_element,
        marker_size=25, 
        marker_types=['.','v'],
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        figsize=more_feat_figsize,
        yscale='log',
        n_feat_per_plot=n_features_per_plot,
        twin_ylabel=twin_ylabel,
        twin_plotlabel=twin_plotlabel,
        twin_color='grey', 
        twin_alpha=0.8, 
        twin_linewidth=2,
        title_fontsize=title_fontsize, 
        legend_fontsize=legend_fontsize,
        xlim=xlim, 
        xlabel_fontsize=xlabel_fontsize, 
        xticks_fontsize=xticks_fontsize, 
        ylabel_fontsize=ylabel_fontsize, 
        yticks_fontsize=yticks_fontsize
    )
    
    if export_plots:
        filename = f"single_elements_yscale_log_c{int(cut_in_um)}_"
        filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}.png"
        myplt.save_figure_to_file(fig_list, filename, save_dir)
    
    # Plot all selected elements with a logarithmic scale on the y-axis
    cfig = myplt.plot_more_features(
        main_df, 
        dose_profile=dose_profile,
        feature_list=feature_list,
        ylabel=ylabel_single_element,
        marker_size=9, 
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        figsize=more_feat_figsize,
        yscale='log',
        twin_ylabel=twin_ylabel,
        twin_plotlabel=twin_plotlabel,
        twin_color='grey', 
        twin_alpha=0.8, 
        twin_linewidth=2,
        title_fontsize=title_fontsize, 
        legend_fontsize=legend_fontsize,
        xlim=xlim, 
        xlabel_fontsize=xlabel_fontsize, 
        xticks_fontsize=xticks_fontsize, 
        ylabel_fontsize=ylabel_fontsize, 
        yticks_fontsize=yticks_fontsize
    )
    
    if export_plots:
        filename = f"plot_more_features_yscale_log_c{int(cut_in_um)}_"
        filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}.png"
        myplt.save_figure_to_file(cfig, filename, save_dir)

    # Prepare feature list for fluence plot
    fluence_feat_list = [f.replace('_T', '_f') for f in feature_list]
    
    # Plot fluence of the selected elements
    fig_list = myplt.plot_more_features(
        fluence_df, 
        dose_profile=dose_profile, 
        let_total_profile=main_df[['x', 'LTT']].copy(),
        feature_list=fluence_feat_list,
        ylabel="Fluence [counts]",
        marker_size=25, 
        marker_types=['.','v'],
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        figsize=more_feat_figsize,
        yscale='log',
        n_feat_per_plot=n_features_per_plot,
        twin_ylabel=twin_ylabel,
        twin_plotlabel=twin_plotlabel,
        twin_color='grey', 
        twin_alpha=0.8, 
        twin_linewidth=2,
        title_fontsize=title_fontsize, 
        legend_fontsize=legend_fontsize,
        xlim=xlim, 
        xlabel_fontsize=xlabel_fontsize, 
        xticks_fontsize=xticks_fontsize, 
        ylabel_fontsize=ylabel_fontsize, 
        yticks_fontsize=yticks_fontsize
    )

    if export_plots:
        filename = f"fluence_more_features_yscale_log_c{int(cut_in_um)}_"
        filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}.png"
        myplt.save_figure_to_file(fig_list, filename, save_dir)
        
    if secondary_let_df is not None:       
        fig_list = myplt.plot_more_dataframe(
            df_list=[main_df] + secondary_let_df,
            feature_list=feature_list,
            ylabel=ylabel_single_element,
            marker_size=25,
            dose_profiles=[dose_profile] + secondary_dose_df,
            n_feat_per_plot=1,
            plot_suffix=secondary_plot_suffix,
            figsize=more_df_figsize,
            yscale='linear',
            twin_ylabel=twin_ylabel,
            twin_plotlabel=twin_plotlabel,
            twin_color='grey', 
            twin_alpha=0.8, 
            twin_linewidth=2,
            title_fontsize=title_fontsize, 
            legend_fontsize=legend_fontsize,
            xlim=xlim, 
            xlabel_fontsize=xlabel_fontsize, 
            xticks_fontsize=xticks_fontsize, 
            ylabel_fontsize=ylabel_fontsize, 
            yticks_fontsize=yticks_fontsize
        )
        
        if export_plots:
            filename = f"more_df_features_c{int(cut_in_um)}_"
            filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}.png"
            myplt.save_figure_to_file(fig_list, filename, save_dir)
    

def outliers_search(
    let_df: pd.DataFrame,
    let_type: str,
    outliers_method_list: List[str],
    replace_method_list: List[str],
    primary_x_stop_mm: float,
    cut_in_um: int,
    voxel_in_um: int,
    let_upper_limit: float=100.0,
    lof_neighbors: int=20,
    dbscan_eps: float=0.5,
    dbscan_min_samples: int=5,
    knn_neighbors: int=5,
    verbose: int = 0,
):
    """
    Run the outliers search over all elements of outliers_method_list and
    replace_method_list. 

    Parameters:
    - let_df (DataFrame): DataFrame containing LET data.
    - let_type (str): The type of LET to be analyzed ('track' or 'dose').
    - outliers_method_list (list of str): List of methods for identifying
    outliers. 
    - let_upper_limit (float): Upper limit for LET values. Defaults to 100.0.
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
    - replace_method_list (list of str): List of methods for replacing outliers.
    - primary_x_stop_mm (float): The depth at which the primary particle stops
    (in mm). 
    - cut_in_um (int): Cut in micrometers.
    - voxel_in_um (int): Voxel size in micrometers.
    - verbose (int, optional): Verbosity level for debugging information.

    Returns:
    - pd.DataFrame: The DataFrame with the best outlier-replacement method
    chosen by the user. 
    """
    # Check if the lengths of outliers_method_list and replace_method_list are
    # equal 
    if len(outliers_method_list) != len(replace_method_list):
        raise ValueError("The lengths of outliers_method_list and "
                         "replace_method_list must be equal.")
        
    # Convert verbose to boolean
    verbose_bool = verbose > 0

    results = {}

    for outliers_method, replace_method in zip(outliers_method_list, 
                                               replace_method_list):
        if outliers_method =='lof':
            cbar_label = "LOF Outliers Factor" 
        else:
            cbar_label = "Outliers Normalized Z-Score"
            
        key = f"{outliers_method}_{replace_method}"
        
        outliers_replacements = data_eng.identify_let_outliers(
            let_df, 
            column_type=let_type,
            outliers_method=outliers_method,
            let_upper_limit=let_upper_limit,
            lof_neighbors=lof_neighbors,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            knn_neighbors=knn_neighbors,
            replace_method=replace_method,
            x_bounds=(32.5, primary_x_stop_mm),
            cbar_label=cbar_label,
            cut_in_um=cut_in_um, 
            voxel_in_um=voxel_in_um,
            verbose=verbose_bool
        )
        if not outliers_replacements.empty:
            results[key] = outliers_replacements
      
    if len(results) == 0:
        print("\nNo outliers found.")
        return pd.Series()

    # List the available methods for the user
    print("\nAvailable outlier-replace methods:")
    for i, key in enumerate(results.keys(), 1):
        print(f"{i}. {key}")

    # Prompt the user to select the best method
    selected_index = int(input("\nSelect the number of the best outlier-replace "
                               "method: ")) - 1
    selected_key = list(results.keys())[selected_index]

    print(f"\nSelected method: {selected_key}")
    
    # Select the best outlier-replace method
    outliers_replacements_df = results[selected_key]
    
    # !!!!!
    # The following code is an unfinished job.
    #
    # Explore if the outliers in the total LET distribution are outliers for
    # signle element distribution also.
    element_list = ['proton_1_T', 'proton_T', 'alpha_T', 'deuteron_T', 'O16_T']
    
    for element in element_list:
        myplt.plot_feature_and_mark_outliers_by_let(
            df=let_df,
            outliers_and_replacements_df=outliers_replacements_df,
            feature=element,
            subplot_location=[0.2, 0.4, 0.45, 0.45],
            subplot_x_range=(32.5, 40.0)
        )

    # Return the selected DataFrame
    return outliers_replacements_df

def opt_objective(trial, let_df, optuna_params, verbose=False):
    
    input_data_shape = let_df.shape
    n_feat_in = input_data_shape[1]
    
    # Use optuna_params to define each parameter search space
    learning_rate = trial.suggest_float('learning_rate', 
                                        *optuna_params['learning_rate'],
                                        log=True)
    batch_size = trial.suggest_categorical('batch_size', 
                                          optuna_params['batch_size'])
    ld_scaling = trial.suggest_int('ld_scaling', 
                                   *optuna_params['latent_dim_scaling'])
    latent_dim = [n_feat_in // ld_scaling] 
    ns_upsample = trial.suggest_int('ns_upsample', 
                                    *optuna_params['net_size_upsample'])
    ns_scaling = trial.suggest_int('ns_scaling', 
                                   *optuna_params['net_size_scaling'])
    net_size = [n_feat_in**ns_upsample // ns_scaling, 
                n_feat_in**ns_upsample // ns_scaling, 
                n_feat_in**ns_upsample // ns_scaling]
    epochs = trial.suggest_categorical('epochs', optuna_params['epochs'])
    
    vae_params = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'net_size': net_size,
        'epochs': epochs,
    }
    
    # Call the training function and obtain loss
    history = train_vae(
        let_df=let_df,
        verbose=verbose,
        vae_plot_dir='../opt_plot',
        vae_history_dir='../opt_history',
        **vae_params
    )
    
    vae_loss = history['val_loss'][-1]
    
    return vae_loss # Minimizing the VAE loss

def train_vae(
    let_df,
    verbose,
    vae_plot_dir,
    vae_history_dir,
    **kwargs
):
    input_data_shape = let_df.shape
    n_feat_in = input_data_shape[1]
       
    # Default parameters
    default_train_size = 0.7 
    default_val_size = 0.2 
    default_test_size = 1 - default_train_size - default_val_size
    default_single_scaler = True
    default_batch_size = 100
    default_net_size = [n_feat_in**2 // 2, n_feat_in**2 // 2, n_feat_in**2// 2]
    default_latent_dim = n_feat_in // 2
    default_normalization = nn.BatchNorm1d
    default_activation = nn.ReLU
    default_output_activation = nn.ReLU
    default_learning_rate = 1e-4
    default_penalty_L2 = 0 
    default_training_epochs = 100
    default_train_report_every = 10
    
    # Retrieve parameters from kwargs or use defaults
    train_size = kwargs.get('train_size', default_train_size)
    val_size = kwargs.get('val_size', default_val_size)
    test_size = kwargs.get('test_size', default_test_size)   
    single_scaler = kwargs.get('single_scaler', default_single_scaler)
    batch_size = kwargs.get('batch_size', default_batch_size) 
    net_size = kwargs.get('net_size', default_net_size)
    latent_dim = kwargs.get('letent_dim', default_latent_dim)
    normalization = kwargs.get('normalization', default_normalization)
    activation = kwargs.get('activation', default_activation)
    output_activation = kwargs.get('output_activation', default_output_activation)
    learning_rate = kwargs.get('learning_rate', default_learning_rate)
    penalty_L2 = kwargs.get('penalty_L2', default_penalty_L2)
    training_epochs = kwargs.get('training_epochs', default_training_epochs)
    train_report_every = kwargs.get('train_report_every', default_train_report_every)
    
    # Split the dataset into train, validation nad test subsets
    X_train, X_val, X_test = my_ml_prep.train_val_test_split(
        let_df,             # Input dataset
        train_size=train_size,     
        val_size=val_size,       
        test_size=test_size,      
        random_state=random_seed,   # Random seed for reproducibility
        shuffle=True        # Whether to shuffle the dataset before splitting
    )
    
    # Apply scaling to the train, validation and test subsets
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = \
        my_ml_prep.train_val_test_scale(
            training_data=X_train,    
            validation_data=X_val,
            test_data=X_test,
            single_scaler=single_scaler, 
            scaler_type='standard'
        )
        
    # Plot how the dataset was splitted into train, test, and validation datasets
    myplt.plot_train_test_val_distribution(
        df_before_split = let_df,
        feature_names = ['x', 'LTT'],
        X_train = X_train,
        X_test = X_test,
        X_val = X_val,
    )      
    
    # Convert data arrays into DataLoader
    #
    # In this section, we're converting our training, validation, and test
    # datasets (`X_train`, `X_val`, and `X_test`) into PyTorch DataLoader
    # objects. This step is crucial for efficiently handling our data during
    # model training and evaluation. 
    #
    # By converting our data into DataLoader objects, we're enabling batch
    # processing, which enhances the efficiency of our model training process.
    # DataLoader allows us to iterate over our datasets in mini-batches, rather
    # than processing the entire dataset at once. This not only conserves memory
    # but also facilitates parallel processing, leading to faster training
    # times.  
    #
    # Additionally, DataLoader provides built-in functionality for shuffling our
    # data, which is essential for preventing the model from learning sequential
    # patterns and biases in the data. This ensures that our model generalizes
    # well to unseen data. 
    #
    # Overall, converting our data to DataLoader objects is a key preprocessing
    # step that sets the stage for efficient and effective model training and
    # evaluation. 
    
    train_loader = my_ml_prep.create_data_loader(
        data=X_train_scaled,
        data_type='train',
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = my_ml_prep.create_data_loader(
        data=X_val_scaled,
        data_type='validation',
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = my_ml_prep.create_data_loader(
        data=X_test_scaled,
        data_type='test',
        batch_size=batch_size,
        shuffle=False
    )
        
    # VAE model parameters initialization
    net_params= {
        'input_dim': n_feat_in,
        'hidden_layers_dim': net_size,
        'latent_dim': latent_dim,
        'normalization': normalization,
        'activation': activation,
        'output_activation': output_activation,
    }    
    
    # Adam Optimizer parameters
    optimizer_params = {
        'lr1': learning_rate,
        'l2_reg1': penalty_L2, # try 1e-4, default is 0
    }
      
    # Initialize the VAE model
    vae = my_vae.AutoEncoder(
        net_params=net_params,
        optimizer_params=optimizer_params,
        device=device
    ).to(device)
    
    # Create model summary
    vae_summary = summary(model=vae, input_size=(1, n_feat_in), depth=4)
    
    # Get the rapresentation of the summary as a string
    summary_str = str(vae_summary)
    
    # Export the model summary to a file
    summary_filename = 'model_summary.txt'
    with open(summary_filename, 'w') as file:
        file.write(summary_str)
        
    # Train the model
    history = vae.fit(
        trainloader=train_loader, 
        num_epochs=training_epochs, 
        valloader=val_loader, 
        verbose=True, # Change verbose at the end of the job
        show_every=train_report_every
    )
    
    # Plot both training and validation losses on the same plot
    my_autoenc.plot_history(
        history,
        train_losses = True,
        val_losses = True,
        train_time_per_epoch = False,
        gen_losses = False,
        kl_losses = False,
        gamma_x_list = False,
        x_mse_losses_recon = False,
    )
    
    # Export model parameters and history
    my_autoenc.save_model_and_history(
        model=vae,
        history=history,
        filename=os.path.join(vae_history_dir, 'balancing_vae')
    )
    
    # Plot latent space 
    my_autoenc.plot_latent_space(vae, train_loader, use_tsne=False, 
                    figsize=(10, 8), marker='.', alpha=0.5)
    
    # Plot comparison between test data and predicted by VAE
    test_recon, _ = vae.reconstruct(test_loader)
    my_autoenc.plot_test_and_reconstructed(
        test_data=test_loader,
        recon_data=test_recon,
        scaler=scaler,    
        train_data=train_loader,
    )
       
    if verbose > 0:
        print("\nType ans shape of DataLoaders.dataset: ")
        print(f" - train: {type(train_loader.dataset.data)}, ", end='')
        print(f"{train_loader.dataset.data.shape}")
        print(f" - validation: {type(val_loader.dataset.data)}, ", end='')
        print(f"{val_loader.dataset.data.shape}")
        print(f" - test: {type(test_loader.dataset.data)}, ", end='')
        print(f"{test_loader.dataset.data.shape}")
        print("\nVAE model summary: ")
        print(summary_str)
        print("\n=>=>=>=>=> Model summary saved as: ", summary_filename)
        print("\nList of VAE model parameters")
        my_autoenc.print_model_parameters(vae)
        
    if verbose > 1:
        # Visualize the computation graph
        # - Create input and output toy tensors 
        input_tensor = torch.rand(
            train_loader.dataset.data.shape[0], n_feat_in).to(device) 
        output_tensor = vae(input_tensor)
        # - Generate the graph      ---> To be removed
        dot = make_dot(output_tensor, params=dict(vae.named_parameters()))
        dot.attr(rankdir='LR') # set the rank direction ('TP' or 'LR')
        dot.attr(dpi='300') # set the DPI
        dot.render(os.path.join(vae_plot_dir,"vae_graph_LR"), format="png")
        
    return history

def ensure_directory_exists(directory_path):
    """
    Checks if a directory exists at the given path, and creates it if it does not.

    Parameters:
    - directory_path (str): The path of the directory to check or create.

    Returns:
    - None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created at: {directory_path}")

# %% [markdown]
# ## Exploratory Data Analysis (EDA)


# %% [markdown]
# ## Data engineering


# %% [markdown]
# ## Data Preparation for Machine Learning
#
# In this section, we'll cover the initial steps of data preparation for machine
# learning. This includes splitting the data into training, validation, and test
# sets, as well as scaling the features. These steps are essential for
# optimizing model training and ensuring reliable performance. 
#
# Splitting the data serves to create distinct subsets that fulfill different
# roles in the model development pipeline. The training set forms the bedrock
# upon which the model is trained, enabling it to learn patterns and
# relationships within the data. The validation set acts as a checkpoint during
# training, providing feedback on the model's performance and aiding in
# hyperparameter tuning. Finally, the test set serves as an independent
# benchmark to assess the model's generalization and performance on unseen data. 
# 
# Beyond partitioning the data, scaling features is essential for ensuring
# consistent and meaningful comparisons between them. Feature scaling techniques
# normalize the range of features, preventing certain features from dominating
# others and facilitating convergence during model training. By bringing
# features onto a similar scale, scaling enhances the stability and efficiency
# of the learning process, leading to more robust and reliable models.  
#
# %%

# %% [markdown]
# ### Converting Data to DataLoader


# %% [markdown]
# ## Build model
#
# If you want to better understand the variational autoencoder technique, look
# [here](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73).
#
# To better understand this Autoencoder Class, let me go through it briefly.
# This is a Variational Autoencoder (VAE) with two hidden layers, which (by
# default, but you can change this) 50 and then 12 activations. The latent factors
# are set to 3 (you can change this, too). So we'll first explode our initial
# 14 variables to 50 activations, then compress them to 12, then to 3. From these
# 3 latent factors we then sample to recreate the original 14 values. We do this
# by exploding the 3 latent factors back to 12, then to 50 and finally to 14
# activations (decoding the latent factors, so to speak). With this reconstructed
# batch (recon_batch), we compare it to the original batch, calculate our loss,
# and adjust the weights and biases via our gradient (our optimizer here will be
# Adam). 

# %% [markdown]
# ### Import the VAE architecture from dedicated module

# %%


# %%
# Initialization


# %%
# Apply custom weight initialization to the model
# model.apply(weights_init_uniform_rule)

# %% [markdown]
# The naming of the return parameter `loss_mse` is somewhat misleading, and it
# might be confusing if you are looking at it for the first time. It's common to
# name the return value based on the primary loss component, and in this case,
# the primary component is the Mean Squared Error (MSE) loss. 
#
# However, it's important to note that `loss_mse` does not specifically represent
# just the MSE loss; it represents the total loss. The naming might be a bit
# misleading in this context.
#  
# If it helps with clarity, you could choose to name it differently.


# %% [markdown]
# ### Latent space visualization
# Visualizing the latent space of a VAE involves plotting or projecting the
# learned latent representations onto a lower-dimensional space for
# visualization. Commonly, the latent space has dimensions specified by the
# `latent_dim` parameter in your VAE model.
#
# Here's a general approach to visualize the latent space:
# 1. **Encode Data:**
#    - Pass your data through the encoder part of the trained VAE model to obtain
#      the corresponding latent representation.
# 2. **Scatter Plot or Visualization:**
#    - If the latent space dimensionality is 2 or 3, you can create a scatter
#      plot to visualize the data points in the latent space.
#    - If the latent space has more dimensions, you might consider dimensionality
#      reduction techniques (e.g. PCA,t-SNE) to project the data to 2 or 3
#      dimensions before visualization.

# %%
# 3D Latent Space Visualization
# visualize_latent_space(vae_model=model, data_loader=train_loader, device=device,
#                       dimensionality=3, s=12, c=X_train_inverted[:, 0], 
#                       cbar_label='x [mm]')

# %% [markdown]
# In a well-trained Variational Autoencoder (VAE), the latent space
# visualization should exhibit certain characteristics: 
#
# 1. **Clustering:** Similar data points in the input space should be close to
#    each other in the latent space. Clusters of points may represent different
#    classes or variations in the data. 
#
# 2. **Smooth Transitions:** Points in the latent space along a trajectory
#    should correspond to smooth transitions in the input space. This property
#    indicates that the model has learned a continuous and meaningful
#    representation. 
#
# 3. **Separation of Classes:** If the dataset has distinct classes or
#    categories, the latent space should show clear separations between these
#    classes. This separation allows for effective generation and interpolation
#    between different classes. 
#
# 4. **Interpolation:** By moving along a straight line or trajectory in the
#    latent space, generated samples should smoothly transition between
#    different data instances. This showcases the model's ability to generate
#    diverse and realistic samples. 
#
# 5. **Consistency:** Repeated patterns or structures in the latent space
#    indicate that the VAE has learned meaningful representations that capture
#    essential features of the data. 
#
# It's important to note that the interpretation of the latent space depends on
# the nature of the data and the specific goals of the VAE. The latent space
# visualization is a valuable tool for assessing the quality of the learned
# representations and gaining insights into the underlying structure of the
# data. 
#

# %% [markdown]
# Step 3: Main Function
# The main function initializes settings, imports the dataset, and optionally
# runs data exploration and VAE training.

# %%
def main(
    run_data_exploration=False, 
    run_outliers_search=False,
    run_vae_training=True,
    optimize_vae=False,
    n_trials=50,
    optuna_params=None,
    data_dir='../data/t96_1e8_c1000_v100_d50',
    data_file='Let_1000-100.out',
    dose_file='Dose_1000-100.out',
    subplot_specs=None,
    more_feat_figsize=None,
    n_elements_per_plot=1,
    secondary_data_dirs=None,
    secondary_data_files=None,
    secondary_dose_files=None,
    more_df_figsize=None,
    df_labels=None,
    let_type='track',
    element_list=None,
    eda_xlim=(None, None),
    outliers_params=None,
    vae_params=None,
    random_seed=42, # Default random seed
    verbose=0,
    export_plots=False,
    eda_plot_dir=None,
    vae_plot_dir=None,
    vae_history_dir=None,
):
    """
    Main function for data exploration, outlier detection, and Variational
    Autoencoder (VAE) training on LET (Linear Energy Transfer) data files.

    Parameters:
    -----------
    run_data_exploration : bool, optional
        Whether to perform exploratory data analysis on the loaded dataset
        (default is False). 
        
    run_outliers_search : bool, optional
        Whether to perform outlier detection and replacement on the LET dataset
        (default is False). 
        
    run_vae_training : bool, optional
        Whether to run the Variational Autoencoder training on the dataset
        (default is True). 
    
    optimize_vae : bool, default=False
        If True, activates the Optuna optimization routine for the VAE model. 
        This optimization attempts to improve model performance by tuning 
        hyperparameters across multiple trials, leveraging Optuna's 
        Bayesian optimization framework.
        
    n_trials : int, default=50
        The number of trials to execute when `optimize_vae` is set to True. 
        Each trial will test a different combination of hyperparameters, 
        up to the maximum specified by this parameter.
        
    optuna_params : dict, optional
        Dictionary defining the ranges or options for hyperparameters to be 
        optimized when `optimize_vae` is True. Keys represent the parameter 
        names, and values specify either a range (as a tuple for continuous 
        values, e.g., (0.001, 0.1) for a learning rate) or a list of possible 
        choices for categorical parameters (e.g., [16, 32, 64] for batch size).
        
    data_dir : str, optional
        Directory containing the primary LET data files (default is
        '../data/t96_1e8_c1000_v100_d50'). 
        
    data_file : str, optional
        Primary LET data file to be loaded (default is 'Let_1000-100.out').
        
    dose_file : str, optional
        File containing dose profile and fluence distributions (default is
        'Dose_1000-100.out'). 
        
    subplot_specs : dict, optional
        Specifications for subplot arrangement, with keys: 
        - 'feat_dist_location': List of 4 floats specifying location and size
        for feature distribution plot. 
        - 'feat_dist_x_range': List of two floats defining the x-axis range for
        feature distribution plot. 
        - 'let_prof_location': List of 4 floats specifying location and size for
        LET profile plot. 
        - 'let_prof_x_range': List of two floats defining the x-axis range for
        LET profile plot. 
        
    more_feat_figsize : tuple, optional
        Figure size for plotting additional features (default is None).
        
    n_elements_per_plot : int, optional
        Number of elements to plot per subplot (default is 1).
        
    secondary_data_dirs : list of str, optional
        List of directories containing secondary LET data files (default is None).
        
    secondary_data_files : list of str, optional
        List of secondary LET data files (default is None).
        
    secondary_dose_files : list of str, optional
        List of secondary dose files for each secondary LET data file (default
        is None). 
        
    more_df_figsize : tuple, optional
        Figure size for additional dataframes (default is None).
        
    df_labels : list of str, optional
        Labels for secondary data frames (default is None).
        
    let_type : str, optional
        Type of LET data to analyze (default is 'track').
        
    element_list : list of str, optional
        List of elements to include in the analysis (default is None).
        
    eda_xlim : tuple, optional
        x-axis limits for exploratory data analysis plots (default is (None, None)).
        
    outliers_params : dict, optional
        Parameters for outlier detection, including methods and thresholds
        (default is None). 
        
    vae_params : dict, optional
        Parameters for VAE training, such as latent dimensions and learning rate
        (default is None). 
        
    random_seed : int, optional
        Seed for random number generation (default is 42).
        
    verbose : int, optional
        Verbosity level for logging details; higher values provide more details
        (default is 0). 
        
    export_plots : bool, optional
        Whether to export generated plots to files (default is False).
        
    eda_plot_dir : str, optional
        Directory to save exploratory data analysis plots if export_plots is
        True (default is None). 
        
    vae_plot_dir : str, optional
        Directory to save VAE-related plots if export_plots is True (default is
        None). 
        
    vae_history_dir : str, optional
        Directory to save VAE training history if export_plots is True (default
        is None). 

    Returns:
    --------
    None
        The function performs data loading, analysis, and optional VAE training
        without returning any value. 

    Notes:
    ------
    - Data files should be in the '.out' format, containing LET, dose, and
      fluence distribution data. 
    - The function provides detailed logging at various stages if 'verbose' is
      greater than 0. 
    - If 'run_data_exploration' or 'run_outliers_search' is enabled, exploratory
      plots and outlier replacement plots can be saved when 'export_plots' is
      True. 
    - For secondary data analysis, secondary LET data directories, data files,
      and dose files must be specified if 'run_data_exploration' is enabled.
    """
    
    # Check if export directories exist, if provided
    if eda_plot_dir is not None:
        ensure_directory_exists(eda_plot_dir)
    if vae_plot_dir is not None:
        ensure_directory_exists(vae_plot_dir)
    if vae_history_dir is not None:
        ensure_directory_exists(vae_history_dir)
            
    # Set default export directories to current working directory if not provided
    if export_plots and eda_plot_dir is None:
        eda_plot_dir = os.getcwd()
    if vae_plot_dir is None:
        vae_plot_dir = os.getcwd()
    if vae_history_dir is None:
        vae_history_dir = os.getcwd()
        
    change_default_settings(seed=random_seed)

    run_identifier = '_'.join(data_dir.split('_')[-2:])
    dataset_identifier = '/'.join([run_identifier, data_file])

    # Import the main data file containing the total LET and individual element
    # distributions used in the Variational Autoencoder training procedure.
    # Use specialized function to import *.out file.
    try: 
        df = eda.import_out_file(os.path.join(data_dir, data_file))
    except OSError as e:
        print(f"Data import failed: {e}")
        return
        

    if verbose > 0:
        print("DataFrame with loaded data from file: ", dataset_identifier)
        print(df.head(10)) 
        
    cut_in_um, voxel_in_um = eda.get_cut_voxel_from_filename(data_file)
    
    # Define the primary particle shot in the Geant4 simulation
    primary_pattern = 'proton_1'
    
    df_non_zero, zero_thr_percent, primary_zero_perc, percentage_eq_to_zero_df = \
        mandatory_let_df_manipulation(
            let_df=df,
            vsize_um=voxel_in_um,
            let_type=let_type,
            primary_pattern='proton',
        )

    """
    # Transform the index feature 'i' in the df DataFrame into a depth feature
    # 'x'.  
    # This conversion uses voxel size in micrometers (voxel_in_um) and converts 
    # the depth to millimeters. 
    df = eda.convert_index_feature_to_depth(
        df, voxel_in_um, 
        input_size='um', 
        output_size='mm'
    )
    """
    
    # Get the detector depth to set the xlim for plotting
    max_depth_mm = np.ceil(df_non_zero['x'].max())
    xlim = [0.0, max_depth_mm]
    
    # Check if xlim has been specified by the user
    if eda_xlim[0] is not None:
        xlim[0] = eda_xlim[0]
    if eda_xlim[1] is not None:
        xlim[1] = eda_xlim[1]
    
    # Load the file containing the dose profile and fluence distributions of
    # each element. These data will not be used in the Variational
    # Autoencoder routine, but will be plotted for reference in some of the
    # figures below. 
    dose_df = eda.import_out_file(os.path.join(data_dir, dose_file))
    
    # Transform the index feature 'i' in the dose_df DataFrame into a depth
    # feature 'x'.  
    dose_df = eda.convert_index_feature_to_depth(
        dose_df, voxel_in_um, 
        input_size='um',                 
        output_size='mm'
    )

    # Extract the dose profile data into a separate DataFrame.
    # 'x' represents the depth in mm, and 'Dose(Gy)' represents the dose in 
    # Gray units. 
    dose_profile = dose_df[['x', 'Dose(Gy)']].copy()
    
    if verbose > 0: 
        # Dataset Overwiew
        print("\nDataset identifier: {}".format(dataset_identifier))
        print("Cut value extracted from filename: ", cut_in_um)
        print("Voxel size value extracted from filename: ", voxel_in_um)
        print(df.head())    
        eda.print_columns_with_pattern(df, primary_pattern)   
        #

    # Remove from DataSeries the features with all zeros ((% == 0) = 100.0)
    percentage_eq_to_zero_df_squeezed = eda.squeeze_and_rename_data_series(
        percentage_eq_to_zero_df,
        drop_value=zero_thr_percent,
        verbose=True
    )
    
    # Let's take a look at the overall LET profile
    cfig = myplt.plot_let_profile(df_non_zero, column_type=let_type,
        dose_profile=dose_profile,
        subplot_location=subplot_specs.get('let_prof_location', None),
        subplot_x_range=subplot_specs.get('let_prof_x_range', None),
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        twin_ylabel='Dose [Gy]', 
        twin_plotlabel='Dose',
        title_fontsize = 20, 
        legend_fontsize=16,
        xlim=xlim, 
        xlabel_fontsize=18, 
        xticks_fontsize=15, 
        ylabel_fontsize=18, 
        yticks_fontsize=15
    )

    if export_plots:
        filename = f"let_profile_c{int(cut_in_um)}_v{int(voxel_in_um)}_"
        filename += f"d{int(max_depth_mm)}.png"
        myplt.save_figure_to_file(cfig, filename, eda_plot_dir)
        
    # Extract the portion of the main DataFrame up to the first minimum value
    # after the peak in the most significant primary particle's column. 
    # [Calling this method after dropping columns full of null values and
    # squeezing the remaining ones ensures that there is a single primary
    # particle column in the DataFrame.] 
    df_early_stop = eda.cut_dataframe_with_primary(
        df_non_zero, column_type=let_type,                             
        primary_particle='proton')
    
    primary_x_stop_mm = df_early_stop['x'].iloc[-1]
    
    if verbose > 0:
        # Display filtered DataFrame
        print("\nDataFrame updated after removing column not related to "
              f"'{let_type}':")
        print(df.head())
        
        # Print statement to introduce the 'describe()' method
        print("\nSummary statistics of the updated DataFrame:")
        # Use the Pandas library to take a first look at the data.
        # The 'describe()' method returns a table containing summary statistics
        # about a DataFrame, such as mean, maximum and minimum values. At the
        # top of the table is the 'counts' row. Different count values indicate
        # that some feature in the DataFrame has missing values. 
        df.describe().transpose()
        
        # Print statement to introduce the 'info()' method
        print("\nInformation summary of the updated DataFrame:")
        # The `info()` method returns a summary of the dataframe as well as a
        # count of the non-null values. 
        df.info()

        # Print the percentage of zero values in the primary particle columns
        print("\nPercent of zero values in the primary particle columns:")
        for index, value in primary_zero_perc.items():
            print(f"{index}: {value}")
        print(f"\nColumns with al least {zero_thr_percent:.3f}% of null values "
              "were dropped.")
        
        print(f"\nPrimary particle stops at {primary_x_stop_mm:.3f} mm.")
        
        # Get the percentage of non-zero values after the primary particle stop.
        perc_non_zero_after_pp = eda.calculate_zero_values_percentage(
            df=df_non_zero,
            x_threshold=primary_x_stop_mm
        )
        perc_non_zero_after_pp.drop(labels=['x', 'LTT'], inplace=True)
        perc_non_zero_after_pp = 100.0 - perc_non_zero_after_pp
        print("\nPercent of non-zero values after primary particle stop:")
        for index, value in perc_non_zero_after_pp.nlargest(15).items():
            print(f"{index}: {value}")


        
    if run_data_exploration:       
        # Create a barplot with the percentage of zero values for each feature
        # of the input Series
        cfig = myplt.barplot_zero_values_percentage(
            percentage_eq_to_zero_df, figsize=(15,6),
            cut_in_um=cut_in_um, 
            voxel_in_um=voxel_in_um
        )
        
        if export_plots:
            filename = f"barplot_zeros_c{int(cut_in_um)}_v{int(voxel_in_um)}_"
            filename += f"d{int(max_depth_mm)}.png"
            myplt.save_figure_to_file(cfig, filename, eda_plot_dir)
        
        # Create a correlation heatmap with LET total over the entire range
        heatmap_title_x_full = myplt.create_correlation_plot_title(
            df_non_zero, 'Correlation Heatmap', cut_in_um=cut_in_um, 
            voxel_in_um=voxel_in_um)

        let_total_corr_df = eda.get_correlation_df(df_non_zero, let_type)

        cfig = myplt.plot_correlation_df(let_total_corr_df, figsize=(4,10),
                                plot_title=heatmap_title_x_full)
        
        if export_plots:
            filename = f"correlation_heatmap_c{int(cut_in_um)}_"
            filename += f"v{int(voxel_in_um)}_d{int(max_depth_mm)}.png"
            myplt.save_figure_to_file(cfig, filename, eda_plot_dir)
        
        # We use combined_plot_comparison function to visualize the relationship
        # between correlation coefficients and the percentage of values equal to
        # 0 for each feature. These function help us compare two sets of data,
        # allowing us to observe any patterns or trends in the correlation
        # coefficients and their association with the percentage of zeros in the
        # data.  
        cfig = myplt.combined_plot_comparison(
            left_data=percentage_eq_to_zero_df_squeezed,
            right_data=let_total_corr_df,
            left_ylabel='Percentage of zero values',
            right_ylabel='Correlation with LTT',
            xlabel='Non-zero Features',
            cut_in_um=cut_in_um, 
            voxel_in_um=voxel_in_um,
            left_ylim=(0.0, 100.0),
            right_ylim=(0.0, 1.0)
        )
        
        if export_plots:
            filename = f"corr_vs_zeros_c{int(cut_in_um)}_"
            filename += f"v{int(voxel_in_um)}_d{int(max_depth_mm)}.png"
            myplt.save_figure_to_file(cfig, filename, eda_plot_dir)
        
        # Plot the LET profile after removing the portion of the DataFrame after
        # the primary particle stopped. 
        subplot_x_stop = np.ceil(primary_x_stop_mm *10) / 10
        subplot_x_range = subplot_specs.get('let_prof_x_range', None)
        if subplot_x_range:
            subplot_x_range[-1]=subplot_x_stop
            
        _ = myplt.plot_let_profile(
            df_early_stop, 
            column_type=let_type,
            dose_profile=dose_profile,
            subplot_location=[0.2, 0.35, 0.25, 0.50],
            subplot_x_range=subplot_x_range,
            cut_in_um=cut_in_um, 
            voxel_in_um=voxel_in_um,
            twin_ylabel='Dose [Gy]', twin_plotlabel='Dose',
            title_fontsize = 20, legend_fontsize=16,
            xlim=(xlim[0], primary_x_stop_mm), 
            xlabel_fontsize=18, 
            xticks_fontsize=15, ylabel_fontsize=18, 
            yticks_fontsize=15
        )
        

        # Extract fluence distribution columns into a separate DataFrame.
        # Fluence columns are identified by the suffix "_f" in their names.
        # The resulting DataFrame contains the depth 'x' and all fluence columns.
        fluence_columns = [col for col in dose_df.columns if "_f" in col]
        fluence_df = dose_df[['x'] + fluence_columns].copy()
        
        # Remove columns with a percentage of values equal to 0 higher than
        # threshold from fluence DataFrame.
        percentage_zero_fluence = eda.calculate_zero_values_percentage(fluence_df)
        fluence_df = eda.drop_zero_columns(
            fluence_df, threshold=zero_thr_percent,
            apply_rename=True,
            verbose=verbose
        )

        if verbose > 0:
            # Print statement to introduce the dose and fluence DataFrames
            print("\nDose DataFrame:")
            print(dose_df.head())
            print("\nFluence DataFrame:")
            print(fluence_df.head())
            print("\nPercentage of null values in fluence DataFrame:") 
            print(percentage_zero_fluence.sort_index())
            
        element_list = ['O18', 'O16', 'O15', 'N16', 'N15',  
                'N14', 'C13', 'C12', 'B11', 'Be9', 
                'Li6', 'alpha', 'triton', 'deuteron', 'proton', 
                'proton_1']
        
        # Load secondary data files if available
        if secondary_data_dirs:           
            if isinstance(secondary_data_dirs, str):
                secondary_data_dirs = [secondary_data_dirs]
            
            if isinstance(secondary_data_files, str):
                secondary_data_files = [secondary_data_files]
                
            if isinstance(secondary_dose_files, str):
                secondary_dose_files = [secondary_dose_files]
            
            # Check if secondary data and dose files lengths match, if provided
            if (secondary_data_files and secondary_dose_files) and \
                (len(secondary_data_files) != len(secondary_dose_files)):
                    raise ValueError("Length of secondary_data_files must match"
                                     " the length of secondary_dose_files.")
                
            # Ensure secondary_data_dirs matches the length of
            # secondary_data_files 
            if len(secondary_data_dirs) != len(secondary_data_files):
                secondary_data_dirs = [secondary_data_dirs] * len(secondary_data_files)
                    
            if secondary_data_files is not None:
                # Initialize list to hold secondary data files
                secondary_let_df = []
                
                for j, secondary_file in enumerate(secondary_data_files):
                    df_temp = eda.import_out_file(
                        os.path.join(secondary_data_dirs[j], secondary_file))
                    
                    _, voxel_in_um_temp = \
                        eda.get_cut_voxel_from_filename(secondary_file)
                    
                    df_temp, _, _, _ = mandatory_let_df_manipulation(
                        let_df=df_temp,
                        vsize_um=voxel_in_um_temp,
                        let_type=let_type,
                        primary_pattern=primary_pattern,
                    )
                    
                    # Append to the list of secondary data
                    secondary_let_df.append(df_temp)
                    
            if secondary_dose_files is not None:
                # Initialize list to hold secondary dose files
                secondary_dose_df = []
                
                for j, secondary_file in enumerate(secondary_dose_files):
                    dose_temp = eda.import_out_file(
                        os.path.join(secondary_data_dirs[j], secondary_file))
                    
                    _, voxel_in_um_temp = \
                        eda.get_cut_voxel_from_filename(secondary_file)
                        
                    dose_temp = eda.convert_index_feature_to_depth(
                        dose_temp, voxel_in_um_temp, 
                        input_size='um', 
                        output_size='mm'
                    )
                    
                    # Append to the list of secondary dose data
                    secondary_dose_df.append(dose_temp)
                    
            if more_df_figsize is None:
                more_df_figsize = (17, 10) # best for one plot in a slide
        
        else:
            secondary_let_df = None
            secondary_dose_df = None
        
        if more_feat_figsize is None:
            more_feat_figsize = (14, 12) # best for two plots in a slide
                    
        plot_and_export(
            main_df=df_non_zero,
            dose_profile=dose_profile,
            fluence_df=fluence_df,
            let_type=let_type,
            feature_on_single_plot='proton_1_T',
            element_list=element_list, 
            n_features_per_plot=n_elements_per_plot,
            primary_x_stop_mm=primary_x_stop_mm,
            cut_in_um=cut_in_um, 
            voxel_in_um=voxel_in_um,
            twin_ylabel='Dose [Gy]', 
            twin_plotlabel='Dose',
            title_fontsize=20, 
            legend_fontsize=16,
            xlim=xlim,
            xlabel_fontsize=18,
            xticks_fontsize=15, 
            ylabel_fontsize=18,
            yticks_fontsize=15, 
            save_dir=eda_plot_dir,
            export_plots=export_plots,
            more_feat_figsize=more_feat_figsize,
            more_df_figsize=more_df_figsize,
            subplot_specs=subplot_specs,
            secondary_let_df=secondary_let_df,
            secondary_dose_df=secondary_dose_df,
            secondary_plot_suffix=df_labels,
        )

    if run_outliers_search:
        # As we can see by comparing the LET distribution for the full particle
        # range and stopping after the primary peak, most of the noisy
        # distribution is removed, but some "unwanted" points still remain. The
        # presence of outliers can significantly reduce the performance and
        # accuracy of a predictable model.
        #
        # The recent adjustments have widened the range, a positive development
        # at first glance. However, it's become apparent that this expansion has
        # ushered in a surge of outliers, complicating our data analysis. Yet,
        # amidst this statistical noise lies a crucial insight: beyond the peak
        # around 34.8 mm to the pp stop, merely 2% of the entire dataset
        # persists.   
        #
        # This sliver of the distribution holds immense significance,
        # particularly for the training of generative models. To leverage its
        # potential effectively, addressing the outliers becomes imperative. By
        # replacing these anomalies, we can bolster the weight and significance
        # of this critical tail end, enhancing the fidelity and accuracy of our
        # model's generative capabilities. 
        #
        # Through visual examination of various plots employing diverse outlier
        # detection methods and replacement strategies, a standout combination
        # emerges: 'DBSCAN' for outlier identification and
        # 'nearest_neighbor_regressor' for replacement generation. This pairing
        # exhibits superior performance compared to other methods. 
    
        # Unpack outliers_params dictionary
        if outliers_params is not None:
            outliers_method_list = outliers_params.get('outliers_method_list', [])
            let_upper_limit = outliers_params.get('let_upper_limit', None)
            lof_neighbors = outliers_params.get('lof_neighbors', None)
            dbscan_eps = outliers_params.get('dbscan_eps', None)
            dbscan_min_samples = outliers_params.get('dbscan_min_samples', None)
            knn_neighbors = outliers_params.get('knn_neighbors', None)
            replace_method_list = outliers_params.get('replace_method_list', [])
        else:
            raise ValueError("Outliers parameters must be provided if "
                             "'run_outliers_search' is True.")
        
        outliers_replacements_df = outliers_search(
            let_df=df_early_stop,
            let_type=let_type,
            outliers_method_list=outliers_method_list,
            replace_method_list=replace_method_list,
            primary_x_stop_mm=primary_x_stop_mm,
            cut_in_um=cut_in_um,
            voxel_in_um=voxel_in_um,
            let_upper_limit=let_upper_limit,
            lof_neighbors=lof_neighbors,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            knn_neighbors=knn_neighbors,
            verbose=verbose
        )
        
        if not outliers_replacements_df.empty:
            # Create a new DataFrame with LET values identified as outliers replaced by
            # their corresponding 
            df_early_stop = data_eng.replace_outliers_in_df(
                df_early_stop, 
                outliers_replacements_df,
                'LTT'
            )
        
        if verbose > 0:
            replacement_column = 'LTT'+'_is_replacement'
            # Check if a replacement has been made
            if replacement_column not in df_early_stop.columns:
                print("\nNo value replaced.")
            elif df_early_stop[replacement_column].any():
                print("\nOutliers have been replaced.")
            else:
                print("\nNo value replaced.")

    if run_vae_training:
        # Check if we need to optimize the VAE using Optuna
        if optimize_vae:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: opt_objective(trial, df_early_stop, optuna_params,
                                            verbose=False),                   
                n_trials=n_trials
            )
            
            # Extract best parameters and log results
            best_params = study.best_params
            best_loss = study.best_value
            print("Best Hyperparameters:", best_params)
            print("Best Loss:", best_loss)
            
            # Re-train VAE with best parameters
            _ = train_vae(
                let_df=df_early_stop,
                verbose=verbose,
                vae_plot_dir=vae_plot_dir,
                vae_history_dir=vae_history_dir,
                **best_params
            )
        else:
            # Train VAE with default parameters
            _ = train_vae(
                let_df=df_early_stop,
                verbose=verbose,
                vae_plot_dir=vae_plot_dir,
                vae_history_dir=vae_history_dir,
                **vae_params
            )
    
    
if __name__ == '__main__':

    subplot_specs_60_mev = {
        'feat_dist_location': [0.2, 0.35, 0.25, 0.50],
        'feat_dist_x_range': [34.0, 36.0],
        'let_prof_location': [0.17, 0.45, 0.40, 0.40],
        'let_prof_x_range': [32.5, 40.0]
    }
    
    subplot_specs_120_mev = {
        'feat_dist_location': [0.5, 0.35, 0.35, 0.50],
        'feat_dist_x_range': [100.0, 120.0],
        'let_prof_location': [0.5, 0.45, 0.37, 0.40],
        'let_prof_x_range': [100.0, 120.0]
    }
    
    subplot_specs_500_mm = {
        'feat_dist_location': [0.5, 0.35, 0.35, 0.50],
        'feat_dist_x_range': [34.0, 36.0],
        'let_prof_location': [0.5, 0.45, 0.37, 0.40],
        'let_prof_x_range': [32.5, 40.0]
    }
    
    element_list = ['O18', 'O16', 'O15', 'N16', 'N15',  
                    'N14', 'C13', 'C12', 'B11', 'Be9', 
                    'Li6', 'alpha', 'triton', 'deuteron', 'proton', 
                    'proton_1']
    
    # Outlier search and replacement generation parameters
    outliers_method_list = ['upper_limit', 'lof', 'lof', 
                            'dbscan', 'dbscan']
 
    replace_method_list = ['median', 'local_mean', 'nearest_neighbor',
                           'mean', 'nearest_neighbor_regressor']
    
    outliers_params = {
        'outliers_method_list': outliers_method_list,
        'let_upper_limit': 21.0,
        'lof_neighbors': 20,
        'dbscan_eps': 0.15,
        'dbscan_min_samples': 8,
        'knn_neighbors': 5,
        'replace_method_list': replace_method_list
    }
    
    # Parameters for the VAE training
    vae_params = {
        'train_size': 0.7,
        'val_size': 0.2,
        'test_size': 0.1,
        'single_scaler': True,
        'batch_size': 100,
        'normalization': nn.BatchNorm1d,
        'activation': nn.ReLU,
        'output_activation': nn.ReLU,
        'learning_rate': 1e-4,
        'penalty_L2': 0, 
        'training_epochs': 10,
        'train_report_every': 1,
        # Any other parameters needed for VAE training
    }
    
    # Optuna parameters for VAE training
    optuna_params = {
                'learning_rate': (1e-5, 1e-2),
                'batch_size': [10, 50, 100, 200],
                'latent_dim_scaling': (1, 4),
                'net_size_upsample': (1, 3),
                'net_size_scaling': (1, 4),
                'epochs': [10, 25, 50], 
            }
    
    labels = ['N: 1e7, E = 122 MeV',
              'N: 1e8, E = 60 MeV']
    
    main(
        run_data_exploration=False, 
        run_outliers_search=False,
        run_vae_training=True,
        optimize_vae=False,
        n_trials=1,
        optuna_params = optuna_params,
        data_dir='../data/t96_1e8_c1000_v1_d50',
        data_file='Let_1000-1.out',
        dose_file='Dose_1000-1.out',
        subplot_specs=subplot_specs_60_mev,
        more_feat_figsize=(17, 10),
        n_elements_per_plot=1,
        # secondary_data_dirs='../data/t96_1e8_c1000_v1_d50',
        # secondary_data_files='Let_1000-1.out',
        # secondary_dose_files='Dose_1000-1.out',
        # df_labels=labels,
        # more_df_figsize=(17,10),
        let_type='track',
        element_list=element_list,
        eda_xlim=(0.0, None), # (None, None) is for automatically generated xlim
        outliers_params=outliers_params,
        vae_params=vae_params,
        random_seed=0,
        verbose=2,
        export_plots=True,
        eda_plot_dir='../eda_plots/',
        vae_plot_dir='../vae_render/',
        vae_history_dir='../model_history/',
    )
    
