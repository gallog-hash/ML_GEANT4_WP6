import logging
import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import core.preprocessing.preprocessing_utils as preprocess
from core.preprocessing.preprocessing_utils import (
    extract_cut_voxel,
    find_bragg_peak_start,
    import_and_process_dose_fluence,
)
from utils import (
    barplot_zero_values_percentage,
    combined_plot_comparison,
    create_correlation_plot_title,
    display_plot,
    ensure_directory_exists,
    get_correlation_df,
    plot_correlation_df,
    plot_feature_distribution,
    plot_let_profile,
    plot_more_dataframe,
    plot_more_features,
    save_figure,
    visualize_zero_values,
)

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Preserved subplot specifications
subplot_specs_60_mev = {
    'feat_dist_location': [0.2, 0.35, 0.25, 0.50],
    'let_prof_location': [0.17, 0.45, 0.40, 0.40]
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

# Preserved element list
default_element_list = [
    'O18', 'O16', 'O15', 'N16', 'N15', 'N14', 'C13', 'C12', 'B11', 'Be9',
    'Li6', 'alpha', 'triton', 'deuteron', 'proton', 'proton_1'
]

# Preserved labels
default_labels = [
    'N: 1e7, E = 122 MeV',
    'N: 1e8, E = 60 MeV'
]

def import_and_clean_data_eda(data_dir: str, data_file: str, 
                            primary_particle: str, let_type: str,
                            eda_xlim: List[Union[float, None]],
                            cut_with_primary: bool, 
                            drop_zero_cols: bool = False,
                            drop_zero_thr: float = 100.0,
                            verbose: int = 0
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                        pd.DataFrame, int, int, List[float], 
                                        float, pd.DataFrame, dict]:
    """
    Import LET data from data file, extract and validate cut and voxel values,
    perform mandatory cleaning, and determine key plotting parameters.
    
    Returns:
    - df (DataFrame): Original imported DataFrame.
    - df_non_zero (DataFrame): DataFrame after cleaning (removing zero-only
        columns).
    - df_processed (DataFrame): Final processed DataFrame (possibly truncated
        based on primary particle stopping point).
    - cut_in_um (int): Cut value in micrometers.
    - voxel_in_um (int): Voxel value in micrometers.
    - xlim (List[float]): X-axis limits for plotting.
    - primary_x_stop_mm (float): Depth at which primary particle stops.
    - percentage_eq_to_zero_df (DataFrame): DataFrame with percentage of zero
        values for each feature.
    - primary_zero_perc (dict): Dictionary of zero percentages for primary
        particle columns.
    """ 
    # Import the main LET data file.
    data_path = os.path.join(data_dir, data_file)
    try:
        df = preprocess.import_out_file(data_path)
    except OSError as e:
        logger.error(f"Data import failed: {e}")
        raise
    
    logger.info("DataFrame loaded from file: %s", data_path)
    if verbose > 0:
        logger.debug("First 10 rows of the DataFrame: \n%s", df.head(10))
        
    # Try to extract cut_in_um and voxel_in_um from the full file path.
    cut_in_um, voxel_in_um = preprocess.extract_cut_voxel(data_path, verbose=bool(verbose))
        
    # Use mandatory cleaning.
    df_non_zero, primary_zero_perc, percentage_eq_to_zero_df = \
        preprocess.mandatory_let_df_manipulation(
            let_df=df,
            vsize_um=voxel_in_um,
            let_type=let_type,
            primary_particle=primary_particle,
            drop_zero_thr=drop_zero_thr,
            drop_zero_cols=drop_zero_cols,
            verbose=verbose
        )
    primary_zero_perc = primary_zero_perc.to_dict()
    if verbose > 0:
        logger.info("Primary zero percentages: %s", primary_zero_perc)
        
    max_depth_mm = np.ceil(df_non_zero['x'].max())
    xlim = [0.0, max_depth_mm]
    if eda_xlim[0] is not None:
        xlim[0] = eda_xlim[0]
    if eda_xlim[1] is not None:
        xlim[1] = eda_xlim[1]
        
    if cut_with_primary:
        df_processed = preprocess.cut_dataframe_with_primary(
            df_non_zero, column_type=let_type,                             
            primary_particle=primary_particle
            )
        primary_x_stop_mm = df_processed['x'].iloc[-1]
        logger.info("Truncated DataFrame based on primary particle.")
    else:
        df_processed = df_non_zero.copy()
        primary_column = preprocess.find_primary_column_with_fewest_zeros(
            df_processed, primary_particle
        )
        primary_x_stop_idx = preprocess.find_min_after_peak_index(
            df_processed[primary_column]
        )
        primary_x_stop_mm = df_processed.iloc[primary_x_stop_idx]['x']
        logger.info("Unaltered copy of non-zero data.")
        
    if verbose > 0:
        logger.info("Primary particle stops at %.3f mm.", primary_x_stop_mm)
        
    return (df, df_non_zero, df_processed, int(cut_in_um), int(voxel_in_um), 
            xlim, primary_x_stop_mm, percentage_eq_to_zero_df, primary_zero_perc)
    

def import_secondary_data(secondary_data_dirs, secondary_data_files,
                        secondary_dose_files, let_type, primary_particle
                        ) -> Tuple[Union[List[pd.DataFrame], None], 
                                    Union[List[pd.DataFrame], None]]:
    """
    Import secondary LET and dose data files, perform mandatory cleaning, and
    convert dose data to depth features.
    
    Returns:
    - secondary_let_df (List[pd.DataFrame] or None): List of secondary LET
        dataframes. 
    - secondary_dose_df (List[pd.DataFrame] or None): List of secondary dose
        dataframes. 
    """
    if secondary_data_dirs:
        if isinstance(secondary_data_dirs, str):
            secondary_data_dirs = [secondary_data_dirs]
        if isinstance(secondary_data_files, str):
            secondary_data_files = [secondary_data_files]
        if isinstance(secondary_dose_files, str):
            secondary_dose_files = [secondary_dose_files]
            
        if (secondary_data_files and secondary_dose_files) and \
        (len(secondary_data_files) != len(secondary_dose_files)):
            raise ValueError("Length of secondary_data_files must match the "
                            "length of secondary_dose_files.")
        
        if len(secondary_data_dirs) != len(secondary_data_files):
            secondary_data_dirs = [secondary_data_dirs] * len(secondary_data_files)
            
        secondary_let_df = []
        for j, secondary_file in enumerate(secondary_data_files):
            current_dir = secondary_data_dirs[j]
            if isinstance(current_dir, list):
                current_dir = current_dir[0]
            df_temp = preprocess.import_out_file(os.path.join(current_dir, secondary_file))
            _, voxel_in_um_temp = extract_cut_voxel(os.path.join(current_dir, secondary_file))
            df_temp, _, _, _ = preprocess.mandatory_let_df_manipulation(
                let_df=df_temp,
                vsize_um=voxel_in_um_temp,
                let_type=let_type,
                primary_particle=primary_particle,
            )
            secondary_let_df.append(df_temp)
            
        secondary_dose_df = []
        for j, secondary_file in enumerate(secondary_dose_files):
            current_dir = secondary_data_dirs[j]
            if isinstance(current_dir, list):
                current_dir = current_dir[0]
            dose_temp = preprocess.import_out_file(os.path.join(current_dir, secondary_file))
            
            _, voxel_in_um_temp = preprocess.extract_cut_voxel(
                os.path.join(current_dir, secondary_file)
            )
            dose_temp = preprocess.convert_index_feature_to_depth(
                dose_temp, voxel_in_um_temp, input_size='um', output_size='mm'
            )
            secondary_dose_df.append(dose_temp)
    else:
        secondary_let_df = None
        secondary_dose_df = None
        
    return secondary_let_df, secondary_dose_df

def plot_eda_figures(df_non_zero: pd.DataFrame, dose_profile: pd.DataFrame,
                    eda_plot_dir: str, export_plots: bool,
                    cut_in_um: int, voxel_in_um: int, xlim: List[float],
                    let_type: str, primary_x_stop_mm: float,
                    percentage_eq_to_zero_df: pd.DataFrame,
                    subplot_specs: dict) -> None:
    """
    Generate EDA plots and save them if export_plots is True.
    """
    
    if subplot_specs is not None and subplot_specs.get('let_prof_x_range') is None:
        subplot_specs['let_prof_x_range'] = [
            df_non_zero.loc[
            find_bragg_peak_start(df_non_zero["LTT" if let_type == "track" 
            else "LDT"]), 'x'], df_non_zero['x'].iloc[-1]
        ]
        
    # Plot overall LET profile.
    cfig = plot_let_profile(df_non_zero, column_type=let_type,
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
    display_plot(cfig)
    
    if export_plots:
        filename = f"let_profile_c{int(cut_in_um)}_v{int(voxel_in_um)}_" \
                    f"d{int(xlim[-1])}"
        save_figure(cfig, eda_plot_dir, filename, formats=['png', 'eps'])
        
    # Barplot for percentage of zero values per feature.
    cfig = barplot_zero_values_percentage(
        percentage_eq_to_zero_df, 
        figsize=(15,6),
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um
    )
    display_plot(cfig)
    
    if export_plots:
        filename = f"barplot_zeros_c{int(cut_in_um)}_v{int(voxel_in_um)}_" \
                    f"d{int(xlim[-1])}"
        save_figure(cfig, eda_plot_dir, filename, formats=['png', 'eps'])
        
    # Correlation heatmap with LET total over the entire range.
    heatmap_title_x_full = create_correlation_plot_title(
        df_non_zero, 'Correlation Heatmap', cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um)
    let_total_corr_df = get_correlation_df(df_non_zero, let_type)
    cfig = plot_correlation_df(let_total_corr_df, figsize=(4,10),
                            plot_title=heatmap_title_x_full)
    display_plot(cfig)
    
    if export_plots:
        filename = f"correlation_heatmap_c{int(cut_in_um)}_" \
                    f"v{int(voxel_in_um)}_d{int(xlim[-1])}"
        save_figure(cfig, eda_plot_dir, filename, formats=['png', 'eps'])
        
    # Combined plot comparison for correlation coefficients and zero values.
    cfig = combined_plot_comparison(
        left_data=percentage_eq_to_zero_df, right_data=let_total_corr_df,
        left_ylabel='Percentage of zero values',
        right_ylabel=f'Correlation with {"LTT" if let_type=="track" else "LDT"}',
        xlabel='Non-zero features',
        cut_in_um=cut_in_um, voxel_in_um=voxel_in_um,
        left_ylim=(0.0, 100.0), right_ylim=(0.0, 1.0)
    )
    display_plot(cfig)
    
    if export_plots:
        filename = f"corr_vs_zeros_c{int(cut_in_um)}_v{int(voxel_in_um)}_" \
                    f"d{int(xlim[-1])}"
        save_figure(cfig, eda_plot_dir, filename, formats=['png', 'eps'])
        
    # Final LET profile with subplot.
    subplot_x_stop = np.ceil(primary_x_stop_mm * 10) / 10
    subplot_x_range = subplot_specs.get('let_prof_x_range', None)
    if subplot_x_range:
        subplot_x_range[1] = subplot_x_stop
        
    cfig = plot_let_profile(df_non_zero, column_type=let_type,
        dose_profile=dose_profile,
        subplot_location=[0.2, 0.35, 0.25, 0.50],
        subplot_x_range=subplot_x_range,
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        twin_ylabel='Dose [Gy]', 
        twin_plotlabel='Dose',
        title_fontsize = 20, 
        legend_fontsize=16,
        xlim=(xlim[0], primary_x_stop_mm), 
        xlabel_fontsize=18, 
        xticks_fontsize=15, 
        ylabel_fontsize=18, 
        yticks_fontsize=15
    )
    display_plot(cfig)

def plot_extened_comparison(
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
    ions_let_figsize: tuple[float]=(12,14),
    fluence_figsize: tuple[float]=(12,14),
    more_df_figsize: tuple[float]=(17,10),
    subplot_specs: Union[dict, None]=None,
    secondary_let_df: Union[List[pd.DataFrame], None]=None,
    secondary_dose_df: Union[List[pd.DataFrame], None]=None,
    secondary_plot_suffix: Union[List[str], None]=None,
    export_plots: bool=False,
    **data_exploration_specs
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
    - ions_let_figsize (tuple(float,float)), optional): Size of ions let plot.
        Default to (12, 14).
    - fluence_figsize (tuple(float,float)), optional): Size of fluence plot.
        Default to (12, 14).
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
    # Define let_label string
    let_label = 'LTT' if let_type == 'track' else 'LDT'
    
    if subplot_specs is not None and subplot_specs.get('feat_dist_x_range') is None:
        subplot_specs['feat_dist_x_range'] = [
            main_df.loc[
            find_bragg_peak_start(main_df[feature_on_single_plot], 0.0009), 'x'],
            np.ceil(primary_x_stop_mm * 10) / 10
        ]
    
    # Plot the distribution of the primary particle to show where it stops.
    cfig = plot_feature_distribution(
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
    display_plot(cfig)

    if export_plots:
        filename = f"{feature_on_single_plot}_distribution_c{int(cut_in_um)}"
        filename += f"_v{int(voxel_in_um)}_d{int(xlim[-1])}"
        save_figure(cfig, save_dir, filename, formats=['png', 'eps'])
    
    # Visualize zero values in the main DataFrame
    cfig = visualize_zero_values(
        main_df, 
        column_as_index='x', 
        cmap='rocket',
        invert_cmap=True, 
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        primary_x_stop=primary_x_stop_mm * 1e3 / voxel_in_um
    )
    display_plot(cfig)
    
    if export_plots:
        filename = f"zeros_heatmap_c{int(cut_in_um)}_v{int(voxel_in_um)}_"
        filename += f"d{int(xlim[-1])}"
        save_figure(cfig, save_dir, filename, formats=['png', 'eps'])
    
    # Plot the LET distribution of selected elements
    let_type_suffix = '_T' if let_type == 'track' else '_D'
    feature_list = [element + let_type_suffix for element in element_list]
    ylabel_single_element = r"Let single elements [keV $\mu$m$^{-1}$]"
    
    let_yscale = data_exploration_specs.get('let_yscale', 'linear')
    
    fig_list = plot_more_features(
        main_df, 
        dose_profile=dose_profile,
        feature_list=feature_list,
        ylabel=ylabel_single_element,
        marker_size=25, 
        marker_types=['.','v'],
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        figsize=ions_let_figsize,
        yscale=let_yscale,
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
    display_plot(fig_list)
    
    if export_plots:
        filename = f"single_elements_yscale_{let_yscale}_c{int(cut_in_um)}_"
        filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}"
        save_figure(fig_list, save_dir, filename, formats=['png', 'eps'])
    
    # Plot all selected elements with a logarithmic scale on the y-axis
    cfig = plot_more_features(
        main_df, 
        dose_profile=dose_profile,
        feature_list=feature_list,
        ylabel=ylabel_single_element,
        marker_size=9, 
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        figsize=ions_let_figsize,
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
    display_plot(cfig)
    
    if export_plots:
        filename = f"plot_more_features_yscale_log_c{int(cut_in_um)}_"
        filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}"
        save_figure(cfig, save_dir, filename, formats=['png', 'eps'])

    # Prepare feature list for fluence plot
    fluence_feat_list = [f.replace(let_type_suffix, '_f') for f in feature_list]
    
    # Plot fluence of the selected elements
    fig_list = plot_more_features(
        fluence_df, 
        dose_profile=dose_profile, 
        let_total_profile=main_df[['x', let_label]].copy(),
        feature_list=fluence_feat_list,
        ylabel="Fluence [counts]",
        marker_size=25, 
        marker_types=['.','v'],
        cut_in_um=cut_in_um, 
        voxel_in_um=voxel_in_um,
        figsize=fluence_figsize,
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
    display_plot(fig_list)

    if export_plots:
        filename = f"fluence_more_features_yscale_log_c{int(cut_in_um)}_"
        filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}"
        save_figure(fig_list, save_dir, filename, formats=['png', 'eps'])
        
    if secondary_let_df is not None:       
        fig_list = plot_more_dataframe(
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
        display_plot(fig_list)
        
        if export_plots:
            filename = f"more_df_features_c{int(cut_in_um)}_"
            filename += f"v{int(voxel_in_um)}_d{int(xlim[-1])}"
            save_figure(fig_list, save_dir, filename, formats=['png', 'eps'])
            
def main(
    cut_with_primary: bool=False,
    data_dir: str = '../data/t96_1e8_c1000_v100_d50',
    data_file: str = 'Let_1000-100.out',
    dose_file: str = 'Dose_1000-100.out',
    subplot_specs=None,
    let_type: str = 'track',
    eda_xlim=(None, None),
    verbose: int = 0,
    export_plots: bool = False,
    eda_plot_dir: str = None,
    df_labels=None,
    data_exploration_specs=None,
    secondary_data_dirs=None,
    secondary_data_files=None,
    secondary_dose_files=None,
    element_list=None,
    drop_zero_cols: bool = False,
    drop_zero_thr: float = 100.0,
):
    """
    Main function for exploratory data analysis: import data, clean it,
    and generate various plots.
    """
    # Resolve paths relative to project root for cross-directory execution
    from utils.filesystem_utils import resolve_path_with_project_root
    
    # Resolve data_dir path
    data_dir = resolve_path_with_project_root(data_dir)
    
    # Handle plot directory resolution
    if eda_plot_dir is not None:
        eda_plot_dir = resolve_path_with_project_root(eda_plot_dir)
        ensure_directory_exists(eda_plot_dir)
    if export_plots and eda_plot_dir is None:
        eda_plot_dir = os.getcwd()
    
    primary_particle = 'proton'
    
    # Use helper function to import and clean LET data.
    (df, df_non_zero, df_processed, cut_in_um, voxel_in_um, xlim,
    primary_x_stop_mm, percentage_eq_to_zero_df, primary_zero_perc) = \
        import_and_clean_data_eda(data_dir, data_file, primary_particle,
                                let_type, eda_xlim, cut_with_primary, 
                                drop_zero_thr=drop_zero_thr,
                                drop_zero_cols=drop_zero_cols,
                                verbose=verbose)
        
    # Process dose data.
    dose_profile, fluence_df = import_and_process_dose_fluence(
        data_dir, dose_file, voxel_in_um, zero_thr_percent=drop_zero_thr, 
        drop_zero_cols=drop_zero_cols, verbose=verbose
    )
    
    # Plot EDA figures.
    plot_eda_figures(df_non_zero, dose_profile, eda_plot_dir, export_plots,
                        cut_in_um, voxel_in_um, xlim, let_type, primary_x_stop_mm,
                        percentage_eq_to_zero_df, subplot_specs)
    
    # Load secondary data files with helper function.
    secondary_let_df, secondary_dose_df = import_secondary_data(
        secondary_data_dirs, secondary_data_files,
        secondary_dose_files, let_type, primary_particle
    )
    
    ions_let_figsize_input = \
        data_exploration_specs.get('ions_let_figsize', (12, 14))
    fluence_figsize_input = \
        data_exploration_specs.get('fluence_figsize', (17, 10))
    n_features_per_plot_input = \
        data_exploration_specs.get('n_elements_per_plot', 1)
    more_df_figsize_input = \
            data_exploration_specs.get('more_df_figsize', (12, 14))
                
    primary_feature = (f'{primary_particle}_1_T' if let_type == 'track' 
                       else f'{primary_particle}_1_D')  
    plot_extened_comparison(
        main_df=df_non_zero,
        dose_profile=dose_profile,
        fluence_df=fluence_df,
        let_type=let_type,
        feature_on_single_plot=primary_feature,
        element_list=element_list, 
        n_features_per_plot=n_features_per_plot_input,
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
        ions_let_figsize=ions_let_figsize_input,
        fluence_figsize=fluence_figsize_input,
        more_df_figsize=more_df_figsize_input,
        subplot_specs=subplot_specs,
        secondary_let_df=secondary_let_df,
        secondary_dose_df=secondary_dose_df,
        secondary_plot_suffix=df_labels,
    )
    
    # Return all key variables for direct inspection
    return {
        # Raw and processed LET data
        "df": df,                           # Original imported LET dataframe
        "df_non_zero": df_non_zero,         # LET dataframe with zero-only cols removed
        "df_processed": df_processed,       # LET dataframe after processing/truncation

        # Dose and fluence data
        "dose_profile": dose_profile,       # Dose profile DataFrame
        "fluence_df": fluence_df,     # Fluence dataframe from dose processing (if available)

        # Key metrics and thresholds
        "cut_in_um": cut_in_um,              # Cut depth in micrometers
        "voxel_in_um": voxel_in_um,          # Voxel size in micrometers
        "xlim": xlim,                        # X-axis limits for plotting
        "primary_x_stop_mm": primary_x_stop_mm,  # Primary particle stop depth in mm
        "percentage_eq_to_zero_df": percentage_eq_to_zero_df,  # % zero values
        "primary_zero_perc": primary_zero_perc,  # Dict of zero percentages for primary columns

        # Secondary datasets (if any)
        "secondary_let_df": secondary_let_df,   # Secondary LET dataframes list or None
        "secondary_dose_df": secondary_dose_df, # Secondary dose dataframes list or None

        # Parameters used
        "params": {
            "cut_with_primary": cut_with_primary,
            "data_dir": data_dir,
            "data_file": data_file,
            "dose_file": dose_file,
            "let_type": let_type,
            "eda_xlim": eda_xlim,
            "verbose": verbose,
        }
    }

if __name__ == '__main__':
    # Run main with your desired parameters
    # NOTE: Paths below are relative to project root. The script auto-detects
    # project root when executed from different directories (e.g., VSCode from src/)
    results = main(
        cut_with_primary=True,
        data_dir='data/thr96_1e8_v1um_cut1mm_ver_11-2-2',
        data_file='Let.out',
        dose_file='Dose.out',
        data_exploration_specs={
            'let_yscale': 'linear',
            'fluence_yscale': 'log',
            'primary_figsize': (12, 7),
            'ions_let_figsize': (12, 14),
            'fluence_figsize': (17, 10),
            'n_elements_per_plot': 1,
            'more_df_figsize': (12, 14),
        },
        subplot_specs={
            'feature_dist_location': [0.2, 0.35, 0.25, 0.50],
            'let_prof_location': [0.17, 0.45, 0.40, 0.40]
        },
        element_list=['O18', 'O16', 'O15', 'N16', 'N15'],
        eda_xlim=(0.0, None),
        drop_zero_cols=False,
        drop_zero_thr=100.0,
        export_plots=True,
        eda_plot_dir='eda_plots/let_dose',
        verbose=1
    )

    # Keep variables in scope       
    print("\nVariables from main() are in the `results` dictionary.")
    try:
        from IPython import embed
        embed()
    except ImportError:
        import code
        code.interact(local=locals())
