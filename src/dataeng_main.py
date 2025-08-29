import logging
import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from core.preprocessing import preprocessing_utils
from core.preprocessing.preprocessing_utils import (
    find_bragg_peak_start,
    import_and_process_dose_fluence,
)
from utils import display_plot, plot_let_profile
from utils.filesystem_utils import ensure_directory_exists, is_interactive_environment
from utils.outlier_detection_utils import identify_let_outliers, replace_outliers_in_df
from utils.outlier_visualization_utils import plot_feature_and_mark_outliers_by_let

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def compute_replacement_quality(let_series: pd.Series,
                                outliers_replacements_df: pd.DataFrame,
                                n_neighbors: int = 5) -> float:
    """
    Compute a replacement quality metric for a candidate method. The metric is
    defined as the mean absolute difference between each replacement value and
    the local baseline computed from the k nearest non-outliers values. Lower is
    better. 
    
    Parameters:
      - let_series (pd.Series): The filtered LET values.
      - outliers_replacements_df (pd.DataFrame): DataFrame with index corresponding
            to outlier positions and columns 'outliers' and 'replacements'.
      - n_neighbors (int): Number of neighbors for the local baseline.
      
    Returns:
      - float: The average absolute error between the replacement values and the
            local baseline. Returns infinity if no neighbors can be found.
    """
    quality_errors = []
    
    # Create a mask for non-outliers.
    non_outliers_mask = ~let_series.index.isin(outliers_replacements_df.index)
    non_outliers_series = let_series[non_outliers_mask]
    
    # Convert indices to numeric values; assume they can be inrpreted as floats.
    try:
        non_outliers_positions = non_outliers_series.index.astype(float).values.reshape(-1, 1)
    except Exception as e:
        logger.error(f"Error converting non-outliers indices: {e}")
        return np.inf
    
    if len(non_outliers_positions) < n_neighbors:
        logger.error("Not enough non-outliers to compute the local baseline. "
                     "Required: %d, Found: %d", n_neighbors, len(non_outliers_positions))
        return np.inf
    
    # Fit a nearest neighbors model to the non-outliers.
    try:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(non_outliers_positions)
    except Exception as e:
        logger.error("Error fitting NearestNeighbors model: %s", e)
        return np.inf
    
    for outlier_idx, row in outliers_replacements_df.iterrows():
        try:
            pos = float(outlier_idx)
        except Exception as e:
            logger.error("Error converting outlier index '%s': %s", outlier_idx, e)
            continue # Skip this outlier.
        
        # Query the nearest neighbors for the outlier position.
        try:
            _, indices = nbrs.kneighbors([[pos]])
        except Exception as e:
            logger.error("Error querying nearest neighbors for position %s: %s", pos, e)
            continue # Skip this outlier.
        
        # Get the neighbor LET values.
        try:
            neighbor_values = non_outliers_series.iloc[indices[0]].values
        except Exception as e:
            logger.error("Error retrieving neighbor values for index %s: %s", outlier_idx, e)
            continue # Skip this outlier.
        
        local_baseline = np.mean(neighbor_values)
        replacement_value = row['replacements']
        quality_errors.append(np.abs(replacement_value - local_baseline))
        
    if quality_errors:
        quality = np.mean(quality_errors)
        logger.debug("Computed replacement quality: %.4f", quality)
        return quality
    else:
        logger.warning("No quality errors computed; returning infinity.")
        return np.inf
        
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
    figs_auto_save: bool = False,
    figs_save_dir: str = None,
) -> pd.DataFrame:
    """
    Run the outlier search over all specified methods and automatically select
    the candidate with the best replacement quality metric.

    Parameters:
      - let_df (DataFrame): The LET data.
      - let_type (str): Type of LET ('track' or 'dose').
      - outliers_method_list (list of str): Methods for identifying outliers.
      - replace_method_list (list of str): Methods for replacing outliers.
      - primary_x_stop_mm (float): Depth (in mm) where the primary particle stops.
      - cut_in_um (int): Cut (in µm).
      - voxel_in_um (int): Voxel size (in µm).
      - let_upper_limit (float): Upper threshold for LET values.
      - lof_neighbors (int): Number of neighbors for LOF calculation.
      - dbscan_eps (float): Maximum distance between samples for DBSCAN.
      - dbscan_min_samples (int): Minimum number of samples for DBSCAN.
      - knn_neighbors (int): Number of neighbors for the KNN regressor.
      - verbose (int): Verbosity level.
      - figs_auto_save (bool): Whether to automatically save figures.
      - figs_save_dir (str): Directory to save figures.

    Returns:
      - pd.DataFrame: The outlier-replacement DataFrame for the selected method.
        If no outliers are found, returns an empty DataFrame with columns
        ['outliers', 'replacements'].
    """
    # Ensure the method lists have equal lengths.
    if len(outliers_method_list) != len(replace_method_list):
        raise ValueError("The lengths of outliers_method_list and "
                         "replace_method_list must be equal.")
    
    # Ensure output directory exists if auto-saving is enabled
    if figs_auto_save and figs_save_dir is not None:
        ensure_directory_exists(figs_save_dir)
        logger.info(f"Ensured directory exists: {figs_save_dir}")
        
    verbose_bool = verbose > 0
    let_prefix = 'LTT' if let_type.lower() == 'track' else 'LDT'
    results = {}
    quality_metrics = {}

    for outliers_method, replace_method in zip(outliers_method_list, 
                                               replace_method_list):
        cbar_label = (
            "LOF Outliers Factor" if outliers_method == "lof"
            else "Outliers Normalized Z-Score"
        )
        key = f"{outliers_method}_{replace_method}"
        
        candidate = identify_let_outliers(
            df=let_df, 
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
            verbose=verbose_bool,
            auto_save=figs_auto_save,
            save_dir=figs_save_dir,
        )
        if not candidate.empty:
            results[key] = candidate
            # Compute the quality metric for the candidate.
            # Here, we use the original LET series used for detection.
            let_columns = let_df.filter(like=let_prefix).columns
            if not let_columns.empty:
                let_series = let_df[let_columns[0]]
                quality_metrics[key] = compute_replacement_quality(
                    let_series, candidate, n_neighbors=knn_neighbors
                )
                if verbose_bool:
                    print(f"Quality metric for {key}: {quality_metrics[key]}")
    
    # Edge-case: if no candidate produced a valid result.
    if not quality_metrics:
        print("\nNo candidate methods produced valid outlier-replacement results.")
        return pd.DataFrame(columns=['outliers', 'replacements'])
      
    # Automatically select the candidate with the minimum quality metric.
    selected_key = min(quality_metrics, key=quality_metrics.get)
    print(f"\nSelected method: {selected_key}")
    print(f"Quality metric: {quality_metrics[selected_key]:.4f}")
    
    # Plot feature distributions with marked outliers for selected elements.
    element_list = ['proton_1_T', 'proton_T', 'alpha_T', 'deuteron_T', 'O16_T']
    for element in element_list:
        plot_feature_and_mark_outliers_by_let(
            df=let_df,
            outliers_and_replacements_df=results[selected_key],
            feature=element,
            subplot_location=[0.2, 0.4, 0.45, 0.45],
            subplot_x_range=(32.5, 40.0),
            save_fig=figs_auto_save,
            save_dir=figs_save_dir
        )

    return results[selected_key]

def process_outliers(df_processed: pd.DataFrame, let_type: str, 
                    primary_x_stop_mm: float, outliers_params: dict, 
                    cut_in_um: int, voxel_in_um: int, verbose: int,
                    figs_auto_save: bool, figs_save_dir: str) -> pd.DataFrame:
    """
    Run outlier search and replacement on the processed DataFrame.
    
    Returns the candidate outlier-replacement DataFrame for the selected method.
    """
    if outliers_params is None:
        raise ValueError("Outliers parameters must be provided.")
    
    # Ensure output directory exists if auto-saving is enabled
    if figs_auto_save and figs_save_dir is not None:
        ensure_directory_exists(figs_save_dir)
        logger.info(f"Ensured directory exists: {figs_save_dir}")
    
    outliers_method_list = outliers_params.get("outliers_method_list", [])
    let_upper_limit = outliers_params.get("let_upper_limit", None)
    lof_neighbors = outliers_params.get("lof_neighbors", None)
    dbscan_eps = outliers_params.get("dbscan_eps", None)
    dbscan_min_samples = outliers_params.get("dbscan_min_samples", None)
    knn_neighbors = outliers_params.get("knn_neighbors", None)
    replace_method_list = outliers_params.get("replace_method_list", [])
    
    results = {}
    quality_metrics = {}
    let_prefix = 'LTT' if let_type.lower() == 'track' else 'LDT'
    
    # Compute the first element of x_bounds using find_bragg_peak_start.
    let_column = df_processed.filter(like=let_prefix).columns[0]
    x_bounds_start = df_processed.loc[
        preprocessing_utils.find_bragg_peak_start(df_processed[let_column]), 'x']
    logger.info("Start of Bragg peak found at x = %.3f mm", x_bounds_start)
    
    for outliers_method, replace_method in zip(outliers_method_list, replace_method_list):
        cbar_label = (
            "LOF Outliers Factor" if outliers_method.lower() == "lof"
            else "Outliers Normalized Z-Score"
        )
        key = f"{outliers_method}_{replace_method}"
        
        candidate = identify_let_outliers(
            df=df_processed,
            column_type=let_type,
            outliers_method=outliers_method,
            let_upper_limit=let_upper_limit,
            lof_neighbors=lof_neighbors,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            knn_neighbors=knn_neighbors,
            replace_method=replace_method,
            x_bounds=(x_bounds_start, primary_x_stop_mm),
            cbar_label=cbar_label,
            cut_in_um=cut_in_um,
            voxel_in_um=voxel_in_um,
            verbose=verbose > 0,
            auto_save=figs_auto_save,
            save_dir=figs_save_dir,
        )
        if not candidate.empty:
            results[key] = candidate
            # Use the original LET series from df_processed.
            let_columns = df_processed.filter(like=let_prefix).columns
            if not let_columns.empty:
                let_series = df_processed[let_columns[0]]
                quality = compute_replacement_quality(
                    let_series, candidate, n_neighbors=knn_neighbors
                )
                quality_metrics[key] = quality
                if verbose > 0:
                    logger.debug("Quality metric for %s: %.4f", key, quality)
    
    if not quality_metrics:
        logger.warning("No candidate methods produced valid outlier-replacement results.")
        return pd.DataFrame(columns=['outliers', 'replacements'])
    
    selected_key = min(quality_metrics, key=quality_metrics.get)
    logger.info("Selected method: %s", selected_key)
    logger.info("Quality metric: %.4f", quality_metrics[selected_key])
    
    # Optionally plot features for selected candidate.
    element_list = ['proton_1_T', 'proton_T', 'alpha_T', 'deuteron_T', 'O16_T']
    for element in element_list:
        plot_feature_and_mark_outliers_by_let(
            df=df_processed,
            outliers_and_replacements_df=results[selected_key],
            feature=element,
            subplot_location=[0.2, 0.4, 0.45, 0.45],
            subplot_x_range=(x_bounds_start, 40.0),
            save_fig=figs_auto_save,
            save_dir=figs_save_dir
        )
    
    return results[selected_key]

def import_and_clean_data(data_dir: str, data_file: str, primary_particle: str, 
                        let_type: str, eda_xlim: List[Union[float, None]], 
                        cut_with_primary: bool, drop_zero_cols: bool = False, 
                        drop_zero_thr: float = 100.0, verbose: int = 0
                        ) -> Tuple[pd.DataFrame, float]:
    """
    Import the LET data and perform mandatory cleaning.
    
    Returns:
    - df_processed: The cleaned DataFrame ready for outlier processing.
    - primary_x_stop_mm: The x-value where the primary particle stops.
    """
    data_path = os.path.join(data_dir, data_file)
    try:
        df = preprocessing_utils.import_out_file(data_path)
    except OSError as e:
        logger.error("Data import failed: %s", e)
        raise

    logger.info("DataFrame loaded from file: %s", data_path)
    if verbose > 0:
        logger.debug("First 10 rows:\n%s", df.head(10))
    
    # Try to extract cut_in_um and voxel_in_um from the full file path.
    cut_in_um, voxel_in_um = preprocessing_utils.extract_cut_voxel(data_path, verbose=verbose)
    
    # Mandatory cleaning.
    df_non_zero, primary_zero_perc, _ = preprocessing_utils.mandatory_let_df_manipulation(
        let_df=df,
        vsize_um=voxel_in_um,
        let_type=let_type,
        primary_particle=primary_particle,
        drop_zero_thr=drop_zero_thr,
        drop_zero_cols=drop_zero_cols,
        verbose=verbose
    )
    
    # Determine plotting x-limits.
    max_depth_mm = np.ceil(df_non_zero['x'].max())
    xlim = [0.0, max_depth_mm]
    if eda_xlim[0] is not None:
        xlim[0] = eda_xlim[0]
    if eda_xlim[1] is not None:
        xlim[1] = eda_xlim[1]
    
    # Determine region based on primary particle stopping point.
    if cut_with_primary:
        df_processed = preprocessing_utils.cut_dataframe_with_primary(
            df_non_zero, column_type=let_type, primary_particle=primary_particle
        )
        primary_x_stop_mm = df_processed["x"].iloc[-1]
        logger.info("Data truncated based on primary particle. "
                    "Primary particle stops at %.3f mm.", primary_x_stop_mm)
    else:
        df_processed = df_non_zero.copy()
        primary_column = preprocessing_utils.find_primary_column_with_fewest_zeros(
            df_non_zero, primary_particle)
        primary_x_stop_idx = preprocessing_utils.find_min_after_peak_index(
            df_non_zero[primary_column]
        )
        primary_x_stop_mm = df_non_zero.iloc[primary_x_stop_idx]["x"]
        logger.info("Using unaltered non-zero data. "
                    "Primary particle stops at %.3f mm.", primary_x_stop_mm)
    
    if verbose > 0:
        logger.debug("Primary particle stops at %.3f mm.", primary_x_stop_mm)
    
    return df_processed, primary_x_stop_mm


def save_processed_data(df_processed: pd.DataFrame, data_dir: str, data_file: str):
    """
    Save the processed DataFrame to an output file in the same directory
    as the input file, using the *.out format.
    """
    output_filename = data_file.replace(".out", "_replaced.out")
    output_path = os.path.join(data_dir, output_filename)
    # Assuming the original *.out file uses tab delimiters.
    df_processed.to_csv(output_path, sep="\t", index=False)
    logger.info("Processed data saved to: %s", output_path)
    return output_path
            
def main( 
    cut_with_primary: bool = True,
    data_dir: str = '../data/thr96_1e8_v1um_cut1mm_ver_11-2-2',
    data_file: str = 'Let.out',
    dose_file: str = 'Dose.out',
    let_type: str = 'track',
    eda_xlim: List[Union[float, None]] = (0.0, None),
    outliers_params: dict = None,
    drop_zero_cols: bool = False,
    drop_zero_thr: float = 100.0,
    verbose: int = 2,
    figs_auto_save: bool = True,
    figs_save_dir: str = None,
):
    """
    Main function for data engineering tasks: import data, clean it, process
    outliers, and save the processed data in the same *.out file format.
    """ 
    
    # Default outliers configuration
    if outliers_params is None:
        outliers_method_list = ['upper_limit', 'lof', 'lof', 'dbscan', 'dbscan']
        replace_method_list = [
            "median",
            "local_mean",
            "nearest_neighbor",
            "mean",
            "nearest_neighbor_regressor",
        ]
        
        outliers_params = {
            'outliers_method_list': outliers_method_list,
            'let_upper_limit': 21.0,
            'lof_neighbors': 20,
            'dbscan_eps': 0.15,
            'dbscan_min_samples': 8,
            'knn_neighbors': 5,
            'replace_method_list': replace_method_list
        }
    
    # Resolve paths relative to project root for cross-directory execution
    from utils.filesystem_utils import resolve_path_with_project_root
    
    # Resolve data_dir path
    data_dir = resolve_path_with_project_root(data_dir)
    
    # Set up figures save directory if not provided
    if figs_save_dir is None:
        data_dir_name = os.path.basename(data_dir)
        figs_save_dir = os.path.join('dataeng_plots', data_dir_name)
    
    # Resolve figures save directory path
    figs_save_dir = resolve_path_with_project_root(figs_save_dir)
    
    # Ensure the figures save directory exists
    if figs_auto_save:
        ensure_directory_exists(figs_save_dir)
        logger.info(f"Ensured figures directory exists: {figs_save_dir}")
    
    primary_particle = 'proton'
    
    # Import and clean the data.  
    df_processed, primary_x_stop_mm = import_and_clean_data(
        data_dir, data_file, primary_particle, let_type, eda_xlim, 
        cut_with_primary, drop_zero_cols=drop_zero_cols, 
        drop_zero_thr=drop_zero_thr, verbose=verbose
    )
    
    # Cache cut_in_um and voxel_in_um to avoid recalculation.
    data_path = os.path.join(data_dir, data_file)
    cut_in_um, voxel_in_um = preprocessing_utils.extract_cut_voxel(data_path)
    
    # Import dose profile for plotting
    try:
        dose_profile, _ = import_and_process_dose_fluence(
            data_dir, dose_file, int(voxel_in_um), 
            zero_thr_percent=100, drop_zero_cols=False, verbose=verbose
        )
        logger.info("Dose profile imported successfully from: %s", dose_file)
    except Exception as e:
        logger.warning("Could not import dose profile: %s", e)
        dose_profile = pd.DataFrame()
    
    # Determine plotting x-limits
    max_depth_mm = np.ceil(df_processed['x'].max())
    xlim = [0.0, max_depth_mm]
    if eda_xlim[0] is not None:
        xlim[0] = eda_xlim[0]
    if eda_xlim[1] is not None:
        xlim[1] = eda_xlim[1]
    
    # Process outliers only if outliers_params is provided.
    if outliers_params is not None:
        candidate_df = process_outliers(
            df_processed, let_type, primary_x_stop_mm,
            outliers_params, 
            cut_in_um=int(cut_in_um),
            voxel_in_um=int(voxel_in_um),
            verbose=verbose,
            figs_auto_save=figs_auto_save,
            figs_save_dir=figs_save_dir,
        )
        if not candidate_df.empty:
            df_processed = replace_outliers_in_df(
                df_processed, candidate_df,
                'LTT' if let_type == 'track' else 'LDT'
            )
            logger.info("Outliers replaced successfully.")
            
            # Determine subplot x-range using dose profile maximum and primary stop
            if not dose_profile.empty:
                dose_max_idx = dose_profile['Dose(Gy)'].idxmax()
                subplot_start_x = dose_profile.loc[dose_max_idx, 'x']
                logger.info("Using dose profile maximum at x = %.3f mm for subplot start", subplot_start_x)
            else:
                # Fallback to Bragg peak start if dose profile is not available
                let_column = 'LTT' if let_type == 'track' else 'LDT'
                bragg_peak_start_idx = find_bragg_peak_start(df_processed[let_column])
                subplot_start_x = df_processed.loc[bragg_peak_start_idx, 'x']
                logger.info("Dose profile not available, using Bragg peak start at x = %.3f mm", subplot_start_x)
            
            subplot_stop_x = np.ceil(primary_x_stop_mm * 10) / 10
            
            logger.info("Subplot x-range: %.3f mm to %.3f mm", subplot_start_x, subplot_stop_x)
            
            # Plot LET profile after outlier replacement with dose profile overlay
            logger.info("Plotting LET profile after outlier replacement...")
            cfig = plot_let_profile(
                df_processed, 
                column_type=let_type,
                dose_profile=dose_profile if not dose_profile.empty else None,
                subplot_location=[0.2, 0.35, 0.25, 0.50],
                subplot_x_range=[subplot_start_x, subplot_stop_x],
                cut_in_um=int(cut_in_um), 
                voxel_in_um=int(voxel_in_um),
                twin_ylabel='Dose [Gy]' if not dose_profile.empty else None, 
                twin_plotlabel='Dose' if not dose_profile.empty else None,
                title_fontsize=20, 
                legend_fontsize=16,
                xlim=xlim, 
                xlabel_fontsize=18, 
                xticks_fontsize=15, 
                ylabel_fontsize=18, 
                yticks_fontsize=15
            )
            display_plot(cfig)
            
            if figs_auto_save and figs_save_dir:
                from utils import save_figure
                filename = f"let_profile_after_outlier_replacement_c{int(cut_in_um)}_v{int(voxel_in_um)}_d{int(xlim[-1])}"
                save_figure(cfig, figs_save_dir, filename, formats=['png', 'eps'])
                logger.info("LET profile plot saved to: %s", figs_save_dir)
                
    output_path = save_processed_data(df_processed, data_dir, data_file)
    
    # Return results for interactive access
    results = {
        'df_processed': df_processed,
        'primary_x_stop_mm': primary_x_stop_mm,
        'dose_profile': dose_profile,
        'cut_in_um': cut_in_um,
        'voxel_in_um': voxel_in_um,
        'xlim': xlim,
        'output_path': output_path,
        'data_path': data_path,
        'figs_save_dir': figs_save_dir
    }
    
    if outliers_params is not None and 'candidate_df' in locals():
        results['candidate_df'] = candidate_df
    
    return results
        

if __name__ == '__main__':
    # Run main and store results for interactive access
    # NOTE: Paths below are relative to project root. The script auto-detects
    # project root when executed from different directories (e.g., VSCode from src/)
    results = main(
        cut_with_primary=True,
        data_dir='data/thr96_1e8_v1um_cut1mm_ver_11-2-2',
        data_file='Let.out',
        dose_file='Dose.out',
        let_type='track',
        eda_xlim=[0.0, None],
        outliers_params=None,
        drop_zero_cols=False,
        drop_zero_thr=100.0,
        verbose=2,
        figs_auto_save=True,
        figs_save_dir=None
    )
    
    # If running interactively, make results available as global variables
    if is_interactive_environment():
        # Make key results available as global variables
        df_processed = results['df_processed']
        primary_x_stop_mm = results['primary_x_stop_mm']
        dose_profile = results['dose_profile']
        cut_in_um = results['cut_in_um']
        voxel_in_um = results['voxel_in_um']
        xlim = results['xlim']
        output_path = results['output_path']
        data_path = results['data_path']
        figs_save_dir = results['figs_save_dir']
        
        if 'candidate_df' in results:
            candidate_df = results['candidate_df']
        
        print("\n" + "="*60)
        print("INTERACTIVE MODE: Results available as variables:")
        print("="*60)
        print(f"• df_processed: DataFrame with {len(df_processed)} rows")
        print(f"• primary_x_stop_mm: {primary_x_stop_mm:.3f}")
        print(f"• dose_profile: DataFrame with {len(dose_profile)} rows")
        print(f"• cut_in_um: {cut_in_um}")
        print(f"• voxel_in_um: {voxel_in_um}")
        print(f"• xlim: {xlim}")
        print(f"• output_path: {output_path}")
        print(f"• data_path: {data_path}")
        print(f"• figs_save_dir: {figs_save_dir}")
        if 'candidate_df' in results:
            print(f"• candidate_df: DataFrame with {len(candidate_df)} outliers")
        print("• results: Complete results dictionary")
        print("="*60)
