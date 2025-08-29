"""
Outlier Visualization Utilities

This module provides visualization functions for outlier detection and analysis,
extracted from the my_plot_pkg package.
"""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .filesystem_utils import save_figure

logger = logging.getLogger(__name__)


def plot_feature_and_mark_outliers_by_let(
    df: pd.DataFrame,
    outliers_and_replacements_df: pd.DataFrame,
    feature: str,
    subplot_location: List[float] = [0.2, 0.4, 0.45, 0.45],
    subplot_x_range: Tuple[float, float] = (32.5, 40.0),
    figsize: Tuple[float, float] = (12, 8),
    main_alpha: float = 0.7,
    outlier_marker_size: float = 50,
    outlier_color: str = 'red',
    replacement_color: str = 'green',
    save_fig: bool = False,
    save_dir: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Plot a feature distribution and mark outliers identified by LET analysis.
    
    This function creates a main plot showing the feature values along the x-axis
    and adds a subplot highlighting the outlier regions with marked outliers
    and their replacements.
    
    Parameters:
        df: DataFrame containing the feature data
        outliers_and_replacements_df: DataFrame with outlier indices and 
                                    replacement values
        feature: Name of the feature column to plot
        subplot_location: [left, bottom, width, height] for subplot positioning
        subplot_x_range: Tuple of (x_min, x_max) for subplot x-axis range
        figsize: Figure size as (width, height)
        main_alpha: Transparency for main plot line
        outlier_marker_size: Size of outlier markers
        outlier_color: Color for outlier markers
        replacement_color: Color for replacement markers
        save_fig: Whether to save the figure
        save_dir: Directory to save the figure
        dpi: Resolution for saved figure
    """
    if feature not in df.columns:
        logger.error(f"Feature '{feature}' not found in DataFrame columns")
        return
    
    if 'x' not in df.columns:
        logger.error("'x' column not found in DataFrame")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Main plot
    ax.plot(df['x'], df[feature], alpha=main_alpha, linewidth=1.5, 
            label=f'{feature} profile')
    
    # Mark outliers if any exist
    if not outliers_and_replacements_df.empty:
        outlier_indices = outliers_and_replacements_df.index
        outlier_x = df.loc[outlier_indices, 'x']
        outlier_y = df.loc[outlier_indices, feature]
        replacement_y = outliers_and_replacements_df['replacements']
        
        # Plot outliers
        ax.scatter(outlier_x, outlier_y, 
                c=outlier_color, s=outlier_marker_size, 
                marker='x', label='Outliers', zorder=5)
        
        # Plot replacements
        ax.scatter(outlier_x, replacement_y, 
                c=replacement_color, s=outlier_marker_size, 
                marker='o', alpha=0.7, label='Replacements', zorder=5)
        
        # Connect outliers to replacements with lines
        for i, idx in enumerate(outlier_indices):
            x_pos = df.loc[idx, 'x']
            y_orig = df.loc[idx, feature]
            y_repl = replacement_y.iloc[i]
            ax.plot([x_pos, x_pos], [y_orig, y_repl], 
                    color='gray', linestyle='--', alpha=0.5, zorder=4)
    
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel(f'{feature} LET [keV μm⁻¹]')
    ax.set_title(f'{feature} Distribution with Outlier Marking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add subplot if outliers exist and subplot range is specified
    if (not outliers_and_replacements_df.empty and 
        subplot_x_range is not None and 
        subplot_location is not None):
        
        _add_outlier_subplot(
            fig, ax, df, outliers_and_replacements_df, feature,
            subplot_location, subplot_x_range, 
            outlier_color, replacement_color, outlier_marker_size
        )
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig and save_dir is not None:
        filename = f"outlier_analysis_{feature}"
        save_figure(fig, save_dir, filename, formats=['png', 'eps'],
                    close_after_save=False)


def _add_outlier_subplot(
    fig: plt.Figure,
    main_ax: plt.Axes,
    df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    feature: str,
    subplot_location: List[float],
    x_range: Tuple[float, float],
    outlier_color: str,
    replacement_color: str,
    marker_size: float
) -> None:
    """Add a subplot focusing on the outlier region."""
    # Create subplot
    subplot_ax = fig.add_axes(subplot_location)
    
    # Filter data to subplot range
    x_mask = (df['x'] >= x_range[0]) & (df['x'] <= x_range[1])
    subset_df = df[x_mask]
    
    if subset_df.empty:
        logger.warning(f"No data points in range {x_range} for subplot")
        return
    
    # Plot subset data
    subplot_ax.plot(subset_df['x'], subset_df[feature], 
                    linewidth=2, alpha=0.8, color='blue')
    
    # Mark outliers in subplot range
    subplot_outliers = outliers_df[outliers_df.index.isin(subset_df.index)]
    
    if not subplot_outliers.empty:
        outlier_indices = subplot_outliers.index
        outlier_x = df.loc[outlier_indices, 'x']
        outlier_y = df.loc[outlier_indices, feature]
        replacement_y = subplot_outliers['replacements']
        
        # Plot outliers and replacements
        subplot_ax.scatter(outlier_x, outlier_y, 
                            c=outlier_color, s=marker_size, 
                            marker='x', zorder=5)
        subplot_ax.scatter(outlier_x, replacement_y, 
                            c=replacement_color, s=marker_size, 
                            marker='o', alpha=0.7, zorder=5)
        
        # Connect with lines
        for i, idx in enumerate(outlier_indices):
            x_pos = df.loc[idx, 'x']
            y_orig = df.loc[idx, feature]
            y_repl = replacement_y.iloc[i]
            subplot_ax.plot([x_pos, x_pos], [y_orig, y_repl], 
                            color='gray', linestyle='--', alpha=0.7)
    
    subplot_ax.set_xlim(x_range[0], main_ax.get_xlim()[1])
    subplot_ax.set_xlabel('Depth (mm)', fontsize=10)
    subplot_ax.set_ylabel(f'{feature}', fontsize=10)
    subplot_ax.set_title('Outlier Detail View', fontsize=10)
    subplot_ax.grid(True, alpha=0.3)


def _add_connection_lines(
    main_ax: plt.Axes, 
    subplot_ax: plt.Axes, 
    x_range: Tuple[float, float]
) -> None:
    """Add connection lines between main plot and subplot."""
    from matplotlib.patches import ConnectionPatch
    
    # Create connection lines at x_range boundaries
    for x_pos in x_range:
        # Get y-limits of main plot for connection points
        main_ylim = main_ax.get_ylim()
        main_y = main_ylim[0] + (main_ylim[1] - main_ylim[0]) * 0.1
        
        # Connect from main plot data coordinates to subplot data coordinates
        con = ConnectionPatch(
            xyA=(x_pos, main_y), xyB=(x_pos, subplot_ax.get_ylim()[1]),
            coordsA='data', coordsB='data',
            axesA=main_ax, axesB=subplot_ax,
            color='gray', alpha=0.5, linestyle=':', linewidth=1
        )
        main_ax.add_patch(con)



def plot_let_outliers_replacement(
    df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    let_column: str,
    outliers_method: str,
    replace_method: str,
    x_bounds: Tuple[float, float] = (32.5, 40.0),
    cbar_label: str = "LOF Outliers Factor",
    cut_in_um: int = 1000,
    voxel_in_um: int = 100,
    figsize: Tuple[float, float] = (12, 8),
    filename: Optional[str] = None,
    save_fig: bool = False,
    save_dir: Optional[str] = None
) -> None:
    """
    Plot LET outliers and their replacements with detailed analysis.
    
    This function creates a comprehensive visualization showing the original
    LET profile, detected outliers, and their replacement values.
    
    Parameters:
        df: DataFrame containing LET data
        outliers_df: DataFrame with outlier indices and replacement values
        let_column: Name of the LET column to analyze
        outliers_method: Method used for outlier detection (e.g., 'lof', 'dbscan')
        replace_method: Method used for replacement (e.g., 'median', 'local_mean')
        x_bounds: Tuple of (x_min, x_max) for analysis region
        cbar_label: Label for colorbar (method-specific)
        cut_in_um: Cut value in micrometers
        voxel_in_um: Voxel size in micrometers
        figsize: Figure size as (width, height)
        filename: Base filename for saving (without extension, auto-generated if None)
        save_fig: Whether to save the figure
        save_dir: Directory to save the figure
    """
    # Filter data to analysis region
    x_mask = (df['x'] >= x_bounds[0]) & (df['x'] <= x_bounds[1])
    x_data_filtered = df.loc[x_mask, 'x']
    let_column_filtered = df.loc[x_mask, let_column]
    
    if x_data_filtered.empty:
        logger.warning(f"No data points in analysis range {x_bounds}")
        return
    
    # Filter outliers to analysis region
    analysis_outliers = outliers_df[outliers_df.index.isin(x_data_filtered.index)]
    
    # Create figure with subplot for statistics
    fig = plt.figure(figsize=figsize)
    
    # Main plot area (left side)
    ax_main = plt.axes([0.1, 0.1, 0.6, 0.8])
    
    # Statistics subplot area (right side)
    ax_stats = plt.axes([0.75, 0.1, 0.23, 0.8])
    ax_stats.axis('off')
    
    # Plot the LET profile as a scatter plot
    ax_main.scatter(
        x_data_filtered, 
        let_column_filtered, 
        label='LET Profile', 
        s=3.0, 
        color='k', 
        alpha=0.5
    )
    
    if not analysis_outliers.empty:
        # Extract outliers and replacement values
        outliers = analysis_outliers['outliers']
        replacements = analysis_outliers['replacements']
        
        # Calculate outlier scores for marker sizing (use replacement difference as proxy)
        outlier_scores = np.abs(outliers - replacements)
        if len(outlier_scores) > 0 and outlier_scores.max() > 0:
            radius = outlier_scores / outlier_scores.max()
        else:
            radius = np.ones(len(outlier_scores)) * 0.5
        
        # Plot outliers as large circles with sizes scaled by scores
        sc_outliers = ax_main.scatter(
            x_data_filtered.loc[outliers.index], 
            outliers, 
            s=1000.0 * radius, 
            c=radius,
            cmap='autumn_r',
            alpha=0.5,
            label='Outliers'
        )

        # Plot the replacement values with a different marker
        ax_main.scatter(
            x_data_filtered.loc[replacements.index], 
            replacements, 
            s=10.0,
            marker='^', 
            color='blue',
            label='Replacements'
        )
        
        # Add a colorbar corresponding to the outlier scores
        cbar = plt.colorbar(sc_outliers, ax=ax_main)
        cbar.set_label(cbar_label)
    
    # Set labels and compose the plot title
    title = 'LET Profile with Outliers and Replacements'
    if cut_in_um and voxel_in_um:
        title += f"\nCut: {cut_in_um} μm, Voxel: {voxel_in_um} μm"
        
    ax_main.set_xlabel('x [mm]')
    ax_main.set_ylabel('LET [keV μm⁻¹]')
    ax_main.legend()
    ax_main.set_title(title)
    ax_main.grid(True, alpha=0.3)
    
    # Statistics summary in right panel
    stats_text = f"Analysis Region:\n[{x_bounds[0]:.1f}, {x_bounds[1]:.1f}] mm\n\n"
    stats_text += f"Data Points: {len(x_data_filtered)}\n"
    stats_text += f"Cut Value: {cut_in_um} μm\n"
    stats_text += f"Voxel Size: {voxel_in_um} μm\n\n"
    
    if not analysis_outliers.empty and len(analysis_outliers) > 0:
        stats_text += f"Outliers Detected: {len(analysis_outliers)}\n"
        stats_text += f"Outlier Rate: {100*len(analysis_outliers)/len(x_data_filtered):.1f}%\n"
        stats_text += f"Detection Method: {outliers_method.upper()}\n"
        stats_text += f"Replacement Method: {replace_method.replace('_', ' ').title()}\n\n"
        
        orig_values = analysis_outliers['outliers']
        repl_values = analysis_outliers['replacements']
        
        stats_text += "Outlier Stats:\n"
        stats_text += f"  Mean: {orig_values.mean():.2f}\n"
        stats_text += f"  Std: {orig_values.std():.2f}\n"
        stats_text += f"  Min: {orig_values.min():.2f}\n"
        stats_text += f"  Max: {orig_values.max():.2f}\n\n"
        
        stats_text += "Replacement Stats:\n"
        stats_text += f"  Mean: {repl_values.mean():.2f}\n"
        stats_text += f"  Std: {repl_values.std():.2f}\n"
        stats_text += f"  Min: {repl_values.min():.2f}\n"
        stats_text += f"  Max: {repl_values.max():.2f}\n"
    else:
        stats_text += "No Outliers Detected"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout to fit labels and title
    plt.tight_layout()
    
    if save_fig and save_dir is not None:
        # Use provided filename or generate method-specific filename
        if filename is None:
            filename = f"outliers_{outliers_method}_{replace_method}_{let_column}"
        save_figure(fig, save_dir, filename, formats=['png', 'eps'], close_after_save=False)
    
    # Note: Figure display removed to prevent GUI window when called from CLI


def plot_outlier_detection_results(
    df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    let_column: str,
    method_name: str,
    x_bounds: Tuple[float, float] = (32.5, 40.0),
    figsize: Tuple[float, float] = (12, 6),
    save_fig: bool = False,
    save_dir: Optional[str] = None
) -> None:
    """
    Plot the results of outlier detection on LET data.
    
    Parameters:
        df: Original DataFrame with LET data
        outliers_df: DataFrame with detected outliers and replacements
        let_column: Name of the LET column to plot
        method_name: Name of the outlier detection method used
        x_bounds: Tuple of (x_min, x_max) for plotting range
        figsize: Figure size as (width, height)
        save_fig: Whether to save the figure
        save_dir: Directory to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                sharex=True, height_ratios=[2, 1])
    
    # Filter data to x_bounds
    x_mask = (df['x'] >= x_bounds[0]) & (df['x'] <= x_bounds[1])
    plot_df = df[x_mask]
    
    if plot_df.empty:
        logger.warning(f"No data points in range {x_bounds}")
        return
    
    # Top plot: LET profile with outliers marked
    ax1.plot(plot_df['x'], plot_df[let_column], 'b-', alpha=0.7, 
            linewidth=1.5, label='Original LET')
    
    if not outliers_df.empty:
        # Mark outliers
        outlier_subset = outliers_df[outliers_df.index.isin(plot_df.index)]
        if not outlier_subset.empty:
            outlier_x = df.loc[outlier_subset.index, 'x']
            outlier_y = df.loc[outlier_subset.index, let_column]
            replacement_y = outlier_subset['replacements']
            
            ax1.scatter(outlier_x, outlier_y, c='red', s=50, 
                    marker='x', label='Outliers', zorder=5)
            ax1.scatter(outlier_x, replacement_y, c='green', s=50, 
                    marker='o', alpha=0.7, label='Replacements', zorder=5)
            
            # Connect with lines
            for i, idx in enumerate(outlier_subset.index):
                x_pos = df.loc[idx, 'x']
                y_orig = df.loc[idx, let_column]
                y_repl = replacement_y.iloc[i]
                ax1.plot([x_pos, x_pos], [y_orig, y_repl], 
                        'gray', '--', alpha=0.5)
    
    ax1.set_ylabel('LET Value')
    ax1.set_title(f'Outlier Detection Results - {method_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Outlier indicators
    outlier_mask = np.zeros(len(plot_df), dtype=bool)
    if not outliers_df.empty:
        outlier_indices = outliers_df.index.intersection(plot_df.index)
        outlier_positions = plot_df.index.get_indexer(outlier_indices)
        outlier_mask[outlier_positions] = True
    
    ax2.fill_between(plot_df['x'], 0, outlier_mask.astype(int), 
                    alpha=0.5, color='red', label='Outlier Regions')
    ax2.set_xlabel('Depth (mm)')
    ax2.set_ylabel('Outlier\nIndicator')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig and save_dir is not None:
        filename = f"detection_{method_name}"
        save_figure(fig, save_dir, filename, formats=['png'], close_after_save=False)
    
    plt.show()