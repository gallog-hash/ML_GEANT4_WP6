# src/utils/plot_utils.py

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from utils import VAELogger

logger = VAELogger("PlotUtils", "info").get_logger()

def barplot_zero_values_percentage(
    perc_zero_df: pd.DataFrame, 
    figsize: tuple[int, int] = (10, 6), 
    xtick_fontsize: int = 10, 
    highlight_color: Optional[str] = None,
    **kwargs: Any
) -> Figure:
    """
    Create a Seaborn bar plot showing the percentage of values lower than or
    equal to 0 in each column of a DataFrame. 

    Parameters:
    - perc_zero_df (DataFrame): A DataFrame containing column names
        as index and the corresponding percentage of values equal to 0 as values.
    - figsize (tuple, optional): Figure size (width, height). Defaults to (10,6). 
    - xtick_fontsize (int, optional): Font size for x-axis tick labels. Defaults
        to 10. 
    - highlight_color (str or None, optional): Color to highlight columns with
        all zeros or 100% values. If None, uses the same color with full alpha.
        Defaults to None.
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
        are provided, they are used to add a second line in the title of the plot.

    Returns:
    - Figure: The created matplotlib figure.
    """
    # Validate input
    if perc_zero_df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    # Extract values as Series for cleaner type handling
    if isinstance(perc_zero_df, pd.Series):
        values = perc_zero_df
    else:
        # It's a DataFrame, get the first column or squeeze if single column
        values = perc_zero_df.iloc[:, 0] if perc_zero_df.shape[1] > 0 else perc_zero_df.squeeze()
    
    # Set Seaborn style for the plot
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get default color palette
        default_colors = sns.color_palette('muted')
        default_color = default_colors[3] if len(default_colors) > 3 else 'blue'
        
        # Separate data for different alpha values
        mask_100 = (values == 100)
        mask_less_100 = ~mask_100
        
        legend_handles: List[Rectangle] = []
        legend_labels: List[str] = []
        
        # Plot bars with values < 100%
        if isinstance(mask_less_100, pd.Series) and mask_less_100.any():
            bar_plot_1 = sns.barplot(
                x=values[mask_less_100].index,
                y=values[mask_less_100],
                color=default_color,
                alpha=0.7, 
                gap=0.2,
                ax=ax
            )
            
            # Get handle for legend
            containers = bar_plot_1.containers
            if containers:
                first_container = containers[0]
                if isinstance(first_container, BarContainer) and first_container.patches:
                    legend_handles.append(first_container.patches[0])
                    legend_labels.append('% < 100%')
        
        # Plot bars with 100% values if any exist
        if isinstance(mask_100, pd.Series) and mask_100.any():
            highlight_col = highlight_color if highlight_color is not None else default_color
            
            bar_plot_2 = sns.barplot(
                x=values[mask_100].index,
                y=values[mask_100],
                color=highlight_col,
                alpha=1.0, 
                gap=0.2,
                ax=ax
            )
            
            # Get handle for legend
            containers = bar_plot_2.containers
            if containers:
                last_container = containers[-1]
                if isinstance(last_container, BarContainer) and last_container.patches:
                    legend_handles.append(last_container.patches[0])
                    legend_labels.append('% = 100%')
        
        # Add legend if we have handles
        if legend_handles and legend_labels:
            ax.legend(handles=legend_handles, labels=legend_labels, loc='lower right')
                
        # Set labels and title
        title = 'Percentage of Values = 0 in Each Column'
        
        # Add additional title information if provided
        cut_in_um = kwargs.get('cut_in_um')
        voxel_in_um = kwargs.get('voxel_in_um')
        if cut_in_um is not None and voxel_in_um is not None:
            title += f"\nCut: {cut_in_um} µm, Voxel: {voxel_in_um} µm"
            
        ax.set_ylabel('% of Values = 0')
        ax.set_xlabel('')
        ax.set_title(title)

        # Set x-axis tick labels with specified fontsize
        current_labels = ax.get_xticklabels()
        ax.set_xticklabels(
            current_labels, 
            rotation='vertical', 
            verticalalignment='top', 
            fontsize=xtick_fontsize
        )

        plt.tight_layout()
        
    return fig

def plot_twin_profile(
    ax: Axes, 
    twin_profiles: Union[pd.DataFrame, List[pd.DataFrame]], 
    **kwargs: Any
) -> Axes:
    """
    Plot the twin_profile(s) on an independent y-axis opposite to the 
    given one 'ax'.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot the second profile on.
        twin_profiles (Union[DataFrame, List[DataFrame]]): A DataFrame or 
            list of DataFrames containing the second profile data.  
        **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
        matplotlib.axes.Axes: The axis with the dose profile plotted.
    """
    # Instantiate a second Axes that shares the same x-axis
    ax2 = ax.twinx()  
    twin_offset = kwargs.get('twin_offset', 1)
    ax2.spines.right.set_position(("axes", twin_offset))
    
    # Convert twin_profiles to a list if it is single DataFrame
    if isinstance(twin_profiles, pd.DataFrame):
        twin_profiles = [twin_profiles]
    
    # Set the color of the profile(s)
    twin_color = kwargs.get('twin_color', 'red')
    
    # Set the linestyle of the profile(s)
    linestyle = ['-', '--', ':', '-.']
    
    # Set the label of the profile(s)
    label = kwargs.get('twin_plotlabel')
    suffix = kwargs.get('twin_plotsuffix')
    labels: List[Optional[str]] = []
    
    if label is not None:
        if len(twin_profiles) > 1 and suffix:
            labels = [f"{label} - {suffix[i]}" for i in range(len(twin_profiles))]
        elif len(twin_profiles) > 1:
            labels = [f"{label} - DataFrame {i}" for i in range(len(twin_profiles))]
        else:
            labels = [label]
    else:
        labels = [None] * len(twin_profiles)
    
    for j, profile in enumerate(twin_profiles):
        if profile.shape[1] < 2:
            continue  # Skip profiles without at least 2 columns
            
        x_values, y_values = profile.iloc[:, 0], profile.iloc[:, 1]
        linestyle_idx = j % len(linestyle)
        
        ax2.plot(
            x_values, 
            y_values, 
            color=twin_color, 
            linewidth=kwargs.get('twin_linewidth', 1.0), 
            linestyle=linestyle[linestyle_idx],
            alpha=kwargs.get('twin_alpha', 0.5), 
            label=labels[j]
        )
        
    ax2.set_ylabel(
        kwargs.get('twin_ylabel', '[a. u.]'), 
        color=twin_color, 
        fontsize=kwargs.get('ylabel_fontsize')
    )
    ax2.grid(None)
    ax2.tick_params(axis='y', color=twin_color, labelcolor=twin_color)
    
    plt.setp(ax2.spines.values(), color=twin_color, linewidth=3)
    plt.setp(
        ax2.get_yticklabels(), 
        fontsize=kwargs.get('yticks_fontsize')
    )
    
    return ax2

def plot_let_profile(
    df: pd.DataFrame, 
    column_type: str = 'track', 
    marker_size: int = 8, 
    marker_type: str = '.',
    dose_profile: Optional[pd.DataFrame] = None,
    subplot_location: Optional[List[float]] = None,
    subplot_x_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> Figure:
    """
    Create a scatter plot for LET profiles.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_type (str, optional): Type of columns to include ('track' or
        'dose'). Defaults to 'track'.
    - marker_size (int, optional): Size of markers in the scatter plot. 
        Defaults to 8. 
    - marker_type (str, optional): Marker type in the scatter plot. 
        Defaults to '.'. 
    - dose_profile (DataFrame, optional): DataFrame containing the dose 
        profile.
    - subplot_location (list, optional): Location and size of the subplot.
        Defaults to None.
    - subplot_x_range (tuple, optional): Range of x data for the subplot. 
        If provided, the subplot will display data within this range. 
        Defaults to None. 
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 
        'voxel_in_um' are provided, they are used to add a second line 
        in the title of the plot.

    Returns:
    - Figure: The created matplotlib figure.
    
    Raises:
    - ValueError: If column_type is invalid or required columns are missing.
    """
    # Validate inputs
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    if column_type not in ['track', 'dose']:
        raise ValueError("Invalid column_type. Use 'track' or 'dose'.")
    
    sns.set_style("whitegrid")
    
    # Define search patterns based on column type
    search_patterns = ['LTT'] if column_type == 'track' else ['LDT']
    
    # Check if df contains columns with the specified search patterns
    matching_columns = [
        col for col in df.columns 
        if any(pattern in col for pattern in search_patterns)
    ]
    
    if not matching_columns:
        raise ValueError(
            f"The DataFrame does not contain columns with the specified "
            f"search patterns for '{column_type}'. "
            f"Expected patterns: {', '.join(search_patterns)}"
        )
    
    # Determine which feature to plot (use first matching column)
    plot_feature = matching_columns[0]
    
    # Get plot parameters
    xlim = kwargs.get('xlim', (0, None))
    figsize = kwargs.get('figsize', (12, 8))
    
    # Limit dose_profile to match the x range of df
    filtered_dose_profile = None
    if dose_profile is not None and not dose_profile.empty:
        if 'x' in df.columns:
            max_x_df = df['x'].max()
            filtered_dose_profile = dose_profile[
                dose_profile['x'] <= max_x_df
            ].copy()
        else:
            filtered_dose_profile = dose_profile.copy()
    
    # Create main plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot dose distribution if dose_profile is provided
    ax2: Optional[Axes] = None
    if filtered_dose_profile is not None:
        ax2 = plot_twin_profile(ax, filtered_dose_profile, **kwargs)
    
    # Use 'x' if available, otherwise use 'i'
    x_column = 'x' if 'x' in df.columns else 'i'
    if x_column not in df.columns:
        raise ValueError("DataFrame must contain either 'x' or 'i' column")
    
    x_values = df[x_column]
    y_values = df[plot_feature]
    
    # Main scatter plot
    ax.scatter(
        x_values, 
        y_values, 
        label=plot_feature, 
        s=marker_size, 
        marker=marker_type
    )
    
    # Add subplot if both location and range are specified
    _add_subplot_if_needed(
        fig, x_values, y_values, subplot_location, subplot_x_range,
        xlim, marker_size, marker_type
    )
    
    # Configure main plot appearance
    _configure_plot_appearance(ax, column_type, xlim, kwargs)
    
    # Handle legends from both axes
    _configure_legends(ax, ax2, filtered_dose_profile, kwargs)
    
    # Configure tick font sizes
    _configure_tick_fonts(ax, ax2, kwargs)
    
    return fig


def _add_subplot_if_needed(
    fig: Figure, 
    x_values: pd.Series, 
    y_values: pd.Series,
    subplot_location: Optional[List[float]], 
    subplot_x_range: Optional[Tuple[float, float]],
    xlim: Tuple[float, Optional[float]], 
    marker_size: int, 
    marker_type: str
) -> None:
    """Add subplot if conditions are met."""
    # Check if subplot should be added
    should_add_subplot = (
        subplot_location is not None and 
        subplot_x_range is not None and 
        xlim[0] == 0.0
    )
    
    if not should_add_subplot:
        return
    
    # Create subplot axes
    ax_sub = fig.add_axes(subplot_location)
    
    min_x, max_x = subplot_x_range
    mask = (x_values >= min_x) & (x_values <= max_x)
    sub_x_values = x_values[mask]
    sub_y_values = y_values[mask]
    
    ax_sub.scatter(
        sub_x_values, 
        sub_y_values, 
        s=marker_size, 
        marker=marker_type
    )
    
    plt.setp(ax_sub.spines.values(), color='k')


def _configure_plot_appearance(
    ax: Axes, 
    feature: str, 
    xlim: Tuple[Optional[float], Optional[float]],
    kwargs: dict[str, Any]
) -> None:
    """Configure plot title, labels, and appearance."""
    # Set title
    if feature == 'track':
        feature = 'LET-track'
    elif feature == 'dose':
        feature = 'LET-dose'

    title = f"Scatter Plot for {feature}"
    cut_in_um = kwargs.get('cut_in_um')
    voxel_in_um = kwargs.get('voxel_in_um')
    
    if cut_in_um is not None and voxel_in_um is not None:
        title += f"\nCut: {cut_in_um} µm, Voxel: {voxel_in_um} µm"
    
    ax.set_title(title, fontsize=kwargs.get('title_fontsize'))
    
    # Set labels
    ax.set_xlabel('x [mm]', fontsize=kwargs.get('xlabel_fontsize'))
    ylabel = f"{feature} [keV $\\mu$m$^{{-1}}$]"
    ax.set_ylabel(ylabel, fontsize=kwargs.get('ylabel_fontsize'))
    
    # Set limits and grid
    ax.set_xlim(xlim)
    ax.grid('both', color='k', linestyle=':')
    plt.setp(ax.spines.values(), color='k')


def _configure_legends(
    ax: Axes, 
    ax2: Optional[Axes], 
    dose_profile: Optional[pd.DataFrame],
    kwargs: dict[str, Any]
) -> None:
    """Configure legends for main and secondary axes."""
    handles1, labels1 = ax.get_legend_handles_labels()
    
    if dose_profile is not None and ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2
    else:
        all_handles = handles1
        all_labels = labels1
    
    ax.legend(
        all_handles, 
        all_labels, 
        bbox_to_anchor=(1.10, 1.0), 
        loc='upper left',
        fontsize=kwargs.get('legend_fontsize')
    )


def _configure_tick_fonts(
    ax: Axes, 
    ax2: Optional[Axes], 
    kwargs: dict[str, Any]
) -> None:
    """Configure font sizes for tick labels."""
    xticks_fontsize = kwargs.get('xticks_fontsize')
    yticks_fontsize = kwargs.get('yticks_fontsize')
    
    plt.setp(ax.get_xticklabels(), fontsize=xticks_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
    
    if ax2 is not None:
        plt.setp(ax2.spines.values(), color='k')
        plt.setp(ax2.get_yticklabels(), fontsize=yticks_fontsize)

def create_correlation_plot_title(
    df: pd.DataFrame, 
    title_prefix: str, 
    **kwargs: Any
) -> str:
    """
    Create the title for a correlation plot based on the 'x' or 'i' values 
    in the DataFrame. 

    Parameters:
    - df (DataFrame): The input DataFrame.
    - title_prefix (str): The prefix text for the title.
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 
        'voxel_in_um' are provided, they are used to add a second line 
        in the title.

    Returns:
    - str: The title for the correlation heatmap.
    
    Raises:
    - ValueError: If DataFrame is empty or missing required columns.
    """
    # Validate input
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    # Determine which column to use for x-values
    x_column = _get_x_column(df)
    x_values = df[x_column].values
    
    # Get first and last x values
    first_x = x_values[0]
    last_x = x_values[-1]
    
    # Create base title
    title = _format_base_title(title_prefix, first_x, last_x)
    
    # Add optional parameters if provided
    title = _add_optional_parameters(title, kwargs)
    
    return title


def _get_x_column(df: pd.DataFrame) -> str:
    """
    Determine which column to use for x-values ('x' or 'i').
    
    Parameters:
    - df (DataFrame): The input DataFrame.
    
    Returns:
    - str: The column name to use ('x' or 'i').
    
    Raises:
    - ValueError: If neither 'x' nor 'i' column is found.
    """
    if 'x' in df.columns:
        return 'x'
    elif 'i' in df.columns:
        return 'i'
    else:
        raise ValueError("DataFrame must contain either 'x' or 'i' column")


def _format_base_title(
    title_prefix: str, 
    first_x: float, 
    last_x: float
) -> str:
    """
    Format the base title with x-value range.
    
    Parameters:
    - title_prefix (str): The prefix text for the title.
    - first_x (float): The first x-value.
    - last_x (float): The last x-value.
    
    Returns:
    - str: The formatted base title.
    """
    template = '{prefix} [{first} $\\leq$ x [mm] $\\leq$ {last}]'
    
    return template.format(
        prefix=title_prefix,
        first=first_x,
        last=round(last_x, 3)
    )


def _add_optional_parameters(title: str, kwargs: dict[str, Any]) -> str:
    """
    Add optional parameters to the title if provided.
    
    Parameters:
    - title (str): The base title.
    - kwargs (dict): Additional keyword arguments.
    
    Returns:
    - str: The title with optional parameters added.
    """
    cut_in_um = kwargs.get('cut_in_um')
    voxel_in_um = kwargs.get('voxel_in_um')
    
    if cut_in_um is not None and voxel_in_um is not None:
        title += f"\nCut: {cut_in_um} µm, Voxel: {voxel_in_um} µm"
    
    return title


def plot_correlation_df(
    corr_dict: pd.DataFrame, 
    figsize: Tuple[int, int] = (4, 8), 
    plot_title: str = 'Correlation Heatmap',
    **kwargs: Any
) -> Figure:
    """
    Plot a correlation heatmap based on a correlation DataFrame.

    Parameters:
    - corr_dict (DataFrame): A DataFrame containing correlation coefficients.
    - figsize (tuple, optional): Figure size (width, height). 
      Defaults to (4, 8).
    - plot_title (str, optional): Title of the plot. Defaults to 
      'Correlation Heatmap'. 
    - **kwargs: Additional keyword arguments for customizing the heatmap.
      Supported options:
      - cmap (str): Colormap for the heatmap. Defaults to 'coolwarm'.
      - annot (bool): Whether to annotate cells. Defaults to True.
      - fmt (str): String formatting code for annotations. Defaults to '.2f'.
      - vmin (float): Minimum value for colormap. Defaults to -1.
      - vmax (float): Maximum value for colormap. Defaults to 1.
      - title_fontsize (int): Font size for the title.
      - cbar_kws (dict): Keyword arguments for colorbar.

    Returns:
    - Figure: The created matplotlib figure.
    
    Raises:
    - ValueError: If DataFrame is empty or contains invalid data.
    """
    # Validate input
    if corr_dict.empty:
        raise ValueError("Correlation DataFrame cannot be empty")
    
    # Clean the data by removing NaN values
    clean_corr_data = corr_dict.dropna()
    
    if clean_corr_data.empty:
        raise ValueError("DataFrame contains only NaN values after cleaning")
    
    # Extract heatmap parameters from kwargs
    heatmap_params = _extract_heatmap_parameters(kwargs)
    
    # Create the plot
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the heatmap
        _create_heatmap(ax, clean_corr_data, heatmap_params)
        
        # Set title with optional font size
        title_fontsize = kwargs.get('title_fontsize')
        ax.set_title(plot_title, fontsize=title_fontsize)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
    return fig


def _extract_heatmap_parameters(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and validate heatmap parameters from kwargs.
    
    Parameters:
    - kwargs (dict): Keyword arguments containing heatmap parameters.
    
    Returns:
    - dict: Dictionary of heatmap parameters with defaults.
    """
    return {
        'annot': kwargs.get('annot', True),
        'cmap': kwargs.get('cmap', 'coolwarm'),
        'fmt': kwargs.get('fmt', '.2f'),
        'vmin': kwargs.get('vmin', -1),
        'vmax': kwargs.get('vmax', 1),
        'cbar_kws': kwargs.get('cbar_kws', {}),
        'square': kwargs.get('square', False),
        'linewidths': kwargs.get('linewidths', 0.5),
        'cbar': kwargs.get('cbar', True)
    }


def _create_heatmap(
    ax: plt.Axes, 
    data: pd.DataFrame, 
    params: dict[str, Any]
) -> None:
    """
    Create the heatmap on the given axes.
    
    Parameters:
    - ax (matplotlib.axes.Axes): The axes to plot on.
    - data (DataFrame): The correlation data to plot.
    - params (dict): Parameters for the heatmap.
    """
    try:
        sns.heatmap(
            data,
            ax=ax,
            annot=params['annot'],
            cmap=params['cmap'],
            fmt=params['fmt'],
            vmin=params['vmin'],
            vmax=params['vmax'],
            cbar_kws=params['cbar_kws'],
            square=params['square'],
            linewidths=params['linewidths'],
            cbar=params['cbar']
        )
    except Exception as e:
        raise ValueError(f"Failed to create heatmap: {str(e)}") from e


def combined_plot_comparison(
    left_data: Union[pd.DataFrame, pd.Series, Dict[str, np.ndarray]],
    right_data: Union[pd.DataFrame, pd.Series, Dict[str, np.ndarray]],
    abs_value: bool = True,
    left_ylabel: Optional[str] = None,
    right_ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    sorting_plot: str = 'left',
    sort_order: str = 'ascending',
    figsize: Tuple[int, int] = (12, 6),
    **kwargs: Any
) -> Figure:
    """
    Create a combined plot comparing two sets of data.

    Parameters:
    - left_data: Data for the left plot. Can be a Series, DataFrame,
      or dictionary, each with a single column and N rows.
    - right_data: Data for the right plot. Can be a Series,
      DataFrame, or dictionary, each with a single column and N rows.
    - abs_value (bool, optional): Whether to plot the absolute value. 
      Defaults to True.
    - left_ylabel (str, optional): Y-label for the left plot. 
      Defaults to None.
    - right_ylabel (str, optional): Y-label for the right plot. 
      Defaults to None.
    - xlabel (str, optional): X-label for the plot. Defaults to None.
    - sorting_plot (str, optional): Data to consider for plot ordering, 
      'left' or 'right'. Defaults to 'left'.
    - sort_order (str, optional): Sort order for plotting, 'ascending' or
      'descending'. Defaults to 'ascending'.
    - figsize (tuple, optional): Figure size (width, height). 
      Defaults to (12, 6). 
    - **kwargs: Additional keyword arguments including:
      - cut_in_um, voxel_in_um: Used for title
      - left_ylim, right_ylim: Y-axis limits for each plot

    Returns:
    - Figure: The created matplotlib figure.
    
    Raises:
    - ValueError: If input data is invalid.
    """
    # Prepare and sort the data
    left_sorted, right_sorted = _prepare_and_sort_data(
        left_data=left_data,
        right_data=right_data,
        sorting_plot=sorting_plot,
        sort_order=sort_order
    )
    
    # Apply absolute value if requested
    if abs_value:
        left_sorted = left_sorted.abs()
        right_sorted = right_sorted.abs()
    
    # Create the plot
    fig = _create_comparison_plot(
        left_sorted, right_sorted, figsize, left_ylabel, 
        right_ylabel, xlabel, kwargs
    )
    
    return fig

def _prepare_and_sort_data(
    left_data: Union[Dict[str, np.ndarray], pd.Series, pd.DataFrame],
    right_data: Union[Dict[str, np.ndarray], pd.Series, pd.DataFrame],
    sorting_plot: str = 'left',
    sort_order: str = 'ascending'
) -> Tuple[pd.DataFrame, pd.DataFrame]: 
    """
    Prepare and sort the input data for plotting.

    Parameters:
    - left_data: Data for the left plot.
    - right_data: Data for the right plot.
    - sorting_plot (str, optional): Data to consider for plot ordering, 
      'left' or 'right'. Defaults to 'left'. 
    - sort_order (str, optional): Sort order for plotting, 'ascending' or
      'descending'. Defaults to 'ascending'. 

    Returns:
    - Tuple[DataFrame, DataFrame]: Sorted left and right data.
    
    Raises:
    - ValueError: If input data is invalid or incompatible.
    """
    # Validate and convert inputs
    left_df = _convert_to_dataframe(left_data, "left_data")
    right_df = _convert_to_dataframe(right_data, "right_data")
    
    # Validate data structure
    _validate_dataframe_structure(left_df, right_df)
    
    # Handle index alignment
    left_aligned, right_aligned = _align_dataframes(left_df, right_df)
    
    # Sort data based on specified criteria
    return _sort_dataframes(
        left_aligned, right_aligned, sorting_plot, sort_order
    )


def _convert_to_dataframe(
    data: Union[Dict[str, np.ndarray], pd.Series, pd.DataFrame],
    data_name: str
) -> pd.DataFrame:
    """Convert input data to DataFrame format."""
    if isinstance(data, dict):
        return pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        return data.to_frame()
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        raise ValueError(f"{data_name} must be a dict, Series, or DataFrame")


def _validate_dataframe_structure(
    left_df: pd.DataFrame, 
    right_df: pd.DataFrame
) -> None:
    """Validate that DataFrames have the correct structure."""
    if left_df.shape[1] != 1:
        raise ValueError("Left DataFrame must have exactly one column")
    
    if right_df.shape[1] != 1:
        raise ValueError("Right DataFrame must have exactly one column")
    
    if left_df.empty or right_df.empty:
        raise ValueError("Input DataFrames cannot be empty")


def _align_dataframes(
    left_df: pd.DataFrame, 
    right_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align DataFrames based on common indices."""
    common_index = left_df.index.intersection(right_df.index)
    
    if len(common_index) == 0:
        raise ValueError(
            "No common rows between 'left_data' and 'right_data'"
        )
    
    # Check if alignment is needed
    left_needs_alignment = len(common_index) != left_df.shape[0]
    right_needs_alignment = len(common_index) != right_df.shape[0]
    
    if left_needs_alignment or right_needs_alignment:
        print(
            "Warning: Number or names of rows in 'left_data' and "
            "'right_data' do not match. Restricting to common rows."
        )
        return left_df.loc[common_index], right_df.loc[common_index]
    
    return left_df, right_df


def _sort_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    sorting_plot: str,
    sort_order: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort DataFrames based on specified criteria."""
    # Validate parameters
    if sorting_plot not in ['left', 'right']:
        raise ValueError("sorting_plot must be 'left' or 'right'")
    
    if sort_order not in ['ascending', 'descending']:
        raise ValueError("sort_order must be 'ascending' or 'descending'")
    
    # Determine which data to sort by
    data_to_sort = left_df if sorting_plot == 'left' else right_df
    ascending = sort_order == 'ascending'
    
    # Sort the data
    sorted_data = data_to_sort.sort_values(
        by=data_to_sort.columns[0], 
        ascending=ascending
    )
    
    # Apply sorting to both DataFrames
    if sorting_plot == 'left':
        return sorted_data, right_df.loc[sorted_data.index]
    else:
        return left_df.loc[sorted_data.index], sorted_data
    
    
def _create_comparison_plot(
    left_data: pd.DataFrame,
    right_data: pd.DataFrame,
    figsize: Tuple[int, int],
    left_ylabel: Optional[str],
    right_ylabel: Optional[str],
    xlabel: Optional[str],
    kwargs: Dict[str, Any]
) -> Figure:
    """Create the comparison plot with two y-axes."""
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Configure left plot
    _configure_left_plot(ax1, left_data, left_ylabel, kwargs)
    
    # Create secondary axis and configure right plot
    ax2 = ax1.twinx()
    _configure_right_plot(ax2, right_data, right_ylabel, kwargs)
    
    # Set common xlabel
    if xlabel:
        ax1.set_xlabel(xlabel)
    
    # Add title if parameters provided
    _add_plot_title(ax1, kwargs)
    
    return fig


def _configure_left_plot(
    ax: plt.Axes,
    data: pd.DataFrame,
    ylabel: Optional[str],
    kwargs: Dict[str, Any]
) -> None:
    """Configure the left plot on the primary axis."""
    left_color = 'tab:red'
    
    # Plot data
    ax.plot(
        data.index,
        data.iloc[:, 0], 
        marker='s',
        linestyle='-',
        color=left_color,
    )
    
    # Configure appearance
    ax.tick_params(axis='y', labelcolor=left_color)
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True, which='major', zorder=1)
    
    # Set y-limits if provided
    left_ylim = kwargs.get('left_ylim', (None, None))
    ax.set_ylim(left_ylim)
    
    # Set ylabel
    if ylabel:
        ax.set_ylabel(ylabel, color=left_color)
    
    # Configure spines
    _configure_left_spines(ax)


def _configure_right_plot(
    ax: plt.Axes,
    data: pd.DataFrame,
    ylabel: Optional[str],
    kwargs: Dict[str, Any]
) -> None:
    """Configure the right plot on the secondary axis."""
    right_color = 'tab:blue'
    
    # Plot data
    ax.plot(
        data.index,
        data.iloc[:, 0], 
        marker='o',
        linestyle='-',
        color=right_color,
    )
    
    # Configure appearance
    ax.tick_params(axis='y', labelcolor=right_color)
    ax.grid(True, which='major', zorder=2)
    
    # Set y-limits if provided
    right_ylim = kwargs.get('right_ylim', (None, None))
    ax.set_ylim(right_ylim)
    
    # Set ylabel
    if ylabel:
        ax.set_ylabel(ylabel, color=right_color)
    
    # Configure spines
    _configure_right_spines(ax)


def _configure_left_spines(ax: plt.Axes) -> None:
    """Configure spines for the left axis."""
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('k')
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(2)


def _configure_right_spines(ax: plt.Axes) -> None:
    """Configure spines for the right axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color('k')
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def _add_plot_title(ax: plt.Axes, kwargs: Dict[str, Any]) -> None:
    """Add title to the plot if parameters are provided."""
    cut_in_um = kwargs.get('cut_in_um')
    voxel_in_um = kwargs.get('voxel_in_um')
    
    if cut_in_um is not None and voxel_in_um is not None:
        title = f"Cut: {cut_in_um} µm, Voxel: {voxel_in_um} µm"
        ax.set_title(title)
        
        
def plot_feature_distribution(
    df: pd.DataFrame, 
    feature: str, 
    marker_size: int = 8, 
    marker_type: str = '.',
    dose_profile: Optional[pd.DataFrame] = None,
    subplot_location: Optional[List[float]] = None,
    subplot_x_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> Figure:
    """
    Create a scatter plot to visualize the distribution of a specific feature 
    in the DataFrame. 

    Parameters:
    - df (DataFrame): The input DataFrame.
    - feature (str): The name of the feature column to plot.
    - marker_size (int, optional): Size of markers in the scatter plot. 
      Defaults to 8.  
    - marker_type (str, optional): Marker type in the scatter plot. 
      Defaults to '.'.  
    - dose_profile (DataFrame, optional): DataFrame containing the dose 
      profile.
    - subplot_location (list, optional): Location and size of the subplot.
      Defaults to None.
    - subplot_x_range (tuple, optional): Range of x data for the subplot. If
      provided, the subplot will display data within this range. 
      Defaults to None. 
    - **kwargs: Additional keyword arguments including:
      - cut_in_um, voxel_in_um: Used for title
      - xlim: X-axis limits
      - figsize: Figure size
      - Font size parameters: title_fontsize, xlabel_fontsize, 
        ylabel_fontsize, legend_fontsize, xticks_fontsize, yticks_fontsize

    Returns:
    - Figure: The created matplotlib figure.
    
    Raises:
    - ValueError: If feature is not found in DataFrame or DataFrame is empty.
    """
    # Validate inputs
    _validate_inputs(df, feature)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Get plot parameters
    figsize = kwargs.get('figsize', (12, 8))
    xlim = kwargs.get('xlim', (None, None))
    
    # Create main figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add dose profile if provided
    ax2 = _add_dose_profile(ax, dose_profile, kwargs)
    
    # Get x and y values
    x_values, y_values = _get_plot_data(df, feature)
    
    # Create main scatter plot
    ax.scatter(
        x_values, 
        y_values, 
        label=feature, 
        s=marker_size, 
        marker=marker_type
    )
    
    # Add subplot if conditions are met
    _add_subplot_if_needed(
        fig, x_values, y_values, subplot_location, 
        subplot_x_range, xlim, marker_size, marker_type
    )
    
    # Configure plot appearance
    _configure_plot_appearance(ax, feature, xlim, kwargs)
    
    # Configure legends
    _configure_feature_legends(ax, ax2, dose_profile, kwargs)
    
    # Configure tick fonts
    _configure_tick_fonts(ax, ax2, kwargs)
    
    return fig


def _validate_inputs(df: pd.DataFrame, feature: str) -> None:
    """Validate input DataFrame and feature."""
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    if feature not in df.columns:
        raise ValueError(
            f"The specified feature '{feature}' is not a column "
            "in the DataFrame."
        )


def _add_dose_profile(
    ax: Axes, 
    dose_profile: Optional[pd.DataFrame], 
    kwargs: dict[str, Any]
) -> Optional[Axes]:
    """Add dose profile to the plot if provided."""
    if dose_profile is not None and not dose_profile.empty:
        # Assuming plot_twin_profile is available from the previous refactor
        return plot_twin_profile(ax, dose_profile, **kwargs)
    return None


def _get_plot_data(
    df: pd.DataFrame, 
    feature: str
) -> Tuple[pd.Series, pd.Series]:
    """Get x and y values for plotting."""
    # Use 'x' if available, otherwise use 'i'
    if 'x' in df.columns:
        x_values = df['x']
    elif 'i' in df.columns:
        x_values = df['i']
    else:
        raise ValueError("DataFrame must contain either 'x' or 'i' column")
    
    y_values = df[feature]
    return x_values, y_values

def _configure_feature_legends(
    ax: Axes,
    ax2: Optional[Axes],
    dose_profile: Optional[pd.DataFrame],
    kwargs: dict[str, Any]
) -> None:
    """Configure legends for main and secondary axes."""
    handles1, labels1 = ax.get_legend_handles_labels()
    
    if dose_profile is not None and ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2
    else:
        all_handles = handles1
        all_labels = labels1
    
    ax.legend(
        all_handles, 
        all_labels, 
        bbox_to_anchor=(1.10, 1.0), 
        loc='upper left',
        fontsize=kwargs.get('legend_fontsize')
    )


def _configure_tick_fonts(
    ax: Axes,
    ax2: Optional[Axes],
    kwargs: dict[str, Any]
) -> None:
    """Configure font sizes for tick labels."""
    xticks_fontsize = kwargs.get('xticks_fontsize')
    yticks_fontsize = kwargs.get('yticks_fontsize')
    
    plt.setp(ax.get_xticklabels(), fontsize=xticks_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
    
    if ax2 is not None:
        plt.setp(ax2.spines.values(), color='k')
        plt.setp(ax2.get_yticklabels(), fontsize=yticks_fontsize)
        
        
def visualize_zero_values(
    df: pd.DataFrame,
    column_as_index: Optional[Union[str, int]] = None,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'rocket',
    invert_cmap: bool = False,
    primary_x_stop: Optional[int] = None,
    **kwargs: Any
) -> Figure:
    """
    Visualize missing values per feature in a DataFrame using a heatmap.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_as_index: (str or int, optional) Name or index of the column 
      to use as row indices. Defaults to None, meaning the default index 
      will be used. 
    - figsize (tuple, optional): Figure size (width, height). 
      Defaults to (12, 6).
    - cmap (str, optional): Color map for the heatmap. Defaults to 'rocket'. 
    - invert_cmap (bool, optional): Whether to invert the colormap. 
      Defaults to False.
    - primary_x_stop (int, optional): Position to draw a horizontal red line.
      Defaults to None.
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 
      'voxel_in_um' are provided, they are used to add a second line 
      in the title.

    Returns:
    - Figure: The created matplotlib figure.
    
    Raises:
    - ValueError: If DataFrame is empty or column_as_index is invalid.
    """
    # Validate inputs
    _validate_zero_viz_inputs(df, column_as_index)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for visualization
    df_processed, ytick_config = _prepare_dataframe_for_viz(
        df, column_as_index
    )
    
    # Create and configure colormap
    custom_cmap, norm = _create_custom_colormap(cmap, invert_cmap)
    
    # Create the heatmap
    _create_zero_values_heatmap(ax, df_processed, ytick_config, 
                               custom_cmap, norm)
    
    # Configure plot appearance
    _configure_zero_viz_appearance(ax, column_as_index, kwargs)
    
    # Add horizontal line if specified
    _add_horizontal_line(ax, primary_x_stop)
    
    # Configure colorbar
    _configure_zero_viz_colorbar(ax)
    
    return fig


def _validate_zero_viz_inputs(
    df: pd.DataFrame, 
    column_as_index: Optional[Union[str, int]]
) -> None:
    """Validate inputs for zero values visualization."""
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    if column_as_index is not None:
        if isinstance(column_as_index, int):
            if not (0 <= column_as_index < len(df.columns)):
                raise ValueError(
                    f"Column index {column_as_index} is out of range. "
                    f"DataFrame has {len(df.columns)} columns."
                )
        elif isinstance(column_as_index, str):
            if column_as_index not in df.columns:
                raise ValueError(
                    f"Column '{column_as_index}' not found in DataFrame"
                )
        else:
            raise ValueError(
                "column_as_index must be a string or integer, or None"
            )


def _prepare_dataframe_for_viz(
    df: pd.DataFrame, 
    column_as_index: Optional[Union[str, int]]
) -> Tuple[pd.DataFrame, Union[bool, List, int]]:
    """
    Prepare DataFrame for visualization by setting index and configuring 
    y-ticks.
    
    Returns:
        Tuple of (processed_dataframe, ytick_configuration)
    """
    df_copy = df.copy()
    yticksteps: Union[bool, List, int] = False
    
    if column_as_index is not None:
        if isinstance(column_as_index, int):
            # Set specified column as index using integer position
            index_col = df_copy.iloc[:, column_as_index]
            df_copy.set_index(index_col, inplace=True)
            df_copy.index.name = None
            df_copy.drop(df_copy.columns[column_as_index], axis=1, 
                        inplace=True)
        elif isinstance(column_as_index, str):
            # Set specified column as index using column name
            df_copy.set_index(column_as_index, inplace=True)
            df_copy.index.name = None
        
        # Configure y-tick steps based on DataFrame size
        if len(df_copy) <= 5:
            yticksteps = df_copy.index.to_list()
        else:
            yticksteps = max(1, len(df_copy) // 10)
    
    return df_copy, yticksteps


def _create_custom_colormap(
    cmap_name: str, 
    invert_cmap: bool
) -> Tuple[ListedColormap, BoundaryNorm]:
    """Create custom colormap for zero/non-zero values."""
    # Handle colormap inversion
    if invert_cmap:
        cmap_name = cmap_name + '_r'
    
    # Get the colormap
    base_cmap = plt.cm.get_cmap(cmap_name)
    
    # Extract colors - handle both discrete and continuous colormaps
    if hasattr(base_cmap, 'colors'):
        # Discrete colormap
        color1 = base_cmap.colors[0]
        color2 = base_cmap.colors[-1]
    else:
        # Continuous colormap - sample from ends
        color1 = base_cmap(0.0)
        color2 = base_cmap(1.0)
    
    # Create custom binary colormap
    custom_cmap = ListedColormap([color1, color2])
    
    # Define boundaries and norm for discrete mapping
    bounds = [0, 0.5, 1]
    norm = BoundaryNorm(bounds, custom_cmap.N)
    
    return custom_cmap, norm


def _create_zero_values_heatmap(
    ax: Axes,
    df_processed: pd.DataFrame,
    ytick_config: Union[bool, List, int],
    cmap: ListedColormap,
    norm: BoundaryNorm
) -> None:
    """Create the zero values heatmap."""
    # Create boolean mask for zero values
    zero_mask = (df_processed == 0)
    
    # Create heatmap
    sns.heatmap(
        zero_mask, 
        yticklabels=ytick_config, 
        cbar=True, 
        cmap=cmap, 
        norm=norm,
        ax=ax
    )


def _configure_zero_viz_appearance(
    ax: Axes,
    column_as_index: Optional[Union[str, int]],
    kwargs: dict[str, Any]
) -> None:
    """Configure plot title and axis labels."""
    # Set title
    title = 'Zero Values Distributions'
    cut_in_um = kwargs.get('cut_in_um')
    voxel_in_um = kwargs.get('voxel_in_um')
    
    if cut_in_um is not None and voxel_in_um is not None:
        title += f"\nCut: {cut_in_um} µm, Voxel: {voxel_in_um} µm"
    
    ax.set_title(title)
    ax.set_xlabel('Features')
    
    # Set y-label if custom index is used
    if column_as_index is not None:
        ylabel = str(column_as_index)
        ylabel = ylabel + " [mm]" if ylabel == 'x' else ylabel
        ax.set_ylabel(ylabel)


def _add_horizontal_line(ax: Axes, primary_x_stop: Optional[int]) -> None:
    """Add horizontal line at specified position if provided."""
    if primary_x_stop is not None:
        ax.axhline(y=primary_x_stop, color='red', linewidth=2)


def _configure_zero_viz_colorbar(ax: Axes) -> None:
    """Configure colorbar ticks and labels."""
    # Get the colorbar from the heatmap
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(['Non-zero', 'Zero'])
            
def plot_more_features(
    df: pd.DataFrame, 
    feature_list: List[str], 
    ylabel: str,
    marker_size: int = 8, 
    marker_types: Union[str, List[str]] = '.',
    dose_profile: Optional[pd.DataFrame] = None,
    let_total_profile: Optional[pd.DataFrame] = None,
    n_feat_per_plot: Optional[int] = None,
    **kwargs: Any
) -> List[Figure]:
    """
    Create scatter plots to visualize the distribution of multiple features.

    Parameters:
    - df (DataFrame): The input DataFrame containing features to plot.
    - feature_list (list): List of feature names to create scatter plots for.
    - ylabel (str): Y-axis label for the features being plotted.
    - marker_size (int, optional): Size of markers in scatter plot. 
      Defaults to 8.
    - marker_types (str or list of str, optional): Marker type(s) for 
      scatter plot. Defaults to '.'.
    - dose_profile (DataFrame, optional): DataFrame containing dose profile.
    - let_total_profile (DataFrame, optional): DataFrame containing LET 
      total profile.
    - n_feat_per_plot (int, optional): Number of features per plot. 
      Defaults to None (all features on one plot).
    - **kwargs: Additional keyword arguments including font sizes, limits,
      cut_in_um, voxel_in_um for plot customization.

    Returns:
    - List[Figure]: List of created matplotlib figures.
    
    Raises:
    - ValueError: If specified features are not found in DataFrame.
    """
    # Validate inputs
    _validate_feature_inputs(df, feature_list)
    
    # Set plotting style
    sns.set_style("whitegrid")
    
    # Prepare plotting parameters
    plot_params = _prepare_plot_parameters(
        feature_list, marker_types, n_feat_per_plot, kwargs
    )
    
    # Create figures
    figures = _create_feature_figures(
        df, feature_list, ylabel, marker_size, dose_profile,
        let_total_profile, plot_params, kwargs
    )
    
    return figures


def _validate_feature_inputs(
    df: pd.DataFrame, 
    feature_list: List[str]
) -> None:
    """Validate DataFrame and feature list inputs."""
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    if not feature_list:
        raise ValueError("Feature list cannot be empty")
    
    missing_features = [
        feat for feat in feature_list if feat not in df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Features not found in DataFrame: {missing_features}"
        )


def _prepare_plot_parameters(
    feature_list: List[str],
    marker_types: Union[str, List[str]],
    n_feat_per_plot: Optional[int],
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare plotting parameters and color configuration."""
    # Ensure marker_types is a list
    if isinstance(marker_types, str):
        marker_types = [marker_types]
    
    # Set default features per plot
    if n_feat_per_plot is None:
        n_feat_per_plot = len(feature_list)
    
    # Create color map
    cmap = plt.cm.get_cmap('nipy_spectral', len(feature_list))
    colors = [cmap(i) for i in range(len(feature_list))]
    
    # Determine LET column type
    let_col = 'LTT' if feature_list[0].endswith('_T') else 'LDT'
    
    # Calculate number of figures needed
    n_fig = int(np.ceil(len(feature_list) / n_feat_per_plot))
    
    return {
        'marker_types': marker_types,
        'n_feat_per_plot': n_feat_per_plot,
        'colors': colors,
        'let_col': let_col,
        'n_fig': n_fig
    }


def _create_feature_figures(
    df: pd.DataFrame,
    feature_list: List[str],
    ylabel: str,
    marker_size: int,
    dose_profile: Optional[pd.DataFrame],
    let_total_profile: Optional[pd.DataFrame],
    plot_params: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> List[Figure]:
    """Create all figures for feature plotting."""
    figures = []
    
    for i in range(plot_params['n_fig']):
        fig = _create_single_feature_figure(
            df, feature_list, ylabel, marker_size, dose_profile,
            let_total_profile, plot_params, i, kwargs
        )
        figures.append(fig)
    
    return figures


def _create_single_feature_figure(
    df: pd.DataFrame,
    feature_list: List[str],
    ylabel: str,
    marker_size: int,
    dose_profile: Optional[pd.DataFrame],
    let_total_profile: Optional[pd.DataFrame],
    plot_params: Dict[str, Any],
    fig_index: int,
    kwargs: Dict[str, Any]
) -> Figure:
    """Create a single figure with specified features."""
    # Create figure and main axis
    figsize = kwargs.get('figsize', (12, 8))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add LET total profile
    twin_stacked_ax = _add_let_total_profile(
        ax, df, let_total_profile, plot_params['let_col'], kwargs
    )
    
    # Add dose profile if provided
    ax2 = None
    if dose_profile is not None:
        ax2 = plot_twin_profile(ax, dose_profile, **kwargs)
        ax2.set_yscale(kwargs.get('yscale', 'linear'))
    
    # Plot selected features for this figure
    _plot_features_on_axis(
        ax, df, feature_list, plot_params, fig_index, marker_size
    )
    
    # Configure axis scaling
    _configure_axis_scaling(ax, twin_stacked_ax, kwargs)
    
    # Configure plot appearance
    _configure_figure_appearance(ax, ylabel, kwargs)
    
    # Configure legends
    _configure_figure_legends(
        ax, twin_stacked_ax, ax2, dose_profile, kwargs
    )
    
    # Configure tick fonts
    _configure_figure_tick_fonts(ax, twin_stacked_ax, ax2, dose_profile, kwargs)
    
    return fig


def _add_let_total_profile(
    ax: Axes,
    df: pd.DataFrame,
    let_total_profile: Optional[pd.DataFrame],
    let_col: str,
    kwargs: Dict[str, Any]
) -> Axes:
    """Add LET total profile to the plot."""
    if let_total_profile is None:
        twin_profile = df[['x', let_col]].copy()
    else:
        twin_profile = let_total_profile
    
    twin_kwargs = {
        'twin_color': 'b',
        'twin_offset': 1.15,
        'twin_alpha': kwargs.get('twin_alpha', 0.5),
        'twin_linewidth': 2,
        'twin_linestyle': 'solid',
        'twin_plotlabel': 'Total',
        'twin_ylabel': "LET total [keV $\\mu$m$^{-1}$]",
        'ylabel_fontsize': kwargs.get('ylabel_fontsize'),
        'yticks_fontsize': kwargs.get('yticks_fontsize')
    }
    
    return plot_twin_profile(ax, twin_profile, **twin_kwargs)


def _plot_features_on_axis(
    ax: Axes,
    df: pd.DataFrame,
    feature_list: List[str],
    plot_params: Dict[str, Any],
    fig_index: int,
    marker_size: int
) -> None:
    """Plot features on the given axis."""
    # Get x values
    x_values = df['x'] if 'x' in df.columns else df['i']
    
    # Set up feature range for this figure
    n_feat_per_plot = plot_params['n_feat_per_plot']
    start_idx = fig_index * n_feat_per_plot
    end_idx = (fig_index + 1) * n_feat_per_plot
    current_features = feature_list[start_idx:end_idx]
    
    # Set color cycle
    current_colors = plot_params['colors'][start_idx:end_idx]
    color_cycle = cycler(color=current_colors)
    ax.set_prop_cycle(color_cycle)
    
    # Plot each feature
    for j, feature in enumerate(current_features):
        y_values = df[feature]
        marker_idx = j % len(plot_params['marker_types'])
        marker = plot_params['marker_types'][marker_idx]
        
        ax.scatter(
            x_values, 
            y_values, 
            label=feature, 
            s=marker_size,
            marker=marker, 
            alpha=0.8
        )


def _configure_axis_scaling(
    ax: Axes,
    twin_stacked_ax: Axes,
    kwargs: Dict[str, Any]
) -> None:
    """Configure y-axis scaling for both axes."""
    yscale = kwargs.get('yscale', 'linear')
    ax.set_yscale(yscale)
    twin_stacked_ax.set_yscale(yscale)


def _configure_figure_appearance(
    ax: Axes,
    ylabel: str,
    kwargs: Dict[str, Any]
) -> None:
    """Configure figure title, labels, and grid."""
    # Set title if parameters provided
    if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
        title = (f"\nCut: {kwargs['cut_in_um']} µm, "
                f"Voxel: {kwargs['voxel_in_um']} µm")
        ax.set_title(title, fontsize=kwargs.get('title_fontsize'))
    
    # Set labels
    ax.set_xlabel('x [mm]', fontsize=kwargs.get('xlabel_fontsize'))
    ax.set_ylabel(ylabel, fontsize=kwargs.get('ylabel_fontsize'))
    
    # Set limits and appearance
    ax.set_xlim(kwargs.get('xlim', (None, None)))
    ax.grid('both', color='k', linestyle=':')
    plt.setp(ax.spines.values(), color='k')


def _configure_figure_legends(
    ax: Axes,
    twin_stacked_ax: Axes,
    ax2: Optional[Axes],
    dose_profile: Optional[pd.DataFrame],
    kwargs: Dict[str, Any]
) -> None:
    """Configure legends combining all axes."""
    # Get handles and labels from main axis
    handles1, labels1 = ax.get_legend_handles_labels()
    
    # Get handles and labels from LET total axis
    handles_s, labels_s = twin_stacked_ax.get_legend_handles_labels()
    
    # Combine handles and labels
    if dose_profile is not None and ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2 + handles_s
        all_labels = labels1 + labels2 + labels_s
    else:
        all_handles = handles1 + handles_s
        all_labels = labels1 + labels_s
    
    # Create legend
    ax.legend(
        all_handles, 
        all_labels, 
        bbox_to_anchor=(0.52, -0.11),
        loc='upper center', 
        ncols=4, 
        numpoints=1,
        scatterpoints=3, 
        markerscale=1.5, 
        scatteryoffsets=[0.5],
        fontsize=kwargs.get('legend_fontsize')
    )


def _configure_figure_tick_fonts(
    ax: Axes,
    twin_stacked_ax: Axes,
    ax2: Optional[Axes],
    dose_profile: Optional[pd.DataFrame],
    kwargs: Dict[str, Any]
) -> None:
    """Configure font sizes for all tick labels."""
    xticks_fontsize = kwargs.get('xticks_fontsize')
    yticks_fontsize = kwargs.get('yticks_fontsize')
    
    # Main axis ticks
    plt.setp(ax.get_xticklabels(), fontsize=xticks_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
    plt.setp(ax.yaxis.get_offset_text(), fontsize=yticks_fontsize)
    
    # LET total axis ticks
    plt.setp(twin_stacked_ax.get_yticklabels(), fontsize=yticks_fontsize)
    
    # Dose profile axis ticks if present
    if dose_profile is not None and ax2 is not None:
        plt.setp(ax2.spines.values(), color='k')
        plt.setp(ax2.get_yticklabels(), fontsize=yticks_fontsize)

        
def plot_more_dataframe(
    df_list: List[pd.DataFrame], 
    feature_list: List[str],
    ylabel: Optional[str] = None,
    marker_size: int = 8,
    marker_types: Union[str, List[str]] = '.',
    dose_profiles: Optional[List[pd.DataFrame]] = None,
    let_totals: Optional[List[pd.DataFrame]] = None,
    plot_suffix: Optional[List[str]] = None,
    n_feat_per_plot: Optional[int] = None,  
    **kwargs: Any  
) -> List[Figure]:
    """
    Create scatter plots for multiple DataFrames with features overlaid.

    Parameters:
    - df_list: List of input DataFrames containing features to plot.
    - feature_list: List of feature names to create scatter plots for.
    - ylabel: Y-axis label for the features. If None, uses generic label.
    - marker_size: Size of markers in scatter plot. Defaults to 8.
    - marker_types: Marker type(s) for scatter plot. Can be string or 
      list of strings.
    - dose_profiles: List of DataFrames containing dose profiles for 
      each DataFrame in df_list.
    - let_totals: List of DataFrames containing LET totals for each 
      DataFrame in df_list.
    - plot_suffix: List of suffixes for labeling different DataFrames.
    - n_feat_per_plot: Number of features per plot. If None, plots all 
      features together.
    - **kwargs: Additional keyword arguments including font sizes, limits,
      cut_in_um, voxel_in_um for plot customization.

    Returns:
    - List[Figure]: List of created matplotlib figures.
    
    Raises:
    - ValueError: If input validation fails or DataFrames are incompatible.
    """
    # Validate inputs
    _validate_dataframe_inputs(df_list, feature_list, dose_profiles, 
                              let_totals, plot_suffix, marker_types)
    
    # Set plotting style  
    sns.set_style("whitegrid")
    
    # Prepare plot parameters
    plot_params = _prepare_dataframe_plot_params(
        df_list, feature_list, marker_types, plot_suffix, 
        n_feat_per_plot, let_totals, dose_profiles
    )
    
    # Create figures
    figures = _create_dataframe_figures(
        df_list, feature_list, ylabel, marker_size, plot_params, kwargs
    )

    return figures


def _validate_dataframe_inputs(
    df_list: List[pd.DataFrame],
    feature_list: List[str], 
    dose_profiles: Optional[List[pd.DataFrame]],
    let_totals: Optional[List[pd.DataFrame]],
    plot_suffix: Optional[List[str]],
    marker_types: Union[str, List[str]]
) -> None:
    """Validate inputs for plot_more_dataframe function."""
    if not df_list:
        raise ValueError("df_list cannot be empty")
    
    if not feature_list:
        raise ValueError("feature_list cannot be empty")
    
    # Check if all DataFrames contain the required features
    for i, df in enumerate(df_list):
        if df.empty:
            raise ValueError(f"DataFrame at index {i} is empty")
        
        missing_features = [
            feat for feat in feature_list if feat not in df.columns
        ]
        if missing_features:
            raise ValueError(
                f"DataFrame at index {i} missing features: {missing_features}"
            )
    
    # Validate dose_profiles length if provided
    if dose_profiles is not None and len(dose_profiles) != len(df_list):
        raise ValueError(
            "Length of dose_profiles must match length of df_list"
        )
    
    # Validate let_totals length if provided  
    if let_totals is not None and len(let_totals) != len(df_list):
        raise ValueError(
            "Length of let_totals must match length of df_list"
        )
    
    # Validate plot_suffix length if provided
    if plot_suffix is not None and len(plot_suffix) != len(df_list):
        raise ValueError(
            "Length of plot_suffix must match length of df_list"
        )
    
    # Validate marker_types
    if isinstance(marker_types, list) and len(marker_types) != len(df_list):
        raise ValueError(
            "If marker_types is a list, its length must match df_list"
        )


def _prepare_dataframe_plot_params(
    df_list: List[pd.DataFrame],
    feature_list: List[str],
    marker_types: Union[str, List[str]],
    plot_suffix: Optional[List[str]],
    n_feat_per_plot: Optional[int],
    let_totals: Optional[List[pd.DataFrame]],
    dose_profiles: Optional[List[pd.DataFrame]]
) -> Dict[str, Any]:
    """Prepare parameters for dataframe plotting."""
    # Ensure marker_types is a list
    if isinstance(marker_types, str):
        marker_types = [marker_types] * len(df_list)
    
    # Set default plot_suffix if not provided
    if plot_suffix is None:
        plot_suffix = [f'DataFrame {i}' for i in range(len(df_list))]
    
    # Set default features per plot
    if n_feat_per_plot is None:
        n_feat_per_plot = len(feature_list)
    
    # Determine LET column type from feature list
    let_col = 'LTT' if feature_list[0].endswith('_T') else 'LDT'
    
    # Prepare LET totals if not provided
    if let_totals is None:
        let_totals = []
        for df in df_list:
            if let_col in df.columns and 'x' in df.columns:
                let_totals.append(df[['x', let_col]].copy())
            else:
                # Create empty DataFrame as placeholder
                let_totals.append(pd.DataFrame())
    
    # Check if dose profiles should be plotted
    plot_dose_profiles = (
        dose_profiles is not None and 
        all(dose is not None and not dose.empty for dose in dose_profiles)
    )
    
    return {
        'marker_types': marker_types,
        'plot_suffix': plot_suffix,  
        'n_feat_per_plot': n_feat_per_plot,
        'let_col': let_col,
        'let_totals': let_totals,
        'plot_dose_profiles': plot_dose_profiles,
        'dose_profiles': dose_profiles
    }


def _create_dataframe_figures(
    df_list: List[pd.DataFrame],
    feature_list: List[str], 
    ylabel: Optional[str],
    marker_size: int,
    plot_params: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> List[Figure]:
    """Create figures for multiple DataFrames."""
    figures = []
    n_feat_per_plot = plot_params['n_feat_per_plot']
    
    # Loop through features in chunks
    for i in range(0, len(feature_list), n_feat_per_plot):
        fig = _create_single_dataframe_figure(
            df_list, feature_list[i:i+n_feat_per_plot], ylabel,
            marker_size, plot_params, kwargs
        )
        figures.append(fig)
    
    return figures


def _create_single_dataframe_figure(
    df_list: List[pd.DataFrame],
    current_features: List[str],
    ylabel: Optional[str],
    marker_size: int,
    plot_params: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> Figure:
    """Create a single figure with multiple DataFrames plotted."""
    # Create figure and main axis
    figsize = kwargs.get('figsize', (12, 8))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add LET total profiles if available
    twin_stacked_ax = None
    if plot_params['let_totals'] and any(
        not df.empty for df in plot_params['let_totals']
    ):
        twin_stacked_ax = _add_dataframe_let_totals(
            ax, plot_params, kwargs
        )
    
    # Add dose profiles if available
    ax2 = None
    if plot_params['plot_dose_profiles']:
        ax2 = _add_dataframe_dose_profiles(
            ax, plot_params, kwargs
        )
    
    # Plot features for all DataFrames
    _plot_dataframe_features(
        ax, df_list, current_features, marker_size, plot_params
    )
    
    # Configure plot appearance
    _configure_dataframe_plot_appearance(ax, ylabel, kwargs)
    
    # Configure legends
    _configure_dataframe_legends(
        ax, twin_stacked_ax, ax2, plot_params, kwargs
    )
    
    # Configure tick fonts
    if twin_stacked_ax:
        _configure_dataframe_tick_fonts(
            ax, twin_stacked_ax, ax2, plot_params, kwargs
        )
    
    return fig


def _add_dataframe_let_totals(
    ax: Axes,
    plot_params: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> Axes:
    """Add LET total profiles for multiple DataFrames."""
    let_totals = [df for df in plot_params['let_totals'] if not df.empty]
    
    if not let_totals:
        return None
    
    twin_kwargs = {
        'twin_color': 'mediumvioletred',
        'twin_offset': 1.15,
        'twin_alpha': kwargs.get('twin_alpha', 0.5),
        'twin_linewidth': 2,
        'twin_linestyle': 'solid',
        'twin_plotlabel': plot_params['let_col'],
        'twin_ylabel': "LET total [keV $\\mu$m$^{-1}$]",
        'twin_plotsuffix': plot_params['plot_suffix'],
        'ylabel_fontsize': kwargs.get('ylabel_fontsize'),
        'yticks_fontsize': kwargs.get('yticks_fontsize')
    }
    
    return plot_twin_profile(ax, let_totals, **twin_kwargs)


def _add_dataframe_dose_profiles(
    ax: Axes,
    plot_params: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> Optional[Axes]:
    """Add dose profiles for multiple DataFrames."""
    dose_kwargs = dict(kwargs)
    dose_kwargs['twin_plotsuffix'] = plot_params['plot_suffix']
    
    ax2 = plot_twin_profile(ax, plot_params['dose_profiles'], **dose_kwargs)
    ax2.set_yscale(kwargs.get('yscale', 'linear'))
    
    return ax2


def _plot_dataframe_features(
    ax: Axes,
    df_list: List[pd.DataFrame],
    current_features: List[str],
    marker_size: int,
    plot_params: Dict[str, Any]
) -> None:
    """Plot features for all DataFrames on the given axis."""
    for j, df in enumerate(df_list):
        # Get x values (prefer 'x' over 'i')
        x_values = df['x'] if 'x' in df.columns else df['i']
        
        # Get label suffix
        label_suffix = plot_params['plot_suffix'][j]
        marker = plot_params['marker_types'][j]
        
        # Plot each feature for this DataFrame
        for feature in current_features:
            y_values = df[feature]
            
            ax.scatter(
                x_values,
                y_values,
                marker=marker,
                s=marker_size,
                alpha=0.5,
                label=f'{feature} - {label_suffix}'
            )


def _configure_dataframe_plot_appearance(
    ax: Axes,
    ylabel: Optional[str],
    kwargs: Dict[str, Any]
) -> None:
    """Configure appearance for dataframe plots."""
    # Set title if parameters provided
    cut_in_um = kwargs.get('cut_in_um')
    voxel_in_um = kwargs.get('voxel_in_um') 
    
    if cut_in_um is not None and voxel_in_um is not None:
        title = f"\nCut: {cut_in_um} µm, Voxel: {voxel_in_um} µm"
        ax.set_title(title, fontsize=kwargs.get('title_fontsize'))
    
    # Set labels
    ax.set_xlabel('x [mm]', fontsize=kwargs.get('xlabel_fontsize'))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=kwargs.get('ylabel_fontsize'))
    
    # Set limits and appearance
    ax.set_xlim(kwargs.get('xlim', (None, None)))
    ax.grid('both', color='k', linestyle=':')
    plt.setp(ax.spines.values(), color='k')


def _configure_dataframe_legends(
    ax: Axes,
    twin_stacked_ax: Optional[Axes],
    ax2: Optional[Axes],
    plot_params: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> None:
    """Configure legends for dataframe plots."""
    # Get handles and labels from main axis
    handles1, labels1 = ax.get_legend_handles_labels()
    
    all_handles = handles1.copy()
    all_labels = labels1.copy()
    
    # Add LET total handles if available
    if twin_stacked_ax is not None:
        handles_s, labels_s = twin_stacked_ax.get_legend_handles_labels()
        all_handles.extend(handles_s)
        all_labels.extend(labels_s)
    
    # Add dose profile handles if available
    if plot_params['plot_dose_profiles'] and ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles.extend(handles2)
        all_labels.extend(labels2)
    
    # Create legend
    if all_handles:
        ax.legend(
            all_handles,
            all_labels,
            bbox_to_anchor=(0.52, -0.11),
            loc='upper center',
            ncols=3,
            numpoints=1,
            scatterpoints=3,
            markerscale=1.5,
            scatteryoffsets=[0.5],
            fontsize=kwargs.get('legend_fontsize')
        )


def _configure_dataframe_tick_fonts(
    ax: Axes,
    twin_stacked_ax: Optional[Axes],
    ax2: Optional[Axes],
    plot_params: Dict[str, Any],
    kwargs: Dict[str, Any]
) -> None:
    """Configure tick fonts for dataframe plots."""
    xticks_fontsize = kwargs.get('xticks_fontsize')
    yticks_fontsize = kwargs.get('yticks_fontsize')
    
    # Main axis ticks
    plt.setp(ax.get_xticklabels(), fontsize=xticks_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=yticks_fontsize)
    plt.setp(ax.yaxis.get_offset_text(), fontsize=yticks_fontsize)
    
    # LET total axis ticks
    if twin_stacked_ax is not None:
        plt.setp(twin_stacked_ax.get_yticklabels(), 
                fontsize=yticks_fontsize)
    
    # Dose profile axis ticks if present
    if plot_params['plot_dose_profiles'] and ax2 is not None:
        plt.setp(ax2.spines.values(), color='k')
        plt.setp(ax2.get_yticklabels(), fontsize=yticks_fontsize)
        

def plot_train_test_val_distribution_df(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame = None,
    feature_names: list = None
) -> Figure:
    """
    Plot the distribution of selected features in the training, testing,
    and optionally validation datasets when splits are pandas DataFrames.

    Parameters:
      - X_train (pd.DataFrame): Training dataset.
      - X_test (pd.DataFrame): Testing dataset.
      - X_val (pd.DataFrame, optional): Validation dataset. Defaults to None.
      - feature_names (list, optional): List of feature names to plot. If not
        provided, the first two columns of X_train are used.

    Returns:
      - Figure: The created figure object.
    """
    # Use the first two columns if feature_names is not provided.
    if feature_names is None:
        feature_names = list(X_train.columns[:2])
    
    # Validate that the specified features exist in the DataFrame.
    for feature in feature_names:
        if feature not in X_train.columns:
            raise ValueError(
                f"Feature '{feature}' not found in the training DataFrame."
            )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training data.
    ax.scatter(X_train[feature_names[0]], X_train[feature_names[1]],
               label='Train', alpha=0.7, s=12, marker='o')
    
    # Plot testing data.
    ax.scatter(X_test[feature_names[0]], X_test[feature_names[1]],
               label='Test', alpha=0.3, s=10, marker='x')
    
    # Plot validation data if provided.
    if X_val is not None:
        ax.scatter(X_val[feature_names[0]], X_val[feature_names[1]],
                   label='Validation', alpha=0.5, s=10, marker='|')
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    if X_val is not None:
        ax.set_title('Distribution of Train, Test, and Validation Datasets')
    else:
        ax.set_title('Distribution of Train and Test Datasets')
    ax.legend()
    
    return fig

def plot_training_metrics(
        history: dict,
        output_dir: Path,
        metrics: Optional[List[Union[str, List[str]]]] = None,
        filename: str = "training_history"
    ) -> Figure:
        """
        Generate and save training metric plots, optionally grouped on the same
        axes.

        Args:
            history (dict): Training history with metrics.
            output_dir (Path): Directory where the plot will be saved.
            metrics (list, optional): List of metric keys or list of lists for
            grouped plotting. E.g., ['train_loss', 'val_loss'] or
            [['train_loss', 'val_loss'], ['train_acc', 'val_acc']].
            filename (str): Name of the output file (without extension).
            
        Returns:
            Figure: The created matplotlib figure object.
        """
        if not history:
            logger.warning("Empty history received; returning empty figure.")
            return plt.figure()
        
        # Default: plot each list-like item on its own axis
        if metrics is None:
            metrics = [
                [k] for k, v in history.items() 
                if isinstance(v, (list, tuple)) and len(v) > 1
            ]
        else:
            # Wrap strings into sublists so we always work with groups
            metrics = [
                [m] if isinstance(m, str) else m for m in metrics
            ]
        
        if not metrics:
            logger.warning("No valid list-like metrics to plot.")
            return plt.figure()
    
        num_plots = len(metrics)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]
            
        for ax, group in zip(axes, metrics):
            plotted = False
            for key in group:
                values = history.get(key)
                if isinstance(values, (list, tuple)) and len(values) > 1:
                    ax.plot(values, label=key.replace('_', ' ').title())
                    plotted = True
                else:
                    logger.warning(
                        f"Skipping '{key}': not list-like or too short."
                    )
            if plotted:
                ax.set_title(
                    " vs. ".join([key.replace("_", " ").title() for key in group])
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)

        fig.tight_layout()
        logger.info("Training metrics plot created successfully.")
        
        return fig
