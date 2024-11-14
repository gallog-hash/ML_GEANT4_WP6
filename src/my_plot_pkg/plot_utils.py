import os
import warnings as _warnings
from typing import Dict, List, Optional, Tuple, Union

import inflect
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import seaborn as _sns
from cycler import cycler
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap


def save_figure_to_file(
    figs: Union[_plt.Figure, List[_plt.Figure]],
    filenames: Union[str, List[str]],
    save_dir: str = '.', 
    format: str = None,
    dpi: int = 300
) -> None:
    """
    Save the current figure in a Jupyter Notebook.

    Parameters:
    - figs (plt.Figure or list of plt.Figure): The matplotlib figure(s) to save.
    - filenames (str or list of str): The name(s) of the file(s) to save the
      figure(s) to. If a single string is provided for multiple figures, an 
      index will be appended to the filename.
    - save_dir (str, optional): The directory to save the figure in. Defaults to
    the current directory ('.'). 
    - format (str, optional): The format of the saved file (e.g., 'png', 'pdf',
    'svg'). Defaults to 'png'.
    - dpi (int, optional): The resolution in dots per inch. Defaults to 300.

    Returns:
    - None
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert figs and filenames to lists if they aren't already
    if not isinstance(figs, list):
        figs = [figs]
    if not isinstance(filenames, list):
        filenames = [filenames]
        
    # If a single filename is provided for multiple figures, append an index
    if len(filenames) == 1 and len(figs) > 1:
        base_filename = filenames[0]
        filenames = []
        for i in range(len(figs)):
            if '.' in base_filename:
                name, ext = os.path.splitext(base_filename)
                filenames.append(f"{name}_{i}{ext}")
            else:
                filenames.append(f"{base_filename}_{i}")
        
    if len(figs) != len(filenames):
        raise ValueError("The number of filenames must match the number of figures.")
    
    for fig, filename in zip(figs, filenames):
        # Extract format from filename if present
        if '.' in filename:
            file_format = os.path.splitext(filename)[1][1:]
            filename = os.path.splitext(filename)[0]
            if format is None:
                format = file_format
                
        # Default format if none is specified
        if format is None:
            format = 'png'
        
        # Full path to save the figure
        filepath = os.path.join(save_dir, f"{filename}.{format}")
        
        # Save the current figure
        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved as {filename}.{format} in \n{save_dir}")


def barplot_zero_values_percentage(
    perc_zero_df: _pd.DataFrame, 
    figsize: tuple = (10, 6), 
    xtick_fontsize: int = 10, 
    highlight_color: Optional[str] = None,
    **kwargs
) -> _plt.Figure:
    """
    Create a Seaborn bar plot showing the percentage of values lower than or
    equal to 0 in each column of a DataFrame. 

    Parameters:
    - perc_zero_df (DataFrame): A DataFrame containing column names
      as index and the corresponding percentage of values equal to 0 as values.
    - figsize (tuple, optional): Figure size (width, height). Defaults to (10,
      6). 
    - xtick_fontsize (int, optional): Font size for x-axis tick labels. Defaults
      to 10. 
    - highlight_color (str or None, optional): Color to highlight columns with
      all zeros or 100% values. If None, highlight with a different alpha value.
      Defaults to None.
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title of the plot.

    Returns:
    - plt.Figure: The created matplotlib figure.
    """

    # Set Seaborn style for the plot
    with _sns.axes_style("darkgrid"):

        fig, ax = _plt.subplots(figsize=figsize)

        # Create bar plot for bars with alpha < 1.0
        bar_plot_1 = _sns.barplot(x=perc_zero_df.index,
                                    y=perc_zero_df,
                                    color=_sns.color_palette('muted')[3],
                                    alpha=0.7, gap=0.2, label='% < 100%')
        
        # Set alpha for bars in the first plot
        for bar in bar_plot_1.patches:
            if bar.get_height() == 100:
                bar.set_alpha(0.0)
        
        # Check if there are any features with 100% values
        if any(perc_zero_df == 100):
            # Create bar plot for bars with alpha == 1.0
            bar_plot_2 = _sns.barplot(x=perc_zero_df.index[perc_zero_df == 100], 
                                        y=perc_zero_df[perc_zero_df == 100],
                                        color=highlight_color or _sns.color_palette('muted')[3],
                                        alpha=1.0, gap=0.2, label='% = 100%')

            # Combine handles and labels for the legend
            handles = [bar_plot_1.containers[0][0], bar_plot_2.containers[1][0]]
            labels = ['% < 100%', '% = 100%']
            
            # Add legend for both plots
            ax.legend(handles=handles, labels=labels, loc='lower right')

        else:
            # Add legend only for the first plot
            ax.legend(handles=[bar_plot_1.containers[0][0]], labels=['% < 100%'], 
                        loc='lower right')
                
        # Set labels and title
        title = 'Percentage of Values = 0 in Each Column'
        if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
            title += f"\nCut: {kwargs['cut_in_um']} µm, " \
                f"Voxel: {kwargs['voxel_in_um']} µm"
        ax.set_ylabel('% of Values = 0')
        ax.set_xlabel('')
        ax.set_title(title)

        # Set x-axis tick labels with specified fontsize
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical', 
                           verticalalignment='top', fontsize=xtick_fontsize)

        _plt.show()
        
    return fig

def plot_twin_profile(ax, twin_profiles, **kwargs):
    """
    Plot the twin_profile(s) on an independent y-axis opposite to thebgiven one
    'ax'.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot the second profile on.
        twin_profiles (Union[DataFrame, List[DataFrame]]): A DataFrame or list
        of DataFrames containing the second profile data.  
        **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
        matplotlib.axes.Axes: The axis with the dose profile plotted.
    """
    # Instantiate a second Axes that shares the same x-axis
    ax2 = ax.twinx()  
    ax2.spines.right.set_position(("axes", kwargs.get('twin_offset', 1)))
    
    # Convert twin_profiles to a list if it is single DataFrame
    if isinstance(twin_profiles, _pd.DataFrame):
        twin_profiles = [twin_profiles]
    
    # Set the color of the profile(s)
    twin_color = kwargs.get('twin_color', 'red')
    
    # Set the linestyle of the profile(s)
    linestyle = ['-', '--', ':', '-.']
    
    # Set the label of the profile(s)
    label = kwargs.get('twin_plotlabel', None)
    suffix = kwargs.get('twin_plotsuffix', None)
    labels = []
    if label is not None:
        if len(twin_profiles) > 1 and suffix:
            for i in range(len(twin_profiles)):
                labels.append(label + f' - {suffix[i]}')
        elif len(twin_profiles) > 1 and suffix is None:
            for i in range(len(twin_profiles)):
                labels.append(label + f' - DataFrame {i}')
        else:
            labels = [label]
    else:
        labels = [None] * len(twin_profiles)
    
    for j, profile in enumerate(twin_profiles):
        x_values, y_values = profile.iloc[:, 0], profile.iloc[:, 1]
        ax2.plot(x_values, y_values, color=twin_color, 
                linewidth=kwargs.get('twin_linewidth', 1.0), 
                linestyle=linestyle[j % len(twin_profiles)],
                alpha=kwargs.get('twin_alpha', 0.5), 
                label=labels[j])
        
    ax2.set_ylabel(kwargs.get('twin_ylabel', '[a. u.]'), 
                   color=twin_color, 
                   fontsize=kwargs.get('ylabel_fontsize', None))
    ax2.grid(None)
    # ax2.set_ylim(0, None)
    ax2.tick_params(axis='y', color=twin_color, labelcolor=twin_color)
    _plt.setp(ax2.spines.values(), color=twin_color, linewidth=3)
    _plt.setp(ax2.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
    
    return ax2

def plot_let_profile(
    df: _pd.DataFrame, 
    column_type: str = 'track', 
    marker_size: int = 8, 
    marker_type: str = '.',
    dose_profile: _pd.DataFrame = None,
    subplot_location: list = None, # Default None for optional subplot
    subplot_x_range: tuple = None, # Default None for optional subplot x-range
    **kwargs
) -> _plt.Figure:
    """
    Create a scatter plot for LET profiles.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_type (str, optional): Type of columns to include ('track' or
      'dose'). Defaults to 'track'.
    - marker_size (int, optional): Size of markers in the scatter plot. Defaults
      to 8. 
    - marker_type (str, optional): Marker type in the scatter plot. Defaults to
      '.'. 
    - dose_profile (DataFrame, optional): DataFrame containing the dose profile.
    - subplot_location (list, optional): Location and size of the subplot.
      Defaults to None.
    - subplot_x_range (list, optional): Range of x data for the subplot. If
      provided, the subplot will display data within this range. Defauts to
      None. 
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title of the plot.

    Returns:
    - plt.Figure: The created matplotlib figure.
    """
    _sns.set_style("whitegrid")
    
    # Define search patterns based on column type to drop columns
    if column_type == 'track':
        search_patterns = ['LTT']
    elif column_type == 'dose':
        search_patterns = ['LDT']
    else:
        raise ValueError("Invalid column_type. Use 'track' or 'dose'.")
    
    # Check if df contains columns with the specified search patterns
    contains_patterns = any(any(pattern in col for pattern in search_patterns) 
                            for col in df.columns)

    if not contains_patterns:
        raise ValueError("The DataFrame does not contain columns with the "
                         f"specified search patterns for '{column_type}'. "
                         f"Expected patterns: {', '.join(search_patterns)}")
    
    # Determine which feature to plot
    plot_features = [search_patterns[0]]
    
    # Create a scatter plot with subplots if subplot_location is provided
    fig, ax = _plt.subplots(figsize=kwargs.get('figsize', (12, 8)))       
    
    # Plot dose distribution if dose_profile is provided
    if dose_profile is not None:
        ax2 = plot_twin_profile(ax, dose_profile, **kwargs)
    
    # Use 'x' if available, otherwise use 'i'
    x_values = df['x'] if 'x' in df.columns else df['i']
    
    for feature in plot_features:
        y_values = df[feature]
        ax.scatter(x_values, y_values, label=feature, s=marker_size, 
                    marker=marker_type)
        
    xlim = kwargs.get('xlim', (None, None))
    
    # If both subplot_location and subplot_a_range are specified, add a subplot
    if (subplot_location and subplot_x_range) and xlim[0] == 0.0:
        # Create subplot axes
        ax_sub = fig.add_axes(subplot_location) # Subplot location and size
        
        min_x, max_x = subplot_x_range
        mask = (x_values >= min_x) & (x_values <= max_x)
        sub_x_values = x_values[mask]
        sub_y_values = y_values[mask]
        ax_sub.scatter(sub_x_values, sub_y_values, s=marker_size, 
                       marker=marker_type)  # Adjust as needed
        _plt.setp(ax_sub.spines.values(), color='k')
        
    # Set labels and title for the main plot
    title = f"Scatter Plot for LET-{column_type}"
    if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
        title += f"\nCut: {kwargs['cut_in_um']} µm, " \
                 f"Voxel: {kwargs['voxel_in_um']} µm"
    
    ax.set_title(title, fontsize=kwargs.get('title_fontsize', None))
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
        
    ax.set_xlabel('x [mm]', fontsize = kwargs.get('xlabel_fontsize', None))
    ax.set_ylabel(f"LET$_{column_type[0].lower()} $ [keV $\mu$m$^{{-1}}$]", 
                    fontsize = kwargs.get('ylabel_fontsize', None))
    
    ax.set_xlim(xlim)
    
    ax.grid('both', color='k', linestyle=':')
    _plt.setp(ax.spines.values(), color='k')
    
    # Combine legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    if dose_profile is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
    else:
        handles = handles1
        labels = labels1
    
    ax.legend(handles, labels, bbox_to_anchor=(1.10, 1.0), loc='upper left',
              fontsize=kwargs.get('legend_fontsize', None))
    
    # Set the fontsize for the ticks
    _plt.setp(ax.get_xticklabels(), fontsize=kwargs.get('xticks_fontsize', None))
    _plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
    
    if dose_profile is not None:
        _plt.setp(ax2.spines.values(), color='k')
        _plt.setp(ax2.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
    
    _plt.show()
    
    return fig
    
def plot_feature_distribution(
    df: _pd.DataFrame, 
    feature: str, 
    marker_size: int = 8, 
    marker_type: str = '.',
    dose_profile: _pd.DataFrame = None,
    subplot_location: list = None, # Default None for optional subplot
    subplot_x_range: tuple = None, # Default None for optional subplot x-range
    **kwargs
) -> _plt.Figure:
    """
    Create a scatter plot to visualize the distribution of a specific feature in
    the DataFrame. 

    Parameters:
    - df (DataFrame): The input DataFrame.
    - feature (str): The name of the feature column to plot.
    - marker_size (int, optional): Size of markers in the scatter plot. Defaults
      to 8.  
    - marker_type (str, optional): Marker type in the scatter plot. Defaults to
      '.'.  
    - dose_profile (DataFrame, optional): DataFrame containing the dose profile.
    - subplot_location (list, optional): Location and size of the subplot.
      Defaults to None.
    - subplot_x_range (list, optional): Range of x data for the subplot. If
      provided, the subplot will display data within this range. Defauts to
      None. 
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title of the plot.

    Returns:
    - plt.Figure: The created matplotlib figure.
    """
    _sns.set_style("whitegrid")
    
    # Check if the specified feature is in the DataFrame
    if feature not in df.columns:
        raise ValueError(f"The specified feature '{feature}' is not a column "
                         "in the DataFrame.")
        
    xlim = kwargs.get('xlim', (None, None))
        
    # Create a scatter plot with subplots if subplot_location is provided
    fig, ax = _plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
    
    # Plot dose distribution if dose_profile is provided
    if dose_profile is not None:
        ax2 = plot_twin_profile(ax, dose_profile, **kwargs)
    
    # Use 'x' if available, otherwise use 'i'
    x_values = df['x'] if 'x' in df.columns else df['i']
    
    y_values = df[feature]
                
    ax.scatter(x_values, y_values, label=feature, s=marker_size, marker=marker_type)
    
    # If both subplot_location and subplot_x_range are specified, add a subplot
    if (subplot_location and subplot_x_range) and xlim[0] == 0.0:
        # Create axes for the subplot
        ax_sub = fig.add_axes(subplot_location)
        
        min_x, max_x = subplot_x_range
        mask = (x_values >= min_x) & (x_values <= max_x)
        sub_x_values = x_values[mask]
        sub_y_values = y_values[mask]
        ax_sub.scatter(sub_x_values, sub_y_values, s=marker_size, 
                       marker=marker_type)  # Adjust as needed
        _plt.setp(ax_sub.spines.values(), color='k')
        
    # Set labels and title
    title = f"Distribution of '{feature}'"
    if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
        title += f"\nCut: {kwargs['cut_in_um']} µm, " \
                 f"Voxel: {kwargs['voxel_in_um']} µm"
    ax.set_title(title, fontsize=kwargs.get('title_fontsize', None))
    
    ax.set_xlabel('x [mm]', fontsize = kwargs.get('xlabel_fontsize', None))
    ax.set_ylabel(f"{feature} [keV $\mu$m$^{{-1}}$]", 
                    fontsize = kwargs.get('ylabel_fontsize', None))  
    
    ax.set_xlim(xlim)
    
    ax.grid('both', color='k', linestyle=':')
    _plt.setp(ax.spines.values(), color='k')
    
    # Combine legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    if dose_profile is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
    else:
        handles = handles1
        labels = labels1
    
    ax.legend(handles, labels, bbox_to_anchor=(1.10, 1.0), loc='upper left',
              fontsize=kwargs.get('legend_fontsize', None))
    
    # Set the fontsize for the ticks
    _plt.setp(ax.get_xticklabels(), fontsize=kwargs.get('xticks_fontsize', None))
    _plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
    
    if dose_profile is not None:
        _plt.setp(ax2.spines.values(), color='k')
        _plt.setp(ax2.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
    
    _plt.show()
    
    return fig
    
def plot_more_features(
    df: _pd.DataFrame, 
    feature_list: List[str], 
    ylabel: str,
    marker_size: int = 8, 
    marker_types: Union[str, List[str]] = '.',
    dose_profile: Optional[_pd.DataFrame] = None,
    let_total_profile: Optional[_pd.DataFrame] = None,
    n_feat_per_plot: Optional[int] = None,
    **kwargs
) -> list:
    """
    Create a scatter plot to visualize the distribution of a specific feature in
    the DataFrame. 

    Parameters:
    - df (DataFrame): The input DataFrame.
    - feature_list (list): The list of features to create scatter plot on.
    - ylabel (str): The y-label of the axis on which the features are plotted.
    - marker_size (int, optional): Size of markers in the scatter plot. Defaults
      to 8.  
    - marker_types (str or list of str, optional): Marker(s) type in the scatter
      plot. Defaults to '.'.  
    - dose_profile (DataFrame, optional): DataFrame containing the dose profile.
    - n_feat_per_plot (int, optional): Number of features per plot. Defaults to
      None. 
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title of the plot.

    Returns:
    - list: A list of created matplotlib figures.
    """
    _sns.set_style("whitegrid")
    
    # Create a color map from the gist_rainbow colormap
    cmap = cm.get_cmap('nipy_spectral', len(feature_list))  # Number of unique colors
    
    # Generate colors from the colormap
    colors = [cmap(i) for i in range(len(feature_list))]
    
    # Ensure marker_types is a list
    if isinstance(marker_types, str):
        marker_types = [marker_types]
       
    # Check if the specified feature is in the DataFrame
    for feature in feature_list:
        if feature not in df.columns:
            raise ValueError(f"The specified feature '{feature}' is not a column "
                            "in the DataFrame.")
    
    # Determine the number of features per plot       
    if n_feat_per_plot is None:
        n_feat_per_plot = len(feature_list)
        
    # Determine the LET column-type from the feature list
    let_col = 'LTT' if feature_list[0].endswith('_T') else 'LDT'
    
    # Determine the number of figures
    n_fig = int(_np.ceil(len(feature_list) / n_feat_per_plot))
    
    # Initialize list to hold figures
    figures = []
    
    for i in range(n_fig):
        fig, ax = _plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
            
        # Offset the right spine of twin_stacked. The ticks and label have already
        # been placed of the right by twinx above.   
        if let_total_profile is None:
            twin_profile = df[['x', let_col]].copy()
        else:
            twin_profile = let_total_profile
            
        twin_stacked_ax = plot_twin_profile(
            ax, twin_profile, twin_color='b', twin_offset=1.15, 
            twin_alpha=kwargs.get('twin_alpha', 0.5), twin_linewidth = 2,
            twin_linestyle = 'solid', 
            twin_plotlabel = 'Total', twin_ylabel = "Let total [keV $\mu$m$^{-1}$]",
            ylabel_fontsize = kwargs.get('ylabel_fontsize', None),
            yticks_fontsize = kwargs.get('yticks_fontsize', None)
        )
            
        # Plot dose distribution if dose_profile is provided
        if dose_profile is not None:
            ax2 = plot_twin_profile(ax, dose_profile, **kwargs)
            ax2.set_yscale(kwargs.get('yscale','linear'))
        
        # Plot selected features
        # Use 'x' if available, otherwise use 'i'
        x_values = df['x'] if 'x' in df.columns else df['i']
        
        # Set color cycle for the scatter plots
        color_cycle = cycler(color=colors[i*n_feat_per_plot:(i+1)*n_feat_per_plot])
        ax.set_prop_cycle(color_cycle)
        
        for j, feature in enumerate(feature_list[i*n_feat_per_plot:
            (i+1)*n_feat_per_plot]):
            y_values = df[feature]
            marker = marker_types[j % len(marker_types)]
            ax.scatter(x_values, y_values, label=feature, s=marker_size, 
                       marker=marker, alpha=0.8)
            
        # Set y-scale
        ax.set_yscale(kwargs.get('yscale','linear'))
        twin_stacked_ax.set_yscale(kwargs.get('yscale','linear'))
            
        # Set labels and title
        if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
            title = f"\nCut: {kwargs['cut_in_um']} µm, " \
                    f"Voxel: {kwargs['voxel_in_um']} µm"
            ax.set_title(title, fontsize=kwargs.get('title_fontsize', None))
        
        ax.set_xlabel('x [mm]', fontsize = kwargs.get('xlabel_fontsize', None))
        ax.set_ylabel(ylabel, fontsize = kwargs.get('ylabel_fontsize', None))  
        
        ax.set_xlim(kwargs.get('xlim', (None, None)))
        
        ax.grid('both', color='k', linestyle=':')
        _plt.setp(ax.spines.values(), color='k')
        
        # Combine legends from both axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles_s, labels_s = twin_stacked_ax.get_legend_handles_labels()
        
        if dose_profile is not None:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2 + handles_s
            labels = labels1 + labels2 + labels_s
        else:
            handles = handles1 + handles_s
            labels = labels1 + labels_s
        
        ax.legend(handles, labels, bbox_to_anchor=(0.52, -0.11), 
                loc='upper center', ncols = 4, numpoints = 1,
                scatterpoints = 3, markerscale = 1.5, scatteryoffsets = [0.5],
                fontsize=kwargs.get('legend_fontsize', None))
        
        # Set the fontsize for the ticks
        _plt.setp(ax.get_xticklabels(), fontsize=kwargs.get('xticks_fontsize', None))
        _plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
        _plt.setp(ax.yaxis.get_offset_text(), fontsize=kwargs.get('yticks_fontsize', None))
        _plt.setp(twin_stacked_ax.get_yticklabels(), 
                fontsize=kwargs.get('yticks_fontsize', None))
        
        if dose_profile is not None:
            _plt.setp(ax2.spines.values(), color='k')
            _plt.setp(ax2.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
            
        figures.append(fig)
    
    _plt.show()
    
    return figures

def plot_more_dataframe(
    df_list: List[_pd.DataFrame], 
    feature_list: List[str],
    ylabel: str=None,
    marker_size: int=8,
    marker_types: Union[str, List[str]] = '.',
    dose_profiles: Optional[List[_pd.DataFrame]] = None,
    let_totals: Optional[List[_pd.DataFrame]] = None,
    plot_suffix: Optional[List[str]] = None,
    n_feat_per_plot: Optional[int] = None,  
    **kwargs  
) -> list:
    
    _sns.set_style("whitegrid")
    
    # Determine the LET column-type from the feature list
    let_col = 'LTT' if feature_list[0].endswith('_T') else 'LDT'
    
    p = inflect.engine()
    
    # Check if df_list, and dose_profiles (if present) have the same number of
    # elements  
    if dose_profiles and (len(df_list) != len(dose_profiles)):
        raise ValueError("Length of dose_profiles must match the length of df_list")
        
    # Determine whether to plot dose profiles based on the presence of dose data
    plot_dose_profiles = False
    if all(dose is not None for dose in dose_profiles):
        plot_dose_profiles = True
    
    # If let_totals is not provided, create a list of DataFrames with 'x' and
    # LET column data   
    if let_totals is None:
        let_totals = []
        for j, df in enumerate(df_list):
            let_totals.append(df[['x', let_col]].copy())

    # Ensure let_totals and df_list have the same length 
    if len(df_list) != len(let_totals):
        raise ValueError("Length of let_totals must match the length of df_list")
    
    # Convert marker_types to a list if it is a single string
    if isinstance(marker_types, str):
        marker_types = [marker_types] * len(df_list)
    
    # Ensure the length of marker_types matches the number of DataFrames
    if len(marker_types) != len(df_list):
        raise ValueError("Length of marker_types must match the length of df_list")
    
    # Ensure the length of plot_suffix, if provided, matches the number of
    # DataFrames 
    if plot_suffix:
        if isinstance(plot_suffix, str):
            plot_suffix = [plot_suffix]
        if len(plot_suffix) != len(df_list):
            raise ValueError("Length of plot_suffix must match length of df_list")
        
    # Check if the specified feature is in the each DataFrame in the input list
    for i, df in enumerate(df_list):
        for feature in feature_list:
            if feature not in df.columns:
                raise ValueError(f"The specified feature '{feature}' is not a "
                                 f"column in the {p.ordinal(i+1)} DataFrame.")
    
    # Determine the number of features per plot       
    if n_feat_per_plot is None:
        n_feat_per_plot = len(feature_list)
           
    # Initialize list to hold figures
    figures = []
    
    # Loop through the features and create plots
    for i in range(0, len(feature_list), n_feat_per_plot):
        fig, ax = _plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
        feature_subset = feature_list[i:i+n_feat_per_plot]
          
        twin_stacked_ax = plot_twin_profile(
            ax, let_totals, twin_color='mediumvioletred', twin_offset=1.15, 
            twin_alpha=kwargs.get('twin_alpha', 0.5), twin_linewidth = 2,
            twin_linestyle = 'solid', 
            twin_plotlabel = f"{let_col}",
            twin_ylabel = "Let total [keV $\mu$m$^{-1}$]",
            twin_plotsuffix = plot_suffix,
            ylabel_fontsize = kwargs.get('ylabel_fontsize', None),
            yticks_fontsize = kwargs.get('yticks_fontsize', None)
        )
        
        # Plot dose distribution if dose_profile is provided
        if plot_dose_profiles:           
            ax2 = plot_twin_profile(
                    ax, dose_profiles, 
                    twin_plotsuffix=plot_suffix, 
                    **kwargs
                )
            ax2.set_yscale(kwargs.get('yscale','linear')) 
        
        # Plot feature_subset
        for j, df in enumerate(df_list):
            # Use 'x' if available, otherwise use 'i'
            x_values = df['x'] if 'x' in df.columns else df['i']
            
            lbl = plot_suffix[j] if plot_suffix else f'DataFrame {j}'
                       
            for feature in feature_subset:
                y_values = df[feature]
                ax.scatter(
                    x_values, 
                    y_values,
                    marker=marker_types[j],
                    s=marker_size,
                    alpha=0.5,
                    label=f'{feature} - {lbl}'
                )
                
        # Set y-scale
        ax.set_yscale(kwargs.get('yscale','linear'))
        twin_stacked_ax.set_yscale(kwargs.get('yscale','linear'))
                
        # Set labels and title
        if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
            title = f"\nCut: {kwargs['cut_in_um']} µm, " \
                    f"Voxel: {kwargs['voxel_in_um']} µm"
            ax.set_title(title, fontsize=kwargs.get('title_fontsize', None))
            
        ax.set_xlabel('x [mm]', fontsize = kwargs.get('xlabel_fontsize', None))
        ax.set_ylabel(ylabel, fontsize = kwargs.get('ylabel_fontsize', None))  
        
        ax.set_xlim(kwargs.get('xlim', (None, None)))
        
        ax.grid('both', color='k', linestyle=':')
        _plt.setp(ax.spines.values(), color='k')
        
        # Combine legends from both axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles_s, labels_s = twin_stacked_ax.get_legend_handles_labels()
        
        if plot_dose_profiles:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2 + handles_s
            labels = labels1 + labels2 + labels_s
        else:
            handles = handles1 + handles_s
            labels = labels1 + labels_s
        
        ax.legend(handles, labels, bbox_to_anchor=(0.52, -0.11), 
                loc='upper center', ncols = 3, numpoints = 1,
                scatterpoints = 3, markerscale = 1.5, scatteryoffsets = [0.5],
                fontsize=kwargs.get('legend_fontsize', None))
        
        # Set the fontsize for the ticks
        _plt.setp(ax.get_xticklabels(), fontsize=kwargs.get('xticks_fontsize', None))
        _plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
        _plt.setp(ax.yaxis.get_offset_text(), fontsize=kwargs.get('yticks_fontsize', None))
        _plt.setp(twin_stacked_ax.get_yticklabels(), 
                fontsize=kwargs.get('yticks_fontsize', None))
        
        if plot_dose_profiles:
            _plt.setp(ax2.spines.values(), color='k')
            _plt.setp(ax2.get_yticklabels(), fontsize=kwargs.get('yticks_fontsize', None))
            
        figures.append(fig)
        
    _plt.show()
    
    return figures
        
def plot_dataframe_with_outliers(
    df: _pd.DataFrame, 
    x_col: str, 
    y_col: str,
    outlier_col: str = None,
    xlim: Tuple[Union[float, None], Union[float, None]] = (None, None),
    **kwargs
    ):
    """
    Plot a line plot of the specified columns from a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - x_col (str): The name of the column to use for the x-axis.
    - y_col (str): The name of the column to use for the y-axis.
    - outlier_col (str, optional): The name of the column containing outliers.
      Defaults to None.
    - xlim (Tuple[Union[float, None], Union[float, None]], optional): 
      The limits for the x-axis as a tuple (xmin, xmax).
      Each element can be a float or None. Use None for automatic scaling.
      Defaults to (None, None).
    - **kwargs: Additional keyword arguments for customizing the plot.
      These can include figsize, xlabel, ylabel, title, etc.
    """
    _plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    _plt.plot(df[x_col], df[y_col], label=f'{y_col}', alpha=0.5)
    
    if outlier_col:
        non_outlier = df[df[outlier_col] != -1]
        _plt.scatter(non_outlier['x'], non_outlier['LTT'], color='green', 
                    marker='o', label='Non-Outlier Points', s=8)
        outliers = df[df[outlier_col] == -1]
        _plt.scatter(outliers[x_col], outliers[y_col], color='red', marker='x', 
                    label='Outliers', s=8)
    
    _plt.xlabel(kwargs.get('xlabel', x_col))
    _plt.ylabel(kwargs.get('ylabel', y_col))
    _plt.title(kwargs.get('title', f'Plot of {y_col} against {x_col}'))
    _plt.legend()
    
    # Set x limits
    _plt.xlim(xlim)
    
    _plt.show()

def plot_dataframe_multiple_y(
    df: _pd.DataFrame, 
    x_col: str, 
    y_cols: List[str],
    plot_type: str = 'line',
    xlim: Tuple[Union[float, None], Union[float, None]] = (None, None),
    **kwargs):
    """
    Plot line or scatter plots of the specified columns from a DataFrame with
    multiple y columns. 

    Parameters:
    - df (DataFrame): The input DataFrame.
    - x_col (str): The name of the column to use for the x-axis.
    - y_cols (List[str]): The names of the columns to use for the y-axis.
    - plot_type (str, optional): Type of plot to create. Choose between 'line'
      and 'scatter'. Defaults to 'line'.
    - xlim (Tuple[Union[float, None], Union[float, None]], optional): 
      The limits for the x-axis as a tuple (xmin, xmax).
      Each element can be a float or None. Use None for automatic scaling.
      Defaults to (None, None).
    - **kwargs: Additional keyword arguments for customizing the plot.
      These can include figsize, xlabel, ylabel, title, etc.
    """
    # Check if x_col and all y_cols are in DataFrame columns
    if x_col not in df.columns:
        raise ValueError(f"The specified x column '{x_col}' is not a column in " 
                         "the DataFrame.")
        
    valid_y_cols = [col for col in y_cols if col in df.columns]
    invalid_y_cols = [col for col in y_cols if col not in df.columns]
    if invalid_y_cols:
        warning_message = ("The following y columns are not columns in the " +
                           "DataFrame and will be ignored:\n" + 
                           ",\n".join(invalid_y_cols))
        _warnings.warn(warning_message)

    _plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    
    if plot_type == 'line':
        for y_col in y_cols:
            _plt.plot(df[x_col], df[y_col], label=y_col, alpha=0.5)
    elif plot_type == 'scatter':
        for y_col in valid_y_cols:
            _plt.scatter(df[x_col], df[y_col], label=y_col, s=kwargs.get('marker_size', 8))
    else:
        raise ValueError("Invalid plot_type. Choose between 'line' and 'scatter'.")
    
    _plt.xlabel(kwargs.get('xlabel', x_col))
    _plt.ylabel(kwargs.get('ylabel', ', '.join(y_cols)))
    _plt.title(kwargs.get('title', f'Plot of {", ".join(y_cols)} against {x_col}'))
    _plt.legend()
    
    # Set x limits
    _plt.xlim(xlim)
    
    _plt.show()

def create_correlation_plot_title(df, title_prefix, **kwargs):
    """
    Create the title for a correlation plot based on the 'x' or 'i' values in
    the DataFrame. 

    Parameters:
    - df (DataFrame): The input DataFrame.
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title.

    Returns:
    - str: The title for the correlation heatmap.
    """
    # Use 'x' if available, otherwise use 'i'
    x_values = df['x'].values if 'x' in df.columns else df['i'].values
    first_x = x_values[0]
    last_x = x_values[-1]

    template = '{prefix} [{first} $\leq$ x [mm] $\leq$ {last}]'
    heatmap_title = template.format(prefix=title_prefix, 
                                    first=first_x, 
                                    last=round(last_x, 3))
    
    if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
        heatmap_title += f"\nCut: {kwargs['cut_in_um']} µm, " \
                         f"Voxel: {kwargs['voxel_in_um']} µm"
    
    return heatmap_title

def plot_correlation_df(
  corr_dict: _pd.DataFrame, 
  figsize=(4, 8), 
  plot_title='Correlation Heatmap'
) -> _plt.Figure:
    """
    Plot a correlation heatmap based on a correlation dictionary.

    Parameters:
    - corr_dict (DataFrame): A DataFrame containing correlation coefficients.
    - figsize (tuple, optional): Figure size (width, height). Defaults to (4, 8).
    - plot_title (str, optional): Title of the plot. Defaults to 'Correlation
      Heatmap'. 

    Returns:
    - None
    """
    
    with _sns.axes_style("whitegrid"):
        # Set Seaborn style for the plot
        # _sns.set_theme(style="whitegrid")
    
        # Create a figure with the specified size
        fig, ax = _plt.subplots(figsize=figsize)
    
        # Create a heatmap using Seaborn
        _sns.heatmap(corr_dict.dropna(), annot=True, cmap="coolwarm", 
                    fmt=".2f", vmin=-1, vmax=1)
        
        # Set title
        ax.set_title(plot_title)
        
        _plt.show()
        
    return fig
    
# ----------------------------------------------------------------
# --- Companion method used in 'barplot_comparison' and
# 'combined_plot_comparison'
# ----------------------------------------------------------------
def prepare_and_sort_data(
    left_data: Union[Dict[str, _np.ndarray], _pd.Series, _pd.DataFrame],
    right_data: Union[Dict[str, _np.ndarray], _pd.Series, _pd.DataFrame],
    sorting_plot: str = 'left',
    sort_order: str = 'ascending'
) -> Tuple[_pd.DataFrame, _pd.DataFrame]: 
    """
    Prepare and sort the input data for plotting.

    Parameters:
    - left_data: Data for the left plot.
    - right_data: Data for the right plot.
    - sorting_plot (str, optional): Data to consider for plot ordering, 'left'
      or 'right'. Defaults to 'left'. 
    - sort_order (str, optional): Sort order for plotting, 'ascending' or
      'descending'. Defaults to 'ascending'. 

    Returns:
    - left_data_sorted: Sorted left data.
    - right_data_sorted: Sorted right data.
    """
    # Convert data to DataFrame if not already DataFrame
    if isinstance(left_data, dict):
        left_data = _pd.DataFrame(left_data)
    elif isinstance(left_data, _pd.Series):
        left_data = left_data.to_frame()

    if isinstance(right_data, dict):
        right_data = _pd.DataFrame(right_data)
    elif isinstance(right_data, _pd.Series):
        right_data = right_data.to_frame()

    # Check if input DataFrames have a single column and N rows
    if left_data.shape[1] != 1 or right_data.shape[1] != 1:
        raise ValueError("Input DataFrames must have a single column.")

    # Check if the number of rows match
    common_index = left_data.index.intersection(right_data.index)
    if len(common_index) == 0:
        raise ValueError("No common rows between 'left_data' and 'right_data'.")
    elif len(common_index) != left_data.shape[0] or len(common_index) != right_data.shape[0]:
        print("Warning: Number or names of rows in 'left_data' and "
              "'right_data' do not match.")
        # Restrict the plot to common rows
        left_data = left_data.loc[common_index]
        right_data = right_data.loc[common_index]

    # Sort data for plotting
    if sorting_plot == 'left':
        data_to_sort = left_data
    elif sorting_plot == 'right':
        data_to_sort = right_data
    else:
        raise ValueError("Invalid value for 'sorting_plot'. Use 'left' or 'right'.")

    if sort_order == 'ascending':
        data_to_sort_sorted = data_to_sort.sort_values(by=data_to_sort.columns[0], ascending=True)
    elif sort_order == 'descending':
        data_to_sort_sorted = data_to_sort.sort_values(by=data_to_sort.columns[0], ascending=False)
    else:
        raise ValueError("Invalid value for 'sort_order'. Use 'ascending' or 'descending'.")

    if sorting_plot == 'left':
        left_data_sorted = data_to_sort_sorted
        right_data_sorted = right_data.loc[left_data_sorted.index]
    else:
        right_data_sorted = data_to_sort_sorted
        left_data_sorted = left_data.loc[right_data_sorted.index]

    return left_data_sorted, right_data_sorted

        
def barplot_comparison(
    left_data: Union[_pd.DataFrame, _pd.Series, Dict[str, _np.ndarray]],
    right_data: Union[_pd.DataFrame, _pd.Series, Dict[str, _np.ndarray]],
    left_title: Union[str, None] = None,
    right_title: Union[str, None] = None,
    sorting_plot: str = 'left',
    sort_order: str = 'ascending',
    figsize: tuple = (12, 6)
):
    """
    Create a bar plot comparing two sets of data.

    Parameters:
    - left_data: Data for the left plot. It can be a _pd.Series, _pd.DataFrame,
      or dictionary, each with a single column and N rows.
    - right_data: Data for the right plot. It can be a _pd.Series,
      _pd.DataFrame, or dictionary, each with a single column and N rows.
    - left_title (str, optional): Title for the left plot. Defaults to None.
    - right_title (str, optional): Title for the right plot. Defaults to None.
    - sorting_plot (str, optional): Data to consider for plot ordering, 'left'
      or 'right'. Defaults to 'left'.
    - sort_order (str, optional): Sort order for plotting, 'ascending' or
      'descending'. Defaults to 'ascending'.
    - figsize (tuple, optional): Figure size (width, height). Defaults to (12, 6).

    Returns:
    - None: Displays the bar plot.
    """
    left_data_sorted, right_data_sorted = prepare_and_sort_data(
        left_data=left_data,
        right_data=right_data,
        sorting_plot=sorting_plot,
        sort_order=sort_order
    )
        
    # Create a figure with two subplots
    _, axes = _plt.subplots(1, 2, figsize=figsize, sharey=True,
                             gridspec_kw={'width_ratios': [1, 1]})

    # Plot the left side histogram
    left_plot = _sns.barplot(
        x=left_data_sorted.abs().iloc[:, 0],
        y=left_data_sorted.index, 
        hue=left_data_sorted.iloc[:, 0],
        ax=axes[0], 
        palette='dark:#d65f5f',
    )
    left_plot.legend(loc='center right', bbox_to_anchor=(-0.02, 0.5),
                     ncol=1, mode='expanded', borderaxespad=0.)
    
    # Invert the x-axis direction of the left plot
    axes[0].invert_xaxis()

    # Plot the right side histogram
    right_plot = _sns.barplot(
        x=right_data_sorted.abs().iloc[:, 0],
        y=right_data_sorted.index, 
        hue=right_data_sorted.iloc[:, 0],
        ax=axes[1], 
        palette='dark:#4878d0',
    )
    right_plot.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     ncol=1, mode='expanded', borderaxespad=0.)

    # Set titles for subplots
    if left_title:
        axes[0].set_title(left_title)
    if right_title:
        axes[1].set_title(right_title)

    # Set labels and title
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('Absolute Values')
    axes[1].set_xlabel('Absolute Values')
    
    # Remove y-axis ticks and labels for the left plot
    axes[0].yaxis.tick_right()

    # Adjust layout to place y-axis labels in between the two subplots
    _plt.subplots_adjust(wspace=.29)

    # Show the plot
    _plt.show()  

def combined_plot_comparison(
    left_data: Union[_pd.DataFrame, _pd.Series, Dict[str, _np.ndarray]],
    right_data: Union[_pd.DataFrame, _pd.Series, Dict[str, _np.ndarray]],
    abs_value: bool = True,
    left_ylabel: Union[str, None] = None,
    right_ylabel: Union[str, None] = None,
    xlabel: Union[str, None] = None,
    sorting_plot: str = 'left',
    sort_order: str = 'ascending',
    figsize: tuple = (12, 6),
    **kwargs
) -> _plt.Figure:
    """
    Create a combined plot comparing two sets of data.

    Parameters:
    - left_data: Data for the left plot. It can be a _pd.Series, _pd.DataFrame,
      or dictionary, each with a single column and N rows.
    - right_data: Data for the right plot. It can be a _pd.Series,
      _pd.DataFrame, or dictionary, each with a single column and N rows.
    - abs_value (bool, optional): Whether to plot the absolute value of the data.
      Defaults to True.
    - left_ylabel (str, optional): ylabel for the left plot. Defaults to None.
    - right_ylabel (str, optional): ylabel for the right plot. Defaults to None.
    - xlabel (str, optional): xlabel for the plot. Defauls to None.
    - sorting_plot (str, optional): Data to consider for plot ordering, 'left'
      or 'right'. Defaults to 'left'.
    - sort_order (str, optional): Sort order for plotting, 'ascending' or
      'descending'. Defaults to 'ascending'.
    - figsize (tuple, optional): Figure size (width, height). Defaults to (12,
      6). 
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title.

    Returns:
    -  plt.Figure: The created matplotlib figure.
    """
    left_data_sorted, right_data_sorted = prepare_and_sort_data(
        left_data=left_data,
        right_data=right_data,
        sorting_plot=sorting_plot,
        sort_order=sort_order
    )
    
    if abs_value:
        left_data_sorted = left_data_sorted.abs()
        right_data_sorted = right_data_sorted.abs()
        
    # Create a figure
    fig, ax1 = _plt.subplots(figsize=figsize)

    # Plot the left data on the first axis
    left_color = 'tab:red'
    ax1.plot(
        left_data_sorted.index,
        left_data_sorted.iloc[:, 0], 
        marker='s',
        linestyle='-',
        color=left_color,
    )
    ax1.tick_params(axis='y', labelcolor=left_color)
    ax1.tick_params(axis='x', rotation=90)  # Rotate xtick labels
    ax1.grid(True, which='major', zorder=1)
    ax1.set_ylim(kwargs.get('left_ylim',(None, None)))
    
    # Customize the outer box (spines) of ax1
    ax1.spines['top'].set_visible(True)
    ax1.spines['top'].set_color('k')
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('k')
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_color('k')
    ax1.spines['bottom'].set_linewidth(2)

    # Create a twin axis sharing the same x-axis
    ax2 = ax1.twinx()
    
    # Plot the right data on the second axis
    right_color = 'tab:blue'
    ax2.plot(
        right_data_sorted.index,
        right_data_sorted.iloc[:, 0], 
        marker='o',
        linestyle='-',
        color=right_color,
    )
    ax2.tick_params(axis='y', labelcolor=right_color)
    ax2.grid(True, which='major', zorder=2)
    ax2.set_ylim(kwargs.get('right_ylim',(None, None)))
    
    # Customize the outer box (spines) of ax2
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('k')
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Set titles for subplots
    if left_ylabel:
        ax1.set_ylabel(left_ylabel, color=left_color)
    if right_ylabel:
        ax2.set_ylabel(right_ylabel, color=right_color)
    if xlabel:
        ax1.set_xlabel(xlabel)
        
    # Add title using kwargs if provided
    if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
        title = f"Cut: {kwargs['cut_in_um']} µm, " \
            f"Voxel: {kwargs['voxel_in_um']} µm"
        ax1.set_title(title)
        
    

    # Show the plot
    _plt.show()
    
    return fig

# ----------------------------------------------------------------
# ----------------------------------------------------------------

def visualize_zero_values(
    df: _pd.DataFrame,
    column_as_index: Union[str, int] = None,
    figsize: tuple = (12, 6),
    cmap: str = 'rocket',
    invert_cmap: bool = False,
    primary_x_stop: Optional[int] = None,
    **kwargs
) -> _plt.Figure:
    """
    Visualize missing values per feature in a DataFrame using a heatmap.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_as_index: (str or int, optional) Name or index of the column to use
      as row indices. Defaults to None, meaning the default index will be used. 
    - figsize (tuple, optional): Figure size (width, height). Defaults to (12,
      6).
    - cmap (str or Colormap, optional): Color map for the heatmap. Defaults to
      'rocket'. 
    - invert_cmap (bool, optional): Whether to invert the colormap. Defaults to
      False.
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title.

    Returns:
    - plt.Figure: The created matplotlib figure.
    """
    fig, ax = _plt.subplots(figsize=figsize)
    yticksteps = False
    df_copy = df.copy()  # Make a copy to keep the original DataFrame intact
    
    if column_as_index is not None:
        if isinstance(column_as_index, int):
            df_copy.set_index(df_copy.iloc[:, column_as_index], inplace=True) # Set specified column as index
            df_copy.index.name = None # Remove index name for better visualization
            df_copy.drop(df_copy.columns[column_as_index], axis=1, inplace=True) # Drop the column used as index
        elif isinstance(column_as_index, str):
            df_copy.set_index(column_as_index, inplace=True) # Set specified column as index
            df_copy.index.name = None # Remove index name for better visualization
        
        if len(df_copy) <= 5:
            yticksteps = df_copy.index.to_list()
        else:
            yticksteps = len(df_copy) // 10
    
    # If invert_cmap is True, invert the colormap
    cmap = cmap + '_r' if invert_cmap else cmap
    cmap = _plt.cm.get_cmap(cmap)
    
    # Get two colors from the colormap
    color1 = cmap.colors[0]
    color2 = cmap.colors[-1]
    
    # Create a custom colormap with the specified colors
    cmap = ListedColormap([color1, color2])
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', [color1, color2])
    
    # Define the boundaries for the colorbar
    bounds = [0, 0.5, 1]
    
    # Create a BoundaryNorm instance to  map the colors to the levels
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Visualize zero values
    _sns.heatmap(df_copy == 0, yticklabels=yticksteps, cbar=True, cmap=cmap, 
                 norm=norm)
    
    # Set title and axes labels
    title = 'Zero Values Distributions'
    if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
        title += f"\nCut: {kwargs['cut_in_um']} µm, " \
            f"Voxel: {kwargs['voxel_in_um']} µm"       
    ax.set_title(title)
    ax.set_xlabel('Features')
    if column_as_index is not None:
        ax.set_ylabel(f"{column_as_index}")
        
    # Customize tick labels of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Non-zero', 'Zero'])
    
    # Superimpose a horizontal red line at primary_x_stop if provided
    if primary_x_stop is not None:
        ax.axhline(y=primary_x_stop, color='red', linewidth=2)
    
    _plt.show()
    
    return fig
    
    
def plot_let_outliers_replacements(
    x_data_filtered: _pd.DataFrame,
    let_column_filtered: _pd.DataFrame,
    outliers_and_replacements_df: _pd.DataFrame,
    outlier_scores: List[float],
    radius_scaling: str = 'direct',
    figsize: tuple = (10, 10),
    cbar_label: str = 'Outlier Factor',
    **kwargs
):
    """
    Plot LET profiles with outliers and replacements.

    Parameters:
    - x_data_filtered (pd.DataFrame): Filtered x column DataFrame
    - let_column_filtered (pd.DataFrame): Filtered LET column DataFrame.
    - outliers_and_replacements_df (pd.DataFrame): DataFrame containing both
      outliers and their replacement values.
    - outliers_scores (list): List of outliers scores for bubble plot according
      their significance.
    - radius_scaling_factor (str, optional): Scaling calculation method for the
      radius of outlier circles. Defaults to 'direct'.
    - figsize (tuple, optional): Figure size (width, height). Defaults to (10, 10).
    - cbar_label (str, optional): Label for the colorbar. Defaults to 'Outlier
      Factor'. 
    - **kwargs: Additional keyword arguments. If 'cut_in_um' and 'voxel_in_um' 
      are provided, they are used to add a second line in the title.
    """
    # Check if radius_scaling is a valid option
    if radius_scaling not in ['direct', 'inverted']:
        raise ValueError("Invalid value for radius_scaling. Use 'direct' or "
                         "'inverted'.")
    
    _plt.figure(figsize=figsize)
    _plt.scatter(
        x_data_filtered, 
        let_column_filtered, 
        label='LET Profile', 
        s=3.0, 
        color='k', 
        alpha=0.5
    )
    
    # Extract outliers and replacement values
    outliers = outliers_and_replacements_df['outliers']
    replacements = outliers_and_replacements_df['replacements']
    
    # Highlight outliers:   
    # plot circles with radius proportional to outlier scores
    if radius_scaling == 'direct':
        # outliers values and scores are positive correlated
        radius = outlier_scores / outlier_scores.max()
    elif radius_scaling == 'inverted':    
        # outliers values and scores are negative correlated
        score_range = outlier_scores.max() - outlier_scores.min()
        radius = (outlier_scores.max() - outlier_scores) / score_range

    sc_outliers = _plt.scatter(
        x_data_filtered.loc[outliers.index], 
        outliers, 
        s=1000.0 * radius, 
        c=radius,
        cmap='autumn_r',
        alpha=0.5
    )

    # Plot outliers replacements
    _plt.scatter(
        x_data_filtered.loc[replacements.index], 
        replacements, 
        s=10.0,
        marker='^', 
        color='blue',
        label='Replacements'
    )
    
    # Add colorbar
    cbar = _plt.colorbar(sc_outliers)
    cbar.set_label(cbar_label)

    # Set labels and title
    title = 'LET Profile with Outliers and Replacements'
    if 'cut_in_um' in kwargs and 'voxel_in_um' in kwargs:
        title += f"\nCut: {kwargs['cut_in_um']} µm, " \
            f"Voxel: {kwargs['voxel_in_um']} µm"      
    _plt.xlabel('x [mm]')
    _plt.ylabel('LET [keV $\mu$m$^{{-1}}$]')
    _plt.legend()
    _plt.title(title)
    _plt.show()

# ----------------------------------------------------------------
# ----------------------------------------------------------------

def plot_feature_and_mark_outliers_by_let(
    df: _pd.DataFrame, 
    outliers_and_replacements_df: _pd.DataFrame,
    feature: str, 
    marker_size: int = 8, 
    marker_type: str = '.',
    subplot_location: list = None, # Default None for optional subplot
    subplot_x_range: tuple = None, # Default None for optional subplot x-range
) -> None:
    """
    Create a scatter plot to visualize the distribution of a specific feature in
    the DataFrame and highlight outliers identified on LET-total distribution.

    Parameters:
    - df (DataFrame): The original DataFrame containing the feature.
    - outliers_and_replacements_df (DataFrame): DataFrame containing the
      outliers and replacements. 
    - feature (str): The name of the feature column to plot.
    - marker_size (int, optional): Size of markers in the scatter plot. Defaults
      to 8.  
    - marker_type (str, optional): Marker type in the scatter plot. Defaults to
      '.'.  
    - subplot_location (list, optional): Location and size of the subplot.
      Defaults to None.
    - subplot_x_range (list, optional): Range of x data for the subplot. If
      provided, the subplot will display data within this range. Defauts to
      None. 

    Returns:
    - None: Displays the scatter plot.
    """
    _sns.set_style("whitegrid")
    
    # Check if the specified feature is in the DataFrame
    if feature not in df.columns:
        raise ValueError(f"The specified feature '{feature}' is not a column "
                         "in the DataFrame.")
    
    # Create a scatter plot
    fig, ax = _plt.subplots(figsize=(12, 6))
    if subplot_location:
        ax_sub = fig.add_axes(subplot_location) # Subplot location and size
    
    # Use 'x' if available, otherwise use 'i'
    x_values = df['x'] if 'x' in df.columns else df.index
    
    y_values = df[feature]
                
    ax.scatter(x_values, y_values, label=feature, s=marker_size, 
                 marker=marker_type)
    
    # Highlight outliers
    if not outliers_and_replacements_df.empty:
        outlier_indices = outliers_and_replacements_df.index
        outlier_x_values = x_values[outlier_indices]
        outlier_y_values = y_values[outlier_indices]
        ax.scatter(outlier_x_values, outlier_y_values, color='red', 
                   marker='x', label='Outliers', alpha=0.5)
        
    # If both subplot_location and subplot_a_range are specified, add a subplot
    if subplot_location and subplot_x_range:
        min_x, max_x = subplot_x_range
        mask = (x_values >= min_x) & (x_values <= max_x)
        sub_x_values = x_values[mask]
        sub_y_values = y_values[mask]
        ax_sub.scatter(sub_x_values, sub_y_values, s=marker_size, 
                       marker=marker_type)  # Adjust as needed
        
        if not outliers_and_replacements_df.empty:
            sub_df = df[mask]
            sub_outliers_df = outliers_and_replacements_df[
                outliers_and_replacements_df.index.isin(sub_df.index)
            ]
            sub_outlier_indices = sub_outliers_df.index
            sub_outlier_x_values = x_values[sub_outlier_indices]
            sub_outlier_y_values = y_values[sub_outlier_indices]
            ax_sub.scatter(sub_outlier_x_values, sub_outlier_y_values, color='red',
                        marker='x', alpha=0.5)
        
        bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5)
        _plt.setp(ax_sub.get_xticklabels(), bbox=bbox)
        
    
    # Set labels and title
    ax.set_xlabel('x [mm]')
    ax.set_ylabel(f"{feature} [keV $\mu$m$^{{-1}}$]")  
    ax.set_title(f"Distribution of '{feature}'")
    ax.legend(loc='lower left')
    
    _plt.show()
    
# ----------------------------------------------------------------
# ----------------------------------------------------------------
    
def plot_train_test_val_distribution(
    df_before_split: _pd.DataFrame,
    feature_names: list, 
    X_train: _np.ndarray, 
    X_test: _np.ndarray, 
    X_val: _np.ndarray = None
) -> None:
    """
    Plot the distribution of the selected features in the training, testing, and
    validation datasets. 

    Parameters:
    - df_before_split (pd.Dataframe): DataFrame from which training, test, and
      validation sets are drawn. 
    - feature_names (list): List of selected feature names.
    - X_train (numpy.ndarray): Training dataset.
    - X_test (numpy.ndarray): Testing dataset.
    - X_val (numpy.ndarray, optional): Validation dataset. Defaults to None.

    Returns:
    - None: The function displays the plot.
    """
    
    # Check if feature_names has length 2
    if len(feature_names) != 2:
        raise ValueError("feature_names should contain exactly two feature names.")

    # Check if the specified features exist in the DataFrame
    for feature in feature_names:
        if feature not in df_before_split.columns:
            raise ValueError(f"The feature '{feature}' does not exist in the DataFrame.")

    # Create a figure
    _plt.figure(figsize=(10, 6))

    # Get the indices of the selected features
    indices = [df_before_split.columns.get_loc(feature) for feature in feature_names]

    # Scatter plot for train dataset
    _plt.scatter(X_train[:, indices[0]], X_train[:, indices[1]], label='Train', 
                alpha=0.7, s=12, marker='o')

    # Scatter plot for the test dataset
    _plt.scatter(X_test[:, indices[0]], X_test[:, indices[1]], label='Test', 
                alpha=0.3, s=10, marker='x')

    # If validation dataset is provided, plot it
    if X_val is not None:
        _plt.scatter(X_val[:, indices[0]], X_val[:, indices[1]], label='Validation', 
                    alpha=0.5, s=10, marker='|')

    # Set labels and title
    _plt.xlabel(f'{feature_names[0]}')
    _plt.ylabel(f'{feature_names[1]}')
    if X_val is not None:
        _plt.title('Distribution of Train, Test, and Validation '
                  'Datasets')
    else:
        _plt.title('Distribution of Train, Test Datasets')

    # Add legend
    _plt.legend()

    # Show the plot
    _plt.show()