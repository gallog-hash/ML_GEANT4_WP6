# src/utils/filesystem_utils.py

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def auto_detect_project_root() -> Optional[Path]:
    """
    Auto-detect project root by looking for common markers.
    
    Returns:
        Path to project root if detected, None otherwise
    """
    current = Path.cwd()
    
    # Look for project markers going up the tree
    markers = ["pyproject.toml", "setup.py", "requirements.txt", "src", "data"]
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    return None


def resolve_path_with_project_root(
    path: Union[str, Path], 
    project_root: Optional[Union[str, Path]] = None
) -> Path:
    """
    Resolve a path relative to project root with auto-detection fallback.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        project_root: Optional project root directory for relative path resolution
        
    Returns:
        Path: Resolved absolute path
    """
    path = Path(path)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # Use provided project_root
    if project_root is not None:
        project_root = Path(project_root).resolve()
        return (project_root / path).resolve()
    
    # Auto-detect project root and resolve relative to it
    detected_root = auto_detect_project_root()
    if detected_root is not None:
        resolved_path = (detected_root / path).resolve()
        # Only use auto-detected root if the resolved path exists or makes sense
        if resolved_path.exists() or resolved_path.parent.exists():
            return resolved_path
    
    # Fallback to current working directory
    return path.resolve()


def ensure_directory_exists(directory: Union[str, Path]) -> None:
    """
    Create the directory if it does not already exist.
    
    Args:
        directory (Union[str, Path]): Path of the directory to create.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)


def save_figure(
    figs: Union[Figure, List[Figure]], 
    output_dir: Union[str, Path], 
    filename: str,
    formats: Optional[List[str]] = None,
    close_after_save: bool = True
) -> None:
    """
    Save matplotlib figure(s) in specified formats with support for multiple 
    figures.

    Args:
        figs (Figure or List[Figure]): Figure(s) to save.
        output_dir (Union[str, Path]): Directory where figures will be saved.
        filename (str): Base name for saved files. For multiple figures, 
                       an index will be appended (e.g., "plot_1.png").
        formats (List[str], optional): File formats to save. 
                                     Defaults to ['png', 'eps'].
        close_after_save (bool): Whether to close figures after saving. 
                               Defaults to True.
    
    Raises:
        ValueError: If invalid figures or formats are provided.
        OSError: If unable to create output directory or save files.
    """
    # Validate and prepare parameters
    output_dir = Path(output_dir)
    if formats is None:
        formats = ['png', 'eps']
    
    # Validate formats
    _validate_formats(formats)
    
    # Handle single figure vs list of figures
    if isinstance(figs, Figure):
        _save_single_figure(figs, output_dir, filename, formats, 
                           close_after_save)
    elif isinstance(figs, list):
        _save_figure_list(figs, output_dir, filename, formats, 
                         close_after_save)
    else:
        raise ValueError(
            f"Expected Figure or List[Figure], got {type(figs)}"
        )


def _validate_formats(formats: List[str]) -> None:
    """Validate that all requested formats are supported."""
    supported_formats = {
        'png', 'eps', 'pdf', 'svg', 'jpg', 'jpeg', 'tiff', 'ps'
    }
    
    invalid_formats = [fmt for fmt in formats if fmt.lower() not in 
                      supported_formats]
    
    if invalid_formats:
        raise ValueError(
            f"Unsupported formats: {invalid_formats}. "
            f"Supported formats: {sorted(supported_formats)}"
        )


def _save_single_figure(
    fig: Figure,
    output_dir: Path,
    filename: str,
    formats: List[str],
    close_after_save: bool
) -> None:
    """Save a single figure in all specified formats."""
    if not isinstance(fig, Figure):
        raise ValueError(f"Expected Figure, got {type(fig)}")
    
    saved_files = []
    
    for fmt in formats:
        try:
            filepath = output_dir / f"{filename}.{fmt.lower()}"
            
            # Set format-specific parameters
            save_kwargs = _get_format_kwargs(fmt.lower())
            
            fig.savefig(filepath, **save_kwargs)
            saved_files.append(filepath)
            
        except Exception as e:
            print(f"Warning: Could not save {filename}.{fmt}: {e}")
    
    if saved_files:
        print(f"Saved figure to: {', '.join(str(f) for f in saved_files)}")
    
    if close_after_save:
        plt.close(fig)


def _save_figure_list(
    figs: List[Figure],
    output_dir: Path,
    filename: str,
    formats: List[str],
    close_after_save: bool
) -> None:
    """Save a list of figures with indexed filenames."""
    if not figs:
        print("Warning: Empty figure list provided.")
        return
    
    # Filter valid figures
    valid_figs = [(i, fig) for i, fig in enumerate(figs) 
                  if isinstance(fig, Figure)]
    
    if not valid_figs:
        raise ValueError("No valid figures found in the list")
    
    if len(valid_figs) != len(figs):
        print(f"Warning: {len(figs) - len(valid_figs)} invalid figures "
              "were skipped.")
    
    saved_count = 0
    
    # Save each valid figure
    for original_idx, fig in valid_figs:
        try:
            # Create indexed filename
            if len(valid_figs) > 1:
                indexed_filename = f"{filename}_{original_idx + 1}"
            else:
                indexed_filename = filename
            
            _save_single_figure(
                fig, output_dir, indexed_filename, formats, 
                close_after_save
            )
            saved_count += 1
            
        except Exception as e:
            print(f"Warning: Could not save figure {original_idx + 1}: {e}")
    
    print(f"Successfully saved {saved_count}/{len(valid_figs)} figures")


def _get_format_kwargs(fmt: str) -> dict:
    """Get format-specific save parameters."""
    base_kwargs = {'bbox_inches': 'tight'}
    
    format_specific = {
        'png': {'dpi': 300},
        'eps': {'format': 'eps', 'dpi': 300},
        'pdf': {'format': 'pdf', 'dpi': 300},
        'svg': {'format': 'svg'},
        'jpg': {'dpi': 300, 'quality': 95},
        'jpeg': {'dpi': 300, 'quality': 95},
        'tiff': {'dpi': 300},
        'ps': {'format': 'ps', 'dpi': 300}
    }
    
    kwargs = base_kwargs.copy()
    kwargs.update(format_specific.get(fmt, {}))
    
    return kwargs


def save_plot_batch(
    figures: List[Figure],
    output_dir: Union[str, Path],
    filenames: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Save multiple figures with individual filenames.
    
    Args:
        figures (List[Figure]): List of figures to save.
        output_dir (Union[str, Path]): Directory to save figures.
        filenames (List[str], optional): Individual filenames for each figure.
                                       If None, uses indexed naming.
        **kwargs: Additional arguments passed to save_figure.
    
    Example:
        save_plot_batch(
            [fig1, fig2, fig3], 
            './plots/', 
            ['scatter', 'histogram', 'correlation']
        )
    """
    if filenames is not None:
        if len(filenames) != len(figures):
            raise ValueError(
                f"Number of filenames ({len(filenames)}) must match "
                f"number of figures ({len(figures)})"
            )
        
        # Save each figure with its specific filename
        for fig, filename in zip(figures, filenames):
            save_figure(fig, output_dir, filename, **kwargs)
    else:
        # Use indexed naming
        save_figure(figures, output_dir, "figure", **kwargs)
    

def is_interactive_environment() -> bool:
    """
    Detect if the code is running in an interactive Python environment.
    
    Returns:
        bool: True if running interactively, False otherwise
    """
    # Check for Jupyter notebook/lab
    if 'ipykernel' in sys.modules:
        return True
    
    # Check for IPython (but not Jupyter)
    if 'IPython' in sys.modules:
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return True
        except ImportError:
            pass
    
    # Check if running in interactive mode (python -i or interactive shell)
    if hasattr(sys, 'ps1'):
        return True
    
    # Check for specific IDE environments
    ide_indicators = [
        'PYCHARM_HOSTED',      # PyCharm
        'VSCODE_PID',          # VS Code
        'SPYDER_ARGS',         # Spyder
        'WING_IDE',            # Wing IDE
    ]
    
    if any(indicator in os.environ for indicator in ide_indicators):
        return True
    
    # Note: We don't check for TTY here because running from CLI 
    # should not trigger plot display even in a terminal
    
    return False
        

def display_plot(
    figs: Optional[Union[Figure, List[Figure]]] = None
) -> None:
    """
    Display plots in interactive environments with support for multiple figures.
    Compatible with both single figures and lists of figures from 
    plot_more_features().
    
    Parameters:
        figs (Figure, List[Figure], or None): Figure(s) to display. 
                                            If None, displays current figure.
    """
    if is_interactive_environment():
        if figs is not None:
            if isinstance(figs, list):
                # Handle list of figures (from plot_more_features)
                _display_figure_list(figs)
            elif isinstance(figs, Figure):
                # Handle single figure
                _display_single_figure(figs)
            else:
                # Invalid type, fall back to plt.show()
                print(f"Warning: Expected Figure or List[Figure], "
                    f"got {type(figs)}. Using plt.show()")
                plt.show()
        else:
            # No specific figure provided, show current figure
            plt.show()
            

def _display_figure_list(figs: List[Figure]) -> None:
    """
    Display a list of figures.
    
    Parameters:
        figs (List[Figure]): List of matplotlib figures to display.
    """
    if not figs:
        print("Warning: Empty figure list provided.")
        return
    
    valid_figs = [fig for fig in figs if isinstance(fig, Figure)]
    
    if not valid_figs:
        print("Warning: No valid figures found in the list.")
        return
    
    if len(valid_figs) != len(figs):
        print(f"Warning: {len(figs) - len(valid_figs)} invalid figures "
            "were skipped.")
    
    # Display each figure
    for i, fig in enumerate(valid_figs):
        try:
            # Make this figure current and display it
            plt.figure(fig.number)
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display figure {i + 1}: {e}")


def _display_single_figure(fig: Figure) -> None:
    """
    Display a single figure.
    
    Parameters:
        fig (Figure): Matplotlib figure to display.
    """
    try:
        # Make this figure current and display it
        plt.figure(fig.number)
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display figure: {e}")
        