# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). 

---

## [Unreleased]

### Added
- **Dedicated Data Engineering Workflow:**  
  Refactored `dataeng_main.py` to focus solely on data import, cleaning, and
  outlier handling by removing all exploratory data analysis (EDA) and plotting
  code that was present in the previous `eda_main.py` version. 
- **Primary Particle Naming Consistency:**  
  Added logic to ensure the primary particle identifier ends with `"_1"` (using
  `primary_search_pattern`), avoiding redundant suffixes. 
- **Improved Documentation:**  
  Updated docstrings and inline comments to better explain the functions and
  data engineering workflow. 

### Changed
- **Code Refactoring:**  
  Removed unnecessary plotting functions and parameters related to EDA. The
  script now concentrates on data engineering tasks. 
- **Variable Naming:**  
  Updated variable names (e.g., using `primary_search_pattern` derived from
  `primary_particle`) for consistency and clarity. 
- **Function Parameters:**  
  Adjusted function signatures (especially in `main`) to reflect the dedicated
  focus on data engineering. 

- **data_processing.py: Improve `outlier_detection` function**
  - **Data Type Consistency:**  
    Now, if `let_column_filtered` is provided as a `pd.Series`, it is
    immediately converted to a DataFrame. This ensures that operations such as
    renaming columns work reliably. 
  - **Combined Input Data:**  
    The x-values and LET values are concatenated into a single DataFrame
    (`detection_data`) used by all outlier detection methods, reducing code
    duplication. 
  - **Fallback Mechanism:**  
    If the chosen detection method does not return outlier scores (as in the
    case of the 'upper_limit' or 'dbscan' methods), the function now calculates
    z-scores as a fallback. 
  - **Explicit Error Handling:**  
    An invalid value for `outliers_method` now raises a clear error.
  - **Docstring Updates:**  
    The documentation now accurately describes the input expectations and the
    return types. 

- **dataprocessing.py: Update `replace_outliers` function**
  - Improved type handling for aggregate calculations by checking if non-outlier
    data is a Series or a DataFrame before extracting the median or mean value. 
  - Enhanced error handling by explicitly raising a `ValueError` for unsupported
    `replace_method` values. 
  - Cleaned up the code and added clarifying comments for better readability.

- **dataprocessing.py: Enhance `local_statistic_outlier_replacement` function**
  - Updated the docstring to clearly describe the expected structure of the
    input data and the role of the LET value column. 
  - Modified the logic to flatten the neighbor indices (if returned as a NumPy
    array) so that they can be used consistently for indexing. 
  - Added handling for cases where no neighbors are found by appending `NaN` as
    the replacement value. 
  - Revised comments to reflect the updated behavior in light of changes to
    `search_neighbors`. 

- **dataprocessing.py: Refactor `search_neighbors` function**
  - Optimized the removal of query points from the reference dataset using
    boolean indexing. 
  - Updated comments to clarify that `search_neighbors` now returns a list where
    each element is a NumPy array of neighbor indices (with shape `(1,
    n_neighbors)`) if the reference dataset is a DataFrame. 

- **plot_utils.py: Enhance `plot_let_outliers_replacements` function**
  - Ensure that each call creates a new figure so that a separate window
    is opened for each plot.
  - Use non-blocking display with `_plt.show(block=False)` and
    `_plt.pause(0.001)` to avoid halting execution.
  - Add `_plt.tight_layout()` to improve the layout and prevent clipping of
    titles and labels.
  - Return the figure object for further processing if needed.

- **plot_utils.py: Improve `plot_feature_and_mark_outliers_by_let` function**
  - Each call now creates a new figure to ensure that multiple plots in a loop
    open in separate windows.
  - Uses non-blocking display with `_plt.show(block=False)` and
    `_plt.pause(0.001)` so execution continues without requiring manual window
    closure.
  - Returns the figure object for further processing or saving.
  - Updated data indexing to use `.loc` for proper alignment when selecting
    outlier data.
  - Added minor code cleanups and enhanced documentation.

- **eda_main.py: Enhance extraction of cut and voxel values and run_identifier**
  - Added fallback extraction for `cut_in_um` and `voxel_in_um` from the last
    directory in `data_dir` using regex when extraction from the filename fails. 
  - Converts values from mm to µm when necessary and logs extraction details.
  - Raises a `ValueError` if the cut and voxel values cannot be obtained.

- **dataprocessing.py: Improve `identify_let_outliers` function** 
  - Ensure the input DataFrame contains the required 'x' column.
  - Normalize and validate input parameters (column_type, outliers_method,
    and replace_method).
  - Filter LET columns using a search pattern; warn if multiple columns are
    found and select the first one.
  - Apply x_bounds filtering if provided.
  - Use updated outlier_detection and replace_outliers functions for robustness.
  - Return a consistent DataFrame with 'outliers' and 'replacements' columns,
    even when empty.
  - Replace print statements with logging for better debug output.
  - Update the docstring for clarity.
  - Call the improved non-blocking plot_let_outliers_replacements to display
    results.

- **dataeng_main.py: Implement metric-based automatic selection for outlier detection**
  - Added a replacement quality metric (mean absolute error between replacement
    values and the local baseline from k-nearest non-outlier values) via a
    helper function. 
  - Updated the `outliers_search` function to automatically select the candidate
    method with the lowest quality metric, eliminating the need for manual user
    input. 

- **plot_utils.py: Add auto-save capability to plot_let_outliers_replacements**
  - Introduced `auto_save` and `save_path` parameters to automatically save
    figures. 
  - The function now calls `fig.savefig(save_path, dpi=300)` when `auto_save` is
    True. 
  - If the directory does not exist, it is created before saving the figure.
  - Maintains non-blocking display using `_plt.show(block=False)` and
    `_plt.pause(0.001)`. 

- **dataprocessing.py / dataeng_main.py: Enhance outlier search and auto-save functionality** 
  - Added edge-case handling in `outliers_search` to check if no candidate
    methods produce valid outlier-replacement results. If `quality_metrics` is
    empty, a message is printed and an empty DataFrame with columns ['outliers',
    'replacements'] is returned. 
  - Updated automatic selection in `outliers_search` to select only valid
    candidates based on the replacement quality metric. 
  - Implemented auto-save functionality in `identify_let_outliers`
    (data_processing.py) and updated  `dataeng_main.py` to pass auto-save
    parameters. Now, when auto-save is enabled, the code checks if the directory
    in `save_path` exists and creates it if needed, and saves the numeric
    outputs/figures automatically using filenames constructed from method
    parameters. 

- **dataeng_main.py: Replace print statements with logging and add entry point**
  - Replaced all print statements with logging calls (INFO/DEBUG/ERROR) for
    better configurability and debugging. 
  - Introduced the standard "if __name__ == '__main__'" entry point pattern.
  - Updated the saving procedure for processed data to generate an output file
    with a _replaced.out suffix in the same directory as the original input. 
  - Key processing steps, such as data import and saving, are now logged.

- **dataeng_main.py: Refactor main function into smaller helper functions**
  - Split the main function into:
    - `import_and_clean_data` for data import and cleaning.
    - `process_outliers` for running outlier detection and replacement,
      including automatic candidate selection based on a quality metric.
    - `save_processed_data` for saving the processed data in the same *.out
      format as the original.
  - This refactoring improves code readability, maintainability, and testing.

- **data_analysis_utils.py: Refactor get_cut_voxel_from_filename**
  - Add error handling for filename parsing in get_cut_voxel_from_filename

- **dataeng_main.py: Enhance import_and_clean_data function**
  - Added fallback extraction of `cut_in_um` and `voxel_in_um` from the last
    directory name in `data_dir` using regex patterns when the filename does not
    provide these values.
  - Converts values from mm to micrometers when needed and logs the extraction
    process.
  - Raises a `ValueError` if neither the filename nor the directory yields valid
    cut and voxel values.

- **eda_main.py: Replace print statements with logging**
  - All print calls in the main function have been replaced with logger.info and
    logger.debug calls for improved output control and consistency.
  - Verbose output is now managed via the logging configuration.
  - Existing EDA plotting and processing functionality remains unchanged.

- **data_analysis_utils.py: Implement extract_cut_voxel function and integrate logging**
  - Added `extract_cut_voxel` function the get the cut and voxel values from the
    full file path. The function first attempts to extract the values from the
    filename; if the extraction fails, it falls back to extracting these values
    from the last directory name in the path.
  - Replaced print statements in `get_cut_voxel_from_filename`,
    `calculate_zero_values_percentage`, and `drop_zero_columns` with logging
    calls. 
  - Enhanced `drop_zero_columns` by including new input arguments:
    - `verbose` to set the verbosity level.
    - `zero_percentage`, the precomputed percentage of zero values for each
      column, to avoid recalculating it if already computed at a higher level.
  - Enhanced `calculate_zero_values_percentage` by adding verbosity-level
    handling via an input parameter. 
  - Enhanced `cut_dataframe_with_primary` and
    `find_primary_column_with_fewest_zeros` by adding dynamic primary search
    pattern handling (appending '_1' if missing).

- **preprocessing_utils.py: Update mandatory_let_df_manipulation**
  - Now uses the `primary_particle` parameter to dynamically create the primary
    search pattern (appending '_1' if missing) instead of using a separate
    parameter. 
  - Added new input parameters:
    - `drop_zero_thr` (float, default 100.0): Specifies the threshold percentage
      for dropping columns with high zero values. 
    - `drop_zero_cols` (bool, default True): Flag indicating whether to drop
      columns exceeding the zero percentage threshold. 
  - Updated documentation to reflect these changes and clarify that the function
    returns the processed DataFrame, primary particle zero percentages, and
    overall zero percentages. 

- **data_analysis_utils.py: Add find_bragg_peak_start function**
  - Added a function that computes the first derivative of an input array or
    Series (approximating a Bragg curve) and returns the index where the 
    derivative exceeds a specified fraction (threshold_ratio) of the maximum
    derivative. 
  - The function supports both NumPy arrays and pandas Series as input.

- **dataeng_main.py: Update import_and_clean_data and main function**
  - Introduced new input parameters `primary_particle`, `drop_zero_cols`, and 
    `drop_zero_thr` to `import_and_clean_data` for standardized handling of the
    primary particle and controlling the dropping of columns with high zero
    percentages. 
  - Updated `mandatory_let_df_manipulation` to use `primary_particle` (ensuring
    it ends with '_1') instead of a hard-coded pattern. 
  - Modified the main function to define `primary_particle` (set to 'proton')
    and pass the new parameters. 
  - In `process_outliers`, compute the starting x-value of the Bragg peak using
    `find_bragg_peak_start` and use it to set the lower x-bound (x_bounds_start)
    for outlier detection and plotting. 
  - Adjusted subplot x-range in outlier processing accordingly.

- **eda_main.py: Enhance data import and cleaning parameters**
  - The helper function `import_and_clean_data_eda` now accepts a new parameter 
    `primary_particle` to standardize the primary particle identifier (ensuring
    it ends with '_1'), along with `drop_zero_cols` and `drop_zero_thr` to
    control the dropping of columns with high zero percentages. 
  - `import_and_process_dose_fluence` has been updated to accept
    `drop_zero_cols` and `drop_zero_thr` parameters. 
  - The plotting function `plot_eda_figures` now uses `primary_x_stop_mm`
    instead of primary_zero_perc for setting the final x-axis limit. 
  - The main function is updated to define `primary_particle` (set to 'proton')
    and pass the new parameters accordingly. 

- **eda_main.py: Rename plot_and_export to plot_extendend_comparison**
  - Renamed the function to better reflect its role in generating extended
    exploratory visualizations (e.g., single-element LET plots, fluence
    distributions, and secondary data comparisons).

- **eda_main.py: Update EDA plotting functions**
  - `plot_eda_figures`: 
    - Dynamically set the feature distribution x-range by checking if 
      subplot_specs['let_prof_x_range'] is not provided and then computing it 
      using eda.find_bragg_peak_start on the selected feature.
    - Updated the final subplot x-axis limit to use df_non_zero['x'].iloc[-1].
  - `plot_extended_comparison`:
    - Dynamically set the feature distribution x-range by checking if 
      subplot_specs['feat_dist_x_range'] is not provided and then computing it 
      using eda.find_bragg_peak_start on the selected feature.
    - Updated the final subplot x-axis limit to use primary_x_stop_mm.

- **VAE Training Script:**
  - Created a dedicated script for Variational Autoencoder (VAE) training by
    extracting and retaining only the VAE design, training, and visualization code
    from main_conditional.py.
  - Removed all EDA and Data Engineering code to focus on VAE-specific routines.
  - Updated module imports, configuration, and helper functions to support a
    streamlined VAE training workflow.

- **preprocessing module:**
  - Copied the following functions from eda_module:
    - `import_out_file`
    - `extract_cut_voxel`
    - `extract_track_or_dose_cols`
    - `convert_index_feature_to_depth`
    - `cut_dataframe_with_primary`
  - These functions are now centralized in the preprocessing module to be
    used consistently across EDA, Data Engineering, and VAE training pipelines.
  - Removal from eda_module is postponed for now.

- **ml_prep module: Add split_and_scale_dataset function**
  - Added and optimized the split_and_scale_dataset function:
    - Added type hints and changed the verbose parameter to a boolean.
    - Added validation to ensure train, validation, and test sizes sum to 1.
    - Improved default parameter handling and error messaging.

- **VAE Training Script:**
  - Integrated a new `import_and_clean_data` function that leverages the 
    preprocessing module to import and clean LET data for VAE training.
  - Updated the data preparation pipeline to use `split_and_scale_dataset` from 
    the ml_prep module for splitting the dataset into training, validation, and 
    test sets and scaling them.
  - These changes streamline the VAE training workflow and ensure consistency 
    in data preprocessing across the project.

- **preprocessing module:**
  - Fixed an error where an incorrect function was included instead of
    `cut_dataframe_with_primary`. The correct implementation is now restored,
    ensuring that data truncation based on the primary particle stopping point
    works as expected.

- **VAE Training Script:**
  - Added `reorder_identity_features` to reposition identity features 
    before dataset splitting, supporting the new AutoEncoder architecture.
  - **Reminder:** Further updates are needed in the autoenc module:
    - Refactor `plot_latent_space` to better visualize the latent space with
      identity features.
    - Refactor `plot_test_and_reconstructed` to correctly reattach or visualize
      identity features.
    - These updates will be implemented in a dedicated branch.

- **VAE Module:**
  - Updated the AutoEncoder loss computation in __build_loss to support the 
    identity feature bypass:
    - When identity features are present (using the new identity_dim parameter),
      the reconstruction loss (MSE) is computed only on the processed features.
    - This prevents penalizing the identity features that bypass the encoder.

- **VAE Module:**
  - Updated the AutoEncoder class to support an identity feature bypass.
    - A new parameter `identity_dim` is added to `net_params` (default 0) to
      define the number of identity features.
    - The input is split into a processed part (to be encoded and decoded) and
      an identity part (bypassing the latent space).
    - The final output concatenates the reconstructed processes features with
      the unchanged identity features.

- **VAE Training Script:**
  - Update `train_vae` to support identity feature bypass
    - Added a new parameter `identity_dim` (default 0) in `vae_params_opt`.
    - Passed `identity_dim` to `net_params` so the AutoEncoder can split the
      input into processed and identity features.
    - Verified that the training pipeline remains unchenged apart from the new
      identity feature ahndling.

- **autoenc module:**
  - Improved `plot_latent_space` to support the new identity feature bypass.
    - The function now checks if the model has an identity bypass and processes 
      only the processed features.
    - Added type annotations and error handling (e.g., checking for an empty 
      DataLoader and proper label extraction).
    - Changed `label_column` to `color_column` to better indicate that the
      values are used for coloring; changed `labels` to `color_values` for the same reason.
    - Updated the docstring to document these changes.

- **ml_prep module:**
  - Added a `col_mapping` attribute to the DataBuilder class when the input 
    data is a pandas DataFrame, storing a mapping from column names to indices.
  - Updated the `create_data_loader` function to attach this column mapping to 
    the returned DataLoader object, facilitating downstream tasks that require 
    access to column names.

- **ml_prep module:**
  - Updated `train_val_test_split` to include a `preserve_df` flag and improve
    output clarity.
    - When `preserve_df` is True, the function uses .iloc for row indexing on 
      pandas DataFrames, preserving column names.
    - This allows downstream functions (e.g., DataBuilder) to access column
      mappings.
    - When a single input DataFrame is provided with preserve_df=True, the 
      function now returns a tuple of three DataFrames (train, validation, and 
      test splits) instead of lists.
    - For multiple inputs, the function still returns a tuple of lists.
    - This change helps preserve column names and mappings for downstream 
      tasks.

- **ml_prep module:**
  - Updated `train_val_test_scale` to preserve DataFrame structure when inputs
    are DataFrames:
    - The function now retains the original index and columns by converting
      scaled numpy arrays back to DataFrames.
    - This allows column names and mappings to be preserved for downstream tasks.

- **ml_prep module:**
  - Added a new function `plot_train_test_val_distribution_df` to plot the
    distribution of training, validation, and test datasets when provided as 
    pandas DataFrames. This function uses the DataFrame's column names to select 
    features for plotting, removing the need for a separate original DataFrame 
    argument.

- **autoenc module:**
  - Updated `plot_test_and_reconstructed`:
    - Now supports a `plot_cols` parameter that accepts either column indices 
      or column names.
    - If column names are provided, the function retrieves the column mapping 
      from the DataLoader (via its `col_mapping` attribute) and converts them 
      to indices.
    - This enhancement allows for more flexible selection of features for 
      plotting.

- **vae_training.py:**
  - Updated the data loader creation pipeline by changing
    create_data_loaders_from_dataframes to return a namedtuple (`DataLoaders`)
    instead of a dictionary. This improves readability and makes it easier to
    access the train, validation, and test loaders.  

- **EDA Main Script:**
  - Replaced function calls to eda_module with corresponding functions from the
    preprocessing module:
    - `import_out_file`
    - `extract_cut_voxel`
    - `extract_track_or_dose_cols`
    - `convert_index_feature_to_depth`
    - `cut_dataframe_with_primary`
  - This centralization of preprocessing functions ensures consistency across 
    EDA, Data Engineering, and VAE training pipelines.
  - The removal of these functions from eda_module is postponed for now.

- **Data Engineering Main Script:**
  - Replaced calls to eda_module functions with corresponding functions from the
    preprocessing module:
    - `import_out_file`
    - `extract_cut_voxel`
    - `extract_track_or_dose_cols`
    - `convert_index_feature_to_depth`
    - `cut_dataframe_with_primary`
  - Removed these functions from eda_module to centralize and standardize
    preprocessing across multiple pipelines.
  - This change centralizes data preprocessing logic across multiple pipelines,
    ensuring consistent behavior.

- **New Feature (Optuna Optimization):**
  - Introduced a new script `vae_optuna.py` for hyperparameter optimization 
    of the Variational Autoencoder using Optuna.
  - Developed in a dedicated branch (`optuna_optimization`) to allow for 
    independent development and testing before merging into the main codebase.

- **VAE Optimization Script (vae_optuna.py):**
  - Removed data import and cleaning from the Optuna objective function.
  - Introduced `prepare_data()`, which preloads, cleans, splits, and scales data,
    storing results in global variables.
  - The objective function now uses these precomputed DataLoaders for training,
    reducing overhead and ensuring consistency across trials.

- **VAE Optimization Script (vae_optuna.py):**
  - Removed directory configuration from the objective function by defining 
    global constants for `vae_plot_dir` and `vae_history_dir`. The objective 
    now uses these global settings, reducing redundancy and ensuring consistent 
    output paths.

- **VAE Optimization Script (vae_optuna.py):**
  - Extended hyperparameter optimization to include network architecture.
  - The `net_size` parameter is now dynamically optimized by allowing Optuna to
    determine the number of hidden layers (from 1 to 5) and the size of each
    layer. 
  - This enhancement allows the VAE training pipeline to explore a wider range
    of model architectures. 

- **VAE Training Script (vae_training.py):**
  - Updated `train_vae` to return the trained model alongside the training
    history. 
  - Main function now uses the returned model to generate and save plots (latent
    space, test vs. reconstructed data), decoupling training from visualization. 

- **VAE Optimization Script (vae_optuna.py):**
  - Removed plotting-related parameters (`vae_plot_dir` and `vae_history_dir`)
    from the call to `train_vae` in the objective function. 
  - Updated the objective function to unpack the tuple returned by `train_vae`
    (model, history) and extract the final validation loss from
    `history['val_losses']`. 
  - This change further decouples training from visualization and simplifies the
    optimization loop. 

- **VAE Optimization Script (vae_optuna.py):**
  - Added persistent storage using SQLite so that each trial is saved 
    immediately upon completion. The study is created with 
    `storage="sqlite:///optuna_study.db"` and `load_if_exists=True`, 
    enabling the optimization process to resume from previous runs.

- **VAE Optimization Script (vae_optuna.py):**
  - Added code examples for generating visualizations using Optuna's built-in
    functions (e.g., `plot_optimization_history`, `plot_param_importances`, and
    `plot_slice`).
  - These visualizations offer a detailed view of the study progress and
    parameter effects, improving interpretability of the hyperparameter search.

- **VAE Optimization Script (vae_optuna.py):**
  - Added a pruning strategy using Optuna's MedianPruner (n_warmup_steps=5) to
    stop unpromising trials early.
  - Removed `training_epochs` from the hyperparameter search and set it as a fixed
    value (50 epochs) to reduce the search space.
  - Updated the study creation to use a fixed study name ("vae_optimization") with
    SQLite storage, ensuring persistent storage of trial results.

- **VAE Optimizazion Script (vae_optuna.py):**
  - Added more visualization code in `vae_optuna.py` to monitor the optimization
    process. 
  - Generated and saved the following plots:
    - Optimization history (`opt_history.png`)
    - Paramter importance plot (`opt_param_importance.png`)
    - Slice plot for the top four hyperparameters (`opt_slice.png`)
    - Parallel coordinate plot (`opt_parallel_coordinate.png`)
    - Contour plot (`opt_contoutr.png`).
  - The code automatically computes parameter importances using Optuna's
    importance API, select the top four parameters, and passes them to the
    visualization functions.
    
- **VAE Optimization Visualization**:
  - Updated visualization code in `vae_optuna.py`:
    - Capped extreme objective values in the optimization history plot by
      modifying each trace’s y-values to a maximum threshold. 
    - Applied similar capping to the slice plot (y-values) and contour plot
      (z-values), and updated the layout to reflect the clipping. 
    - Updated the parallel coordinate plot to use filtered trial data and
      selected key hyperparameters. 
    - These changes improve the clarity of the optimization visualizations by
      reducing the impact of outlier objective values. 

- **AutoEncoder Fit Function Update:**
  - Modified the `fit` function the `AutoEncoder` class to support early pruning
    during training:
    - An optional `trial` parameter has been added so that, when provided (e.g.,
      during Optuna hyperparameter optimization), the function reports the
      validation loss at each epoch and checks for pruning via `trial.report()`
      and `trial.should_prune()`. When no trial is supplied, the function
      behaves as before, allowing for standard training runs.

- **VAE Optimization Refactoring:**
  - Encapsulated preprocessed data into a new `VAEData` class, eliminating the
    use of globals and simplifying data passing.
  - Created a dedicated `visualize_study` function to separate all visualization
    logic from the optimization workflow.
  - Improved directory handling using `pathlib.Path` for cross-platform
    compatibility. 
  - Refactored `vae_optuna.py` to leverage the updated `fit` function in the
    `AutoEncoder` class, which now accepts an optional `trial` parameter for
    early pruning.

- **AutoEncoder Optimizer Refactoring:**
  - Introduced a new key `optimizer_type` in the `optimizer_params` dictionary
    to allow selecting different optmizers (e.g. Adam, SGD, RMSprop) during
    training and hyperparamter optimization.
  - Updated the `__init__` method in the `AutoEncoder` class to choose the
    optimized based on `optimizer_type`, including fetching optimizer-specific
    paramters such as momentum for SGD.
  - Fixed the device assignment condition to correctly check if a device was
    provided.

- **VAE Optimization Objective Update:**
  - Updated the `opt_objective` function in `vae_optuna.py` to include optimizer
    selection in the hyperparameter search.
  - The function now suggests an optimizer type (choosing among "Adam", "SGD",
    and "RMSprop") and, if needed, additional hyperparameters such as momentum
    for SGD.
  - The key optimizer selection has been updated to `optimizer_type` to ensure
    consistency with the updated `AutoEncoder` class.

- **Contour Plot Logic Improvement in VAE Optimization:**
  - Revised the contour plot generation in `vae_optuna.py` to create unique
    parameter pairs and generate separate contour figures for each pair.

- **Dynamic Identity Dimension in VAE Optimization:**
  - Updated the `prepare_data()` function to store the identity features list in
    the `VAEData` container. 
  - Modified the `opt_objective` function so that `"identity_dim"` is set as
    `len(data.identity_features)` instead of a hard-coded value. 
  - Updated the main function to similarly set `vae_params["identity_dim"]` based
    on the length of `data.identity_features`. 
  - These changes ensure that the model configuration automatically adjusts if
    the identity features list is modified. 

- **VAE Optimization Pipeline Enhancements:**
  - Introduced a helper function `build_vae_model_from_params` that encapsulates
    the logic for reconstructing the network architecture (e.g. generating
    `net_size`), building the `vae_params`, `net_params`, and `optimizer_params`
    dictionaries, and instantiating the `AutoEncoder` model. 
  - Updated the main function in `vae_optuna.py` to re-train the best model
    using the best trial's hyperparameters, export the best model's state
    dictionary (e.g., `best_vae_model.pt`), and generate a reconstruction
    comparison figure on test data.

- **Output Activation Update:**
  - Switched the VAE decoder's output activation to use Softplus to enforce
    non-negative outputs.
  - Centralized this setting by introducing a global constant
    `DEFAULT_OUTPUT_ACTIVATION` (set to `nn.Softplus`), which is now used in
    optimization pipeline.

- **Decoder Helper Function Refactoring (autoenc module):**
  - Updated `concat_rev_lin_layers` in `helpers.py`:
    - Re-ordered the parameters to include an explicit `input_shape`
      (latent_dimension) and `output_shape` (original input dimension).
    - Changed the default output activation from `nn.ReLU` to `nn.Softplus` so
      that the decoder produces non-negative outputs. 
    - Appended a final linear layer to map from the last hidden layer to the
      desidered output dimension, ensuring that the decoder mirrors the encoder
      architecture. exactly.
  - These changes simplify configuration and enforce a symmetric decoder
    structure. 

- **VAE Model Refactoring:**
  - **Decoder Architecture**:
    Reworked the Decoder class to form an exact mirror of the encoder. 
    - Removed the extra projection layers (both the initial linear projection
      and the final output linear layer) that were present in the old version. 
    - Replaced the previous decoder construction with a call to
      `concat_rev_lin_layers` that accepts explicit `input_shape` (latent
      dimension) and `output_shape` (original input dimension) parameters. 
  - **Output Activation:**
    Changed the default output activation from `nn.ReLU` to `nn.Softplus` to
    enforce non-negative outputs. 
  - These modifications simplify the architecture, improve symmetry between
    encoder and decoder, and better enforce domain-specific output constraints. 

- **VAE Optimization Refactoring (vae_optuna.py):**
  - **Data Preparation & DataLoader Creation:**
    - Refactored the `prepare_data()` function so that it now returns a processed
      dataset (dataframes for train, validation, and test) along with the scaler
      and other attributes instead of creating DataLoaders. 
    - Moved DataLoader creation to the `opt_objective()` function, enabling the
      batch size to be tuned as a hyperparameter. 
  - **Data Container Renaming:**
    - Renamed the data container class from VAEData to Data to better reflect
      its contents. 
  - **Model Instantiation Helper:**
    - Added a new helper function `build_vae_model_from_params` to consolidate
      the reconstruction of the network architecture from best trial parameters,
      reducing code duplication between training and optimization pipelines. 
  - **Output Activation Update:**
    - Changed the default output activation from `nn.Softplus` to `nn.ReLU` (via
      the `DEFAULT_OUTPUT_ACTIVATION` constant) to simplify configuration and
      ensure consistency. 
  - **General Clean-up:**
    - Improved variable naming and logging messages for enhanced clarity.

- **Decoder Final Layer Configurability in models.py:**
  - Added a new parameter `use_final_activation` to the `Decoder` class.
  - When `use_final_activation` is set to True, the decoder applies
    normalization and the specified activation (e.g. Softplus) on its final
    layer. When False, the final layer is a plain linear projection. 
  - Updated the `AutoEncoder` class to pass the new `use_final_activation`
    parameter from net_params to the `Decoder` constructor. 
  
- **Net Parameters and Model Instantiation in vae_optuna.py:**
  - Modified the construction of `net_params` in both the `opt_objective` and
    `build_vae_model_from_params` functions to include a new key
    "`use_final_activation`", allowing users to control the behavior of the
    final decoder layer. 
  - Optionally, the `use_final_activation` parameter tuned as a hyperparameter.
  - Updated logging and variable naming for clarity.

### Added
- **Custom Loss with Negative Penalty:**  
  Added a new loss function `customLossWithNegPenalty` that applies an extra
  penalty on negative outputs in the processed (non-identity) features. This
  loss function now optionally computes the loss only on the processed part when
  `proc_dim` is specified, ensuring that identity features are not penalized. 

- **Added vae_module/autoencoder.py:**
  - This new file include an updated version of the AutoEncoder class to
    integrate the new `customLossWithNegPenalty` in both training and validation steps.  
  - Extended the input arguments of the `fit` method to accept a dictionary
    `loss_hyperparams` and then create the loss function accordingly, within the
    `train_step`.

### Changed
- **vae_optuna Integration:**  
  Updated `vae_optuna.py` to pass the new `loss_hyperparams` parameter  into the
  model initialization.   
  Adjusted the hyperparameter search space and documentation accordingly.

### Added
- **Decoder Final Layer Bias:** 
  Initialized the final layer bias to -2.0 in the decoder when
  `use_final_activation` is enabled. This change helps drive the pre-activation
  values into a more negative range, allowing the Softplus activation to output
  values closer to zero and mitigating the lower clamp issue.  

### Changed
- **VAE Optimization Script (vae_optuna.py):**
  - **Architecture and Optimization Enhancements:**
    - Refactored the optimization pipeline to encapsulate preprocessed data in a
      dedicated container returned by a new `prepare_data()` function, thereby
      decoupling data loading from hyperparameter tuning. 
    - Enhanced visualization routines: a helper function `cap_trace_values` was
      introduced to reduce repeated code when capping plot values.
  - **Centralized Global Settings:**
    - Removed hard-coded values for `training_epochs` and `use_final_activation`
      from both the `opt_objective` and `build_vae_model_from_params` functions. 
    - Introduced two global constants, `DEFAULT_TRAINING_EPOCHS` and
      `DEFAULT_USE_FINAL_ACTIVATION`, to centralize these settings. This change
      streamlines configuration and makes it easier to adjust training settings
      in one place. 

- **Helpers Module Update (src/vae_module/helpers.py):**
  - **Enhanced Type Safety and Clarity:**
    Refactored all helper functions to require callable activation and normalization parameters. This change improves type safety by clarifying that users should pass functions (e.g. `nn.ReLU`) rather than instantiated modules.
  - **Robust Normalization Handling:**
    Updated `concat_lin_layers` and `concat_rev_lin_layers` to treat both `None`
    and `nn.Identity` as “no normalization.” In `concat_rev_lin_layers`, added
    explicit handling for the case where `output_activation` is None — in such
    cases, the final layer is built as a plain linear layer.
  - **Improved Documentation:**
    Updated docstrings across all helper functions to clearly explain the
    expected input types, behavior, and default values, aiding future
    maintenance and usage. 
  - **Consistent Layer Encapsulation:**
    Ensured that even single linear layers are encapsulated in `nn.Sequential` 
    for consistent downstream processing. 

- **AutoEncoder Module Update (autoencoder.py):**
  _ **Enhanced Type Annotations & Docstrings:**
   Added detailed type hints and improved docstrings across the module to
   clarify method behaviors and support static analysis. 
  _ **Robust Encoder Construction:**
    Modified the Encoder class to determine the output dimension for the latent
    space layers using the last value in `hidden_layers_dim` instead of indexing
    into the Sequential container.
  - **Naming Conventions:**
    Renamed `customLossWithNegPenalty` to `CustomLossWithNegPenalty` to adhere
    to CamelCase conventions.
  - **Optimizer Refactoring:** 
    Extracted optimizer creation into a helper function (`create_optimizer`) to
    simplify the AutoEncoder constructor and streamline future optimizer
    additions. 
  - **Logging Enhancements:**
    Replaced print statements in the training loop with logging calls for
    improved production-readiness and configurability. 
  - **Improved Error Handling:**
    Added assertions to ensure that the processed dimension is non-negative,
    catching configuration errors early. 
  - **Code Cleanup:**
    Removed commented-out code and improved overall readability and
    maintainability. 
  - **Update Custom Loss Function:**
    - Update `CustomLossWithNegativePenalty` to accept a new boolean parameter
      `use_neg_penalty` to determines whether the negative penalty term is
      applied.
    - Modify the forward pass to set the negative penalty to zero if
      `use_neg_penalty` is False.
    - Update the `AutoEncoder.train_step` and `AutoEncoder.validate_step` to
      retrieve the `use_neg_penalty` flag from the `loss_hyperparameters`
      allowing the optimization procedure to decide wheter ot not to use the
      negative penalty.

- **VAE Optimization Script Update:**
  - **Training Settings & Architecture:**
    - Changed the default for `use_final_activation` from `False` to `True` and
      updated `skip_norm_in_final` to remain `True`. 
  - **Output Activation Enhancements:**
    - Replaced the previous output activation (`nn.Softplus` with beta=10.0)
      with a suite of custom activation classes for testing: 
      - `ELUWithLearnableOffset`: Now used as the default output activation. Its
        initial offset (`offset_init`) is tuned as a hyperparameter (best model
        found with `offset_init` ≈ 1.3).  
      - `ShiftedSoftplus`: Added for testing; however, it did not yield
        satisfactory results. 
      - `PELU`: Introduced for further experimentation; not tested yet.
  - **Hyperparameter Tuning Improvements:**
    - Added a hyperparameter suggestion for `offset_init` to allow the
      optimization procedure to determine the best initial offset for the
      `ELUWithLearnableOffset` activation. 
    - Adjusted other hyperparameters (e.g., `beta` and `neg_penalty_weight`) to
      accommodate the new activation functions. 
  - **Miscellaneous Updates:**
    - Updated network configuration and logging to align with the revised
      training settings and output activations. 
    - Maintained persistent storage and improved visualization functions for
      monitoring the study. 

- **VAE Optimization Script Update:**
  - **Visualization Enhancements:**
    - Added a latent space visualization step before generating test vs.
      reconstructed plot comparison.
    - The generated latent space figure is saved as a PDF in the designated plot
      directory. 

### Added
- **Configuration Export:**
  The optimization pipeline (`vae_optuna.py`) now exports a configuration
  dictionary (containing the best hyperparameters) to a JSON file alongside the
  saved state dictionary. This enables precise reconstruction of the model for
  generative tasks. 
- **Model Loading Enhancement:**
  Added a new class method `load_from_file` to the `AutoEncoder` class. This
  method allows re-instantiation of the model from the saved state dictionary
  and configuration file, ensuring the architecture matches the best trial. 
- **Generative Workflow Script:**
  Introduced a new file, `vae_generate.py`, which implements the generative
  workflow. This script loads the trained model using the new `load_from_file`
  method, processes lower-resolution input data, and generates high-resolution
  outputs. (Further code customization required before testing)
    
### Changed
- **Data Scaling Improvements (ml_prep/data_utils.py):**
  - Updated the `train_val_test_scale` function (and indirectly
    split_and_scale`_dataset) to handle empty validation and test splits (i.e.
    when `val_size` or `test_size` is 0.0). Now, before applying
    `scaler.transform()`, the code checks whether the input split is non-empty,
    ensuring that empty splits are returned without errors. 

- **Generative Pipeline Enhancements in vae_generate.py:**
  - **Low-Resolution Data Preparation:**
    Downsampled the high-density LET data (micrometer sampling) to simulate
    low-density input (millimeter sampling) and created a dedicated Data
    instance with a DataLoader. The DataLoader now uses the optimized batch size
    retrieved from the configuration JSON. 
  - **Latent Space Interpolation:**
    Implemented latent space interpolation with an adjustable `upsample_factor`
    to generate intermediate latent codes between adjacent latent vectors. This
    allows the decoder to produce high-resolution outputs (e.g., upsample from
    millimeter to micrometer resolution) without directly upsampling the input. 
  - **Model Loading & Generation:**
    Integrated the new `AutoEncoder.load_best_model` method to re-instantiate
    the trained model from the saved state dictionary and configuration. The
    updated `generate_high_res_output` function now decodes both original and
    interpolated latent codes to generate the final high-resolution LET output. 
  - **Visualization:**
    Added code to plot and optionally save the generated high-resolution output
    alongside the low-resolution input. 

- **AutoEncoder Module Refactoring:**
  - **Optimizer Creation:**
    Replaced the hardcoded optimizer instantiation with a dedicated helper
    function (`create_optimizer`), allowing flexible selection of optimizer type 
    and parameters.   
  - **Activation Function:**
    Introduced new custom activation function (`ELUWithLearnableOffset`) for
    testing different output activation strategies. 
  - **Code Cleanup:**
    Improved overall code organization and consistency to better support
    reconstruction and integration with the generative pipeline. 

- **Generative Workflow Script Update:**
  - Upscaling the identity features via linear interpolation (instead of simple
    replication). 
  - Concatenating the decoded processed features with the upsampled identity
    features. 
  - Inverse transforming the generated data and sorting by the 'x' column before
    plotting. 

### Fixed
- **Redundant Suffix Issue:**  
  Resolved an issue where the primary particle identifier could end up with a
  redundant `"_1"` by checking if it already ends with that suffix before
  appending. 

### Added
- Created a new utility module `src/vae_module/activations.py` for activation
  functions. 

### Changed
- Updated `src/vae_module/__init__.py` to include `activations.py`.

### Removed
- Removed redundant code in `src/vae_optuna.py`.
- Removed redundant code in `src/vae_module/autoencoder.py`.

### Changed
- Updated `split_and_scale_dataset` to include a `shuffle` parameter for
  controlling dataset shuffling. 
- Updated function docstring for better clarity on new parameters.

- **Generative Workflow Script Update:**
- Updated `split_and_scale_dataset` call in `prepare_data()` to allow dataset
  shuffling via `shuffle=False` for deterministic behavior. 
- Improved visualization by replacing `plt.plot()` with `plt.scatter()` for
  better distinction between generated and low-resolution data. 
- Adjusted plotting properties:
  - Increased transparency (`alpha=0.9` for generated data and `alpha=0.7` for
    low-res data). 
  - Changed low-res data markers to `'x'` and adjusted size for clarity.
- Removed unnecessary sorting of `generated_high_res_df` by `'x'`, optimizing
  performance. 

- **Generative Workflow Script Update:**
- Refactored `vae_generate.py` to use a single global variable `RESAMPLE_FACTOR`
  for both downsampling and upsampling. 
- Removed redundant manual assignments of `downsample_factor` and
  `upsample_factor`, ensuring consistent resampling behavior. 
- Improved code maintainability by centralizing resampling factor control.

- **Generative Workflow Script Update:**
- Improved visualization:
  - Modified the **plot title** to include `RESAMPLE_FACTOR` dynamically.
  - Changed x-axis label from `"Sample Index"` to `"x [mm]"`.
  - Changed y-axis label from `"LET [units]"` to `"LTT [keV μm⁻¹]"` for improved
    clarity. 


---

## [1.0.0] - 2025-02-04

### Added
- **Initial Release:**  
  This is the first release of `dataeng_main.py`, providing a dedicated data
  engineering script that: 
  - Imports and cleans LET and dose data.
  - Determines the primary particle stopping point.
  - Implements outlier detection and replacement strategies.
