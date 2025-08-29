# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### [2025-08-28]

#### 🧹 Removed

- **Project Structure Cleanup:**
  - **model_history/**: Removed legacy model history directory
    - Contained outdated training history JSON files from November 2024 - February 2025
    - Current architecture uses `vae_history/` for model histories
    - No references found in current codebase
  - **dev/**: Removed development directory with legacy code
    - Contained `older_version/vae_optuna_claude.py` with outdated module imports
    - Used deprecated module structure (`ml_prep`, `preprocessing_module`, `vae_module`)
    - No longer compatible with current structured architecture
  - **notebooks/**: Removed Jupyter notebooks directory
    - Contained `VAE_data_augmentation_torch.ipynb`
    - Legacy development artifact no longer maintained
  - **backup/**: Removed backup directory
    - Contained `vae_optuna.py` backup file
    - Legacy development artifact superseded by current optimization pipeline

- **Unused Files Cleanup:**
  - **src/vae_performance_metrics.py**: Removed unused performance metrics module
    - Functions were duplicated in `src/vae_generate.py` as class methods
    - No active imports or usage found in codebase
    - Functionality preserved through `VAEGenAnalyzer` class methods
  - **README.md**: Removed reference to deleted performance metrics module
    - Updated directory structure documentation to reflect current codebase

#### 🔄 Refactored

- **Utils Module Enhancement:**
  - **src/utils/latent_utils.py**: Enhanced latent space plotting functionality
    - Added `close_after_save` parameter to plotting methods for better memory
      management in non-interactive environments
    - Improved function signatures with proper return type annotations
    - Enhanced plot lifecycle management for batch processing scenarios
  - **src/utils/__init__.py**: Expanded public API exports
    - Added `is_interactive_environment` utility for environment detection
    - Enhanced module accessibility for interactive vs batch execution modes

- **Analysis Pipeline Improvements:**
  - **src/vae_optimization_analysis.py**: Streamlined imports and dependencies
    - Removed unused torch import for cleaner dependency management  
    - Integrated `BaseVAEPipeline` for consistent pipeline architecture
    - Added `display_plot` import for improved visualization handling
    - Refactored import organization for better maintainability

- **Configuration Updates:**
  - **src/configs/**: Enhanced configuration consistency across modules
    - **base_config.py**: Minor configuration field adjustments
    - **optimization_analysis_config.json**: Updated analysis parameters
    - **task_config.py**: Configuration schema refinements for better validation

- **VAE Generation Pipeline:**
  - **src/vae_generate.py**: Major refactoring for pipeline architecture
    - Migrated `VAEGenerate` class to inherit from `BaseVAEPipeline` for
      consistent architecture and automatic path resolution
    - Simplified logger initialization by leveraging pipeline base class
    - Enhanced `VAEGeneratorAnalysis` class constructor to accept logger
      directly instead of logger object wrapper
    - Added `display_plot` integration for improved visualization handling
    - Updated plot save paths to use resolved `output_dir` from config

- **Generation Configuration:**
  - **src/configs/generation_config.json**: Path resolution system updates
    - Replaced hardcoded `study_path` with modular `optuna_io_dir` and 
      `database` fields for better path management
    - Updated file path fields to use filename-based approach:
      - `hparams_config_path` → `hparams_config_filename`
      - `model_path` → `model_weights_filename` 
      - `history_path` → `history_filename`
    - Renamed `plot_dir` to `output_dir` for consistency across pipeline

- **Pipeline Infrastructure:**
  - **src/core/base_pipeline.py**: Enhanced path resolution
    - Added automatic config update with resolved output directory path
    - Improved integration with generation pipeline requirements

- **Post-Analysis Enhancements:**
  - **src/vae_post_analysis.py**: Improved analysis workflow integration
    - Enhanced compatibility with new pipeline architecture
    - Better integration with updated utility modules
    - Removed manual path resolution logic (now handled by BaseVAEPipeline)

- **Optimization Analysis Cleanup:**
  - **src/vae_optimization_analysis.py**: Streamlined directory management
    - Removed redundant `ensure_directory_exists` calls
    - Leveraged BaseVAEPipeline automatic directory creation

- **Filesystem Utilities:**
  - **src/utils/filesystem_utils.py**: Enhanced file system interaction
    - Improved path handling and environment detection capabilities
    - Better support for both interactive and batch execution modes

- **Plotting Infrastructure:**
  - **src/utils/plot_utils.py**: Enhanced visualization capabilities
    - Improved plot lifecycle management and memory handling
    - Better integration with environment detection for display decisions

### [2025-08-27]

#### 🔄 Refactored

- **Path Resolution System Overhaul:**
  - **Dynamic Path Configuration**: Completely refactored hardcoded path handling
    across the entire configuration system
    - **src/core/base_pipeline.py**: Added dynamic optuna I/O directory and
      database path resolution with project root support
    - **src/configs/base_config.py**: Removed hardcoded `study_path`, added
      flexible `optuna_io_dir` and `database` fields with intelligent defaults
    - **src/configs/task_config.py**: Migrated PostTrainingConfig from hardcoded
      paths to filename-based approach with runtime path expansion
    - **src/configs/post_training_config.json**: Updated configuration to use
      directory + filename pattern instead of absolute paths

- **Database File Relocation:**
  - **Project Structure**: Moved `optuna_study.db` out of source directory to
    project root and `vae_opt_output/` locations
    - **Clean Separation**: Source code directory no longer contains data files
    - **Improved Organization**: Better separation between code and generated data

- **Configuration Architecture Improvements:**
  - **Runtime Path Resolution**: Implemented centralized path resolution in
    BaseVAEPipeline that works regardless of project root location
  - **Backward Compatibility**: Maintained compatibility through sensible defaults
    and fallback mechanisms
  - **Relocatable Configurations**: Projects can now be moved between environments
    without path reconfiguration
  - **Enhanced Logging**: Added path resolution logging for debugging and
    transparency

- **Configuration Standardization and Cleanup:**
  - **Configuration Field Naming**: Standardized configuration field names 
    across all modules for consistency
    - **src/configs/optuna_config.json**: Changed `output_dir` to `optuna_io_dir`
      for semantic clarity and added explicit `database` field
    - **src/configs/post_training_config.json**: Fixed field name from 
      `model_filename` to `model_weights_filename` for consistency
    - **src/configs/task_config.py**: Removed "best_" prefixes from default 
      filenames in TrainingConfig and eliminated redundant field overrides 
      in PostTrainingConfig
  - **Base Pipeline Improvements**: Enhanced conditional path handling in 
    BaseVAEPipeline
    - **src/core/base_pipeline.py**: Added proper conditional checks for 
      optional directory fields to prevent initialization errors
    - **Separation of Concerns**: Better distinction between different 
      pipeline types and their directory requirements
  - **VAE Optimization Module**: Updated optimization pipeline for consistency
    - **src/vae_optimization.py**: Migrated from mixed directory usage to 
      consistent `optuna_io_dir` references throughout
    - **Timestamped Directory Handling**: Cleaner path management for 
      timestamped output directories
    - **Argument Parser**: Fixed description text to accurately reflect 
      optimization functionality

### [2025-08-26]

#### 🐛 Fixed

- **Training Pipeline Improvements:**
  - **src/core/training_utils.py**: Fixed missing `display_plot` import and
    function call in `VAETrainer`
    - **Import Addition**: Added missing import for `display_plot` from
      `utils.filesystem_utils`
    - **Plot Display Integration**: Added proper `display_plot()` call in
      training pipeline for visualization consistency
    - **Training Visualization**: Ensures training plots are properly
      displayed/saved according to execution environment

- **File I/O Enhancements:**
  - **UTF-8 Encoding**: Added explicit UTF-8 encoding when writing model summary
    files
    - **Character Set Safety**: Prevents encoding issues with special characters
      in model summaries
    - **Cross-platform Compatibility**: Ensures consistent file writing behavior
      across different operating systems

- **Configuration Management:**
  - **Path Configuration**: Updated configuration paths to be relative to
    project root
    - **Portability**: Improved project portability by using relative paths
      instead of absolute paths
    - **Deployment Flexibility**: Easier deployment and sharing of configuration
      files

#### 🔄 Refactored

- **Architecture Modernization:**
  - **Legacy Module Removal**: Completed removal of legacy modules and
    consolidated to structured architecture
    - **Module Consolidation**: Migrated remaining dependencies from legacy
      modules to structured core and utils modules
    - **Code Organization**: Improved logical grouping of functionality across
      the codebase
    - **Import Standardization**: Standardized imports to use the new structured
      module hierarchy

### [2025-08-25]

#### 🐛 Fixed

- **CLI Plot Display Issue:**
  - **src/utils/plot_utils.py**: Fixed `plot_train_test_val_distribution_df()`
    causing unwanted plot windows in CLI execution
    - **Explicit Axes Usage**: Replaced global `plt.*` calls with explicit axes
      methods (`ax.scatter()`, `ax.set_xlabel()`, etc.) to prevent automatic
      display
    - **Consistent Pattern**: Aligned function implementation with other
      plotting utilities in the module
  - **src/utils/filesystem_utils.py**: Corrected interactive environment
    detection logic
    - **TTY Detection Fix**: Removed incorrect TTY (terminal) detection that was
      causing CLI executions to be treated as interactive
    - **Precise Environment Detection**: Enhanced `is_interactive_environment()`
      to only detect genuine interactive environments (Jupyter, IPython, IDEs)
    - **CLI Compatibility**: Ensured command-line executions properly skip plot
      display while preserving figure creation and saving
  - **Impact**: Resolved execution pause and unwanted plot window opening when
    running `vae_training.py` from command line

### [refactor/shared-utils-structure] - 2025-08-21

#### ✨ Enhanced

- **Interactive Execution Support:**
  - **src/dataeng_main.py**: Enhanced execution model for both CLI and interactive environments
    - **Return Value Architecture**: Modified `main()` to return comprehensive results dictionary with all key outputs (`df_processed`, `primary_x_stop_mm`, `dose_profile`, etc.)
    - **Interactive Detection**: Integration with `utils.filesystem_utils.is_interactive_environment()` for automatic environment detection
    - **Global Variable Export**: Automatic export of results as global variables in interactive sessions (VSCode, Jupyter, IPython)
    - **User-Friendly Output**: Interactive mode displays formatted summary of available variables for immediate access
    - **Dual Compatibility**: Maintains full CLI functionality while adding interactive workspace integration

- **Code Structure Optimization:**
  - **src/dataeng_main.py**: Streamlined function architecture and configuration management
    - **Function Consolidation**: Removed redundant `entry_point()` method, consolidating all logic into enhanced `main()` function
    - **Default Parameter Integration**: Moved configuration defaults directly into `main()` function signature for cleaner API
    - **Automatic Configuration**: Smart defaults for outlier detection parameters and figure saving directories
    - **Path Management**: Intelligent `figs_save_dir` generation based on input data directory structure

- **Data Engineering Visualization Integration:**
  - **src/dataeng_main.py**: Added comprehensive LET profile plotting capabilities after outlier replacement
    - **LET Profile Visualization**: Integrated `plot_let_profile()` method from EDA pipeline for consistent visualization
    - **Dose Profile Overlay**: Automatic dose profile import and overlay on LET plots when available
    - **Bragg Peak Detection**: Intelligent subplot focusing using dose profile maximum or Bragg peak detection
    - **Subplot Configuration**: Added configurable subplot location (`[0.2, 0.35, 0.25, 0.50]`) with dynamic x-range based on dose maximum and primary particle stopping point
    - **Auto-Save Integration**: Automatic figure saving when `figs_auto_save=True` with descriptive filenames
    - **Enhanced Logging**: Comprehensive logging of plot generation, dose import status, and peak detection results

- **Preprocessing Module Consolidation:**
  - **core/preprocessing/preprocessing_utils.py**: Centralized dose/fluence processing functionality
    - **Function Migration**: Moved `import_and_process_dose_fluence()` from `eda_main.py` to preprocessing utilities
    - **Unified Interface**: Single source of truth for dose profile and fluence data import across all modules
    - **Enhanced Error Handling**: Robust error handling with graceful fallbacks for missing dose files
    - **Consistent API**: Standardized parameter structure (`data_dir`, `dose_file`, `voxel_in_um`, `zero_thr_percent`, `drop_zero_cols`, `verbose`)

#### 🔄 Refactored

- **Code Consolidation and Standardization:**
  - **src/eda_main.py**: Updated to use centralized preprocessing functions
    - **Import Updates**: Added `import_and_process_dose_fluence` to imports
      from `preprocessing_utils`
    - **Function Removal**: Removed local `import_and_process_dose_fluence()`
      definition to eliminate duplication
    - **Consistent Dependencies**: Now uses shared preprocessing module for all
      dose/fluence operations
  
  - **src/dataeng_main.py**: Complete refactoring of dose handling workflow
    - **Removed Custom Functions**: Eliminated custom `import_dose_profile()`
      function
    - **Standardized Implementation**: Now uses
      `import_and_process_dose_fluence()` from preprocessing module
    - **Enhanced Peak Detection**: Added `find_bragg_peak_start` import and
      implementation for accurate peak identification
    - **Intelligent Range Selection**: Prioritizes dose profile maximum for
      subplot x-range, with Bragg peak detection as fallback
    - **Improved Error Handling**: Comprehensive try-catch blocks with
      informative logging for dose import failures

#### 🛠️ Technical Improvements

- **Visualization Enhancement Architecture:**
  - **Dose Profile Integration**: Seamless integration of dose profiles with LET
    visualization for comprehensive analysis
  - **Peak Detection Logic**: Dual-method approach for identifying critical regions:
    1. **Primary Method**: Use dose profile maximum
       (`dose_profile['Dose(Gy)'].idxmax()`) for clinically relevant peak
       identification
    2. **Fallback Method**: Use Bragg peak detection (`find_bragg_peak_start()`)
       when dose data unavailable
  - **Dynamic Subplot Configuration**: Automatic calculation of subplot x-range
    based on detected peaks and particle stopping points
  - **Multi-Format Export**: Support for PNG and EPS format exports with proper
    DPI settings and descriptive filenames

- **Module Architecture Improvements:**
  - **Reduced Code Duplication**: Eliminated redundant dose processing
    implementations across modules
  - **Improved Maintainability**: Single location for dose/fluence processing
    logic enables easier updates and bug fixes
  - **Enhanced Testability**: Centralized functions are easier to unit test
    independently
  - **Consistent Interfaces**: Standardized parameter patterns across all
    modules using dose processing

#### 📈 Benefits

- **User Experience**: 
  - **Visual Feedback**: Users now see immediate visual confirmation of outlier
    replacement effects on critical Bragg peak region
  - **Clinical Relevance**: Dose profile overlay provides therapeutically
    relevant context to LET visualizations
  - **Automated Focus**: Intelligent subplot focusing eliminates manual
    region-of-interest selection

- **Developer Experience**:
  - **Code Reusability**: Centralized dose processing functions usable across
    entire codebase
  - **Debugging Efficiency**: Comprehensive logging enables easy troubleshooting
    of visualization and processing steps
  - **Maintenance Simplicity**: Single source of truth for dose processing
    reduces maintenance overhead

- **Technical Quality**:
  - **Robustness**: Graceful handling of missing dose files ensures core
    functionality remains available
  - **Flexibility**: Fallback mechanisms ensure visualization works regardless
    of data availability
  - **Consistency**: Standardized approach to dose/fluence handling across all
    analysis pipelines

### [refactor/shared-utils-structure] - 2025-08-21

#### 🔄 Refactored

- **Legacy Module Migration to Structured Utilities:**
  - **From `data_eng_pkg.data_processing` to `utils.outlier_detection_utils`:**
    - Migrated `identify_let_outliers()` with significant architectural
      improvements:
      - Split monolithic function into focused, single-purpose functions
      - Added private helper functions (`_identify_outliers_by_method`,
        `_compute_replacement_values`)
      - Enhanced error handling with comprehensive validation and logging
      - Streamlined interface with cleaner parameter structure
      - Added comprehensive type annotations and enhanced docstrings
    - Migrated `replace_outliers_in_df()` with simplified interface
    - Refactored helper functions into private methods:
      - `search_neighbors()` → `_compute_*_replacements()` functions
      - `nearest_neighbor_replacement()` →
        `_compute_nearest_neighbor_replacements()`
      - `local_statistic_outlier_replacement()` →
        `_compute_local_mean_replacements()`
      - `knn_regressor_replacement()` → `_compute_knn_regressor_replacements()`

  - **From `my_plot_pkg.plot_utils` to `utils.outlier_visualization_utils`:**
    - Migrated `plot_let_outliers_replacements()` →
      `plot_let_outliers_replacement()` (renamed and enhanced):
      - Added subplot support with outlier detail views
      - Enhanced plotting with connection lines between outliers and replacements
      - Integrated statistical summaries directly in plots
      - Improved figure management with multiple format saving support
    - Enhanced `plot_feature_and_mark_outliers_by_let()`:
      - Added configurable styling options (colors, markers, transparency)
      - Improved subplot functionality for detailed analysis regions
      - Enhanced visual indicators for outliers and replacements
    - **New visualization functions:**
      - `plot_outlier_detection_results()` - Comprehensive outlier analysis plots
      - Enhanced statistical visualization capabilities

  - **From `eda_module.data_analysis_utils` to `utils.data_analysis_utils`:**
    - Migrated `get_correlation_df()` with minimal changes:
      - Simplified imports and reduced dependencies
      - Maintained functionality while improving code organization
    - **Other functions moved to `core.preprocessing.preprocessing_utils`:**
      - `find_bragg_peak_start()` - Peak analysis functionality
      - Various data manipulation functions for better logical grouping

#### ✨ Enhanced

- **Code Organization Benefits:**
  - **Modular Architecture:** Functions logically grouped by purpose (detection vs visualization)
  - **Maintainability:** Smaller, focused functions easier to test and modify
  - **Reusability:** Clean interfaces allow functions to be used independently
  - **Standards Compliance:** Follows Python conventions for private functions and module organization

- **Performance Improvements:**
  - Optimized algorithms with reduced redundant calculations
  - Better memory management in outlier detection routines
  - Streamlined visualization rendering pipeline

- **Developer Experience:**
  - Enhanced documentation with detailed parameter descriptions
  - Comprehensive type hints throughout refactored modules
  - Improved error messages and logging capabilities
  - Better separation of concerns between modules

#### 🛠️ Technical Improvements

- **Outlier Detection (`utils.outlier_detection_utils`):**
  - Modularized detection algorithms (LOF, DBSCAN, upper limit thresholding)
  - Flexible replacement strategies (median, mean, local statistics, KNN
    regression)
  - Robust error handling with graceful fallbacks
  - Configurable visualization integration

- **Visualization (`utils.outlier_visualization_utils`):**
  - Enhanced plotting with statistical overlays and detailed analysis views
  - Configurable styling and layout options
  - Multi-format figure export (PNG, EPS) with proper DPI settings
  - Integration with filesystem utilities for automatic directory management

- **Data Analysis (`utils.data_analysis_utils`):**
  - Streamlined correlation analysis functionality
  - Reduced external dependencies while maintaining full compatibility
  - Clean import structure for better module loading

#### 📈 Migration Impact

- **Backward Compatibility:** All existing functionality preserved during migration
- **Performance:** Improved execution speed through optimized algorithms
- **Extensibility:** Modular design facilitates easy addition of new detection/replacement methods
- **Testing:** Isolated functions enable more comprehensive unit testing
  capabilities

### [refactor/shared-utils-structure]

#### ✨ Added

- **Data Engineering Module Modernization:**
  - **src/dataeng_main.py**: Complete migration from legacy module imports to
    structured architecture.
  - Migrated from `import data_eng_pkg as dataeng`, `import eda_module as eda`,
    `import my_plot_pkg as myplt`, `import preprocessing_module as preprocess`
    to new structured imports:
    - `from core.preprocessing import preprocessing_utils`  
    - `from utils.filesystem_utils import ensure_directory_exists`
    - `from utils.outlier_detection_utils import identify_let_outliers,`
      `replace_outliers_in_df`
    - `from utils.outlier_visualization_utils import`
      `plot_feature_and_mark_outliers_by_let`
  - **Enhanced Directory Management**: Added `ensure_directory_exists()` calls
    with comprehensive logging for figure directories.
  - **Improved Data Path Handling**: Replaced
    `eda.get_cut_voxel_from_filename()` with
    `preprocessing_utils.extract_cut_voxel()` for better path resolution and
    verbose logging.
  - **Enhanced Plotting Capabilities**: Updated plotting functions to support
    `save_fig` and `save_dir` parameters for automatic figure saving.
  - **Updated Default Configuration**: Modified default data directories to use
    current project structure (`'../data/thr96_1e8_v1um_cut1mm_ver_11-2-2'` and
    `'Let.out'`)

- **Plot Utilities Enhancement:**
  - Refactored `plot_more_dataframe()` from `src/my_plot_pkg` and added it to
    `src/utils/plot_utils.py`
  - Broke down monolithic 164-line function into 12 modular helper functions
    for better maintainability
  - Added comprehensive input validation with clear error messages
  - Leveraged existing `plot_twin_profile()` helper for consistent styling
  - Enhanced type safety with proper type hints throughout
  - Improved documentation with detailed docstrings
  - All functions comply with 80-character line limit requirement
  - Updated `src/utils/__init__.py` exports to include additional plot utility
    functions

- **Documentation:** New user/developer guide for `vae_optimization_analysis.py`
  covering structure, CLI usage, outputs, common tasks, and developer reference.
  Available as `wiki/vae_optimization_analysis_guide.md`.

- `wiki/vae_optimization_user_guide.md`: New user/developer guide detailing
  structure, CLI usage, outputs, and key functions of the `vae_optimization.py`
  script. Supports both CLI and interactive usage.

- Initial version of `wiki/vae_generate_guide.md` to document generation
  pipeline.
- Initial version of `wiki/vae_training_guide.md` to document training pipeline.

- `LatentSpaceMixin` in `utils/latent_utils.py` to encapsulate reusable latent
  space operations (extraction, projection, plotting, and saving).
- `LatentAware` protocol to statically define required attributes and methods
  for mixin consumers.
- Helper methods: `plot_and_save_latent_space` and
  `plot_and_save_multiple_latents` for streamlined latent visualization.

- Support for configurable `study_name` and `study_path` in
  `VAEOptimizer.create_study()` via OptunaStudyConfig.
- Addressed conflict where `study_path` and `use_timestamp_output_dir=True`
  could lead to inconsistency by realigning `study_path` dynamically.
- Added validation for proper Optuna SQLite URI format and `.db` extension in
  study paths.

- New configuration option: `use_timestamp_output_dir` (default: `True`) in
  `base_config.py` to control whether pipeline output directories include a
  timestamp suffix.

- Safeguard in `BaseVAEPipeline.model_summary()` to skip summary generation if
  `input_dim` is not set.
- Type check in `VAEOptimizer.get_network_hyperparams()` to raise an error when
  `input_dim` is undefined.

- New utility function: `create_data_loaders()` in `utils/data_loader_utils.py`
  for creating configurable train/val/test DataLoaders with per-split shuffle
  and type support.

- New module: `src/utils/plot_utils.py` for centralized visualization logic.
- `plot_training_metrics()` now externalized from BaseVAEPipeline with improved
  logging and flexibility.
- `plot_train_test_val_distribution_df()` moved from `my_plot_pkg` into `utils`,
  consolidating internal plotting tools.

- **Plot Utilities Enhancement:**
- `plot_training_metrics()` method to `BaseVAEPipeline` to support grouped
  training metric plots (e.g., train vs val loss).

- `BaseVAEPipeline` class in `src/core/base_pipeline.py` to centralize shared
  VAE training utilities: model saving/loading, logging, device setup, and
  training history management.

- `VAETrainer.save_model_state_dict()`: Saves the best model’s state_dict to a
  user-specified or default path.
- `VAETrainer.save_history()`: Writes the recorded training loss history to a
  `.json` file.
- `VAETrainer.save_training()`: Unified method that saves both model weights and
  training history to a specified output directory.
  - Supports optional file name overrides.
  - Automatically creates output directory if needed.
- All methods include structured logging and graceful handling of missing data.

- Finalized `vae_training.py` using the shared `VAETrainer` class pattern.
- Included evaluation logic for reconstruction and latent space projection.
- Added UMAP/TSNE projection support and flexible coloring via
  `extract_latents()`.

- Added `save_figure(fig, output_dir, filename)` to
  `src/utils/filesystem_utils.py`:
  - Saves a matplotlib figure as both PNG and EPS formats.
  - Ensures the output directory exists before saving.

- Extended `VAEDataPreprocessor` (`src/core/preprocessing/data_preprocessor.py`)
  to support both "training" and "inference" modes:
  - "training" mode handles full data cleaning, splitting, and scaling.
  - "inference" mode enables input downsampling and separate scaling for
    generation tasks.
- Added strict validation of the `mode` argument during preprocessor
  initialization.
- Centralized downsampling and preprocessing logic within the class.

- Created `src/core/preprocessing/` directory for modular preprocessing
  components.
- Added `preprocessing_utils.py`:
  - Centralized reusable functions for data import, cleaning, voxel extraction,
    and column dropping.
  - Integrated consistent logging across preprocessing operations.
- Added `data_preprocessor.py`:
  - Implemented `VAEDataPreprocessor` class to load, preprocess, split, and
    scale datasets for VAE training and evaluation.
  - Supports configurable scalers, random seeds, and identity feature
    reordering.
- Modularized preprocessing code to prepare for future refactoring of data
  pipelines.

- Integrated `log_params_dict()` for structured logging of network, optimizer,
  and loss hyperparameters in optimization trials to `vae_optimization.py`.

- Added `log_params_dict()` to `utils/logger.py`:
  - Logs parameter dictionaries under a structured header with optional scoped
    logger
  - Useful for debugging configs, hyperparameters, and metrics across scripts

- Added `src/core/training_utils.py`:
  - Contains `create_optimizer()` for flexible training setup
  - Contains `build_loss_fn()` supporting multiple loss strategies (standard,
    inverse, sigmoid)

- Added `__init__.py` files to `core/` and `core/models/` to expose clean import
  interfaces:
  - e.g., `from core.models import AutoEncoder, ELUWithLearnableOffset`
  - e.g., `from core import build_vae_model_from_params`

- Added configuration module in `src/configs/`:
  - `base_config.py`: defines shared `BaseVAEConfig` dataclass
  - `task_config.py`: includes specialized config classes for training,
    generation, Optuna studies, and analysis
  - `optimization_analysis_config.json`: example config for post-optimization
    analysis
  - `optuna_config.json`: config for study optimization workflow with detailed
    training and preprocessing options
  - `post_training_config.json`: config for post-training evaluation with model
    and history references

- Added `config_loader.py` utility:
  - Provides `load_config_from_json()` to parse JSON into typed dataclass
    configs
  - Includes logging, error handling, and type hooks (e.g., auto-casting strings
    to `Path`)

- Added `src/utils/study_utils.py` with reusable Optuna utilities:
  - `load_optuna_study()` for safe study loading with logging and error handling
  - `summarize_best_trial()` for consistent logging of best trial parameters and
    user attributes

- Added `src/utils/__init__.py` with named exports:
  - `VAELogger`, `load_model`, and `load_model_config`
  - Enables simplified imports: `from utils import load_model`

- `VAELogger` class in `src/utils/logger.py` for centralized, extensible logging
  across the VAE pipeline.
  - Supports dynamic log levels and prefix tagging for contextual logging.
  - Replaces get_logger() with class-based instantiation and method wrapping.

#### ♻️ Changed

- **src/eda_main.py**
  - **EDA Module Refactoring**: Completely migrated from legacy `eda_module` to structured core and utils modules
  - Replaced `eda.drop_zero_columns()` with `drop_zero_columns` from `core.preprocessing.preprocessing_utils`
  - Replaced `eda.get_cut_voxel_from_filename()` with `extract_cut_voxel()` from `core.preprocessing.preprocessing_utils`
  - Replaced `eda.find_bragg_peak_start()` with `find_bragg_peak_start` from `core.preprocessing.preprocessing_utils`
  - Replaced `eda.get_correlation_df()` with `get_correlation_df` from `utils.data_analysis_utils`
  - **Imports**: Removed dependency on `eda_module`, now uses properly structured imports from core and utils

- **src/core/preprocessing/preprocessing_utils.py**
  - **Function Migration**: Added `find_bragg_peak_start()` function for Bragg peak analysis
  - **Logical Grouping**: Now contains all peak-related functions (`find_bragg_peak_start` and `find_min_after_peak_index`) in one module
  - **Type Safety**: Added `Union` import to support flexible input types for peak analysis functions

- **src/utils/data_analysis_utils.py**
  - **New Module**: Created dedicated module for data analysis utilities
  - **Function Migration**: Added `get_correlation_df()` function for correlation matrix calculations
  - **Clean Interface**: Provides specialized analysis functions separate from preprocessing utilities

- **src/utils/__init__.py**
  - **Export Updates**: Added `get_correlation_df` to module exports for clean import interface

- **src/vae_post_analysis.py**
  - Renamed hyperparameter configuration parameter from `config_path` to
    `hparams_config_path` to avoid confusion with CLI `--config_path` argument.
  - Updated internal call to `load_model_config` to use
    `self.config.hparams_config_path`.
  - Edited `main()` function for correct execution from both CLI and interactive
    environments.
  - Removed outdated reference to `optuna_summary.txt` in user/developer documentation.

- **configs/post_training_config.json**
  - Updated `post_training_config.json` to replace `config_path` with
    `hparams_config_path`.

- **src/configs/task_config.py`**
  - Updated `PostTrainingConfig` dataclass to define `hparams_config_path`
    instead of `config_path` for hyperparameter configuration.

- **src/vae_optimization_analysis.py**
  - Enhanced CLI entry point with support for both script and interactive usage
    (e.g., Jupyter).
  - Added exception handling for configuration loading failures.

- **configs/optimization_analysis_config.json**
  - Removed unused configuration parameters (`features_to_plot`, `data_dir`,
    `weights_path`, etc.).
  - Retained only essential keys: `study_name`, `study_path`, and `plot_dir`.

- `src/vae_optimization.py`: Enhanced the `main()` function to support both
  command-line and interactive (e.g. Jupyter, VSCode) execution by checking for
  `get_ipython`. This improves flexibility during development and
  experimentation.

- `src/vae_generate.py`: Refactored `plot_input_and_generated()` method in
  `VAEGeneratorAnalysis` for safe feature list handling.
- `src/vae_generate.py`: Linked plotting behavior to `features_to_plot` value
  from the config file (`generation_config.json`) instead of hardcoded values.
- `src/vae_generate.py`: Updated `main()` to pass `features_to_plot` from config
  when generating plots.

- `src/vae_training.py`:
  - Refactored `main()` to support both CLI and interactive execution:
    - Accepts `--config_path` via argparse
    - Defaults to `configs/trainer_config.json` when run in Jupyter or VS Code
      Interactive
  - Improved type hints using `Optional[Union[str, Path]]` for config path
    flexibility

- Refactored `VAETrainer` in `vae_training.py` to use `LatentSpaceMixin` for
  shared latent space logic.
- Removed duplicated `extract_latents()` and `plot_latent_space()` methods from
  `VAETrainer`.
- Introduced support for plotting and saving latent space projections via
  mixin-provided methods.
- Ensured structural compatibility with `LatentAware` protocol by adding
  required attributes.

- Refactored `VAEPostAnalyzer` to use `LatentSpaceMixin`, removing redundant
  implementations of `extract_latents()` and `plot_latent_space()`.

- `create_study()` now defaults to `{output_dir}/optuna_study.db` if no custom
  study path is provided.
- Updated `VAEOptimizer.__init__()` to retain the original parent directory in
  `self.output_dir_parent` for use in conflict resolution.
- Automatically updates `config.study_path` when overridden to align with the
  timestamped output directory.

- Refactored `VAEPostAnalyzer.__init__()` to always assign
  `self.data_components` as a dictionary (output of
  `load_and_preprocess_data()`), eliminating ambiguity and fixing invalid
  `__getitem__` access warnings.
- Updated `plot_training_history()` in `VAEPostAnalyzer` to use `Optional[dict]`
  for better static typing and improved fallback handling.

- Updated `plot_reconstruction()` method in `VAETrainer` to accept
  `Optional[List[str]]`, improving static typing compatibility.
- Fixed LaTeX escape sequence in `set_ylabel()` by using a raw string format
  (`rf""`) for Pylance compliance.
- Replaced `from umap import UMAP` with `from umap.umap_ import UMAP` for
  correct usage in environments with strict import resolution.
- Modified `load_config_from_json()` in `config_loader.py` to accept both `str`
  and `Path` for `config_path`, with automatic path normalization.

- Updated `VAEOptimizer` and `VAETrainer` to respect `use_timestamp_output_dir`
  and create cleaner or consistent output paths when disabled.

- Refactored Optuna plot handling in `vae_optimization_analysis.py` to safely
  handle missing figures or axes.
- Fixed NumPy-to-Python float compatibility issue when passing a cap value to
  `_create_capped_study()`.

- Refactored `VAEOptimizer` in `vae_optimization.py` to inherit from
  `BaseVAEPipeline`, consolidating model lifecycle behavior.
- Removed redundant logic for logging, model saving, device setup, and output
  directory creation.
- Now uses `model_summary()`, `save_model_state_dict()`, and `save_history()`
  methods provided by the base class.

- Refactored `VAEOptimizationAnalyzer` in `vae_optimization_analysis.py` to no
  longer inherit from `BaseVAEPipeline`, simplifying its responsibility to
  Optuna post-analysis only.

- Refactored `vae_training.py`, `vae_generate.py`, and `vae_post_analysis.py` to
  use `create_data_loaders()` instead of repeated loader setup code.
- Updated `vae_optimization.py` objective function to use
  `create_data_loaders()` for cleaner and more consistent training/validation
  loader construction.

- `VAEOptimizationAnalyzer` now inherits from `BaseVAEPipeline` to streamline
  pipeline lifecycle behavior.
- Replaced custom `model_summary()` with the shared implementation from the base
  class.

- Updated `vae_training.py` to import plotting utilities from
  `utils.plot_utils`.

- Updated `vae_training.py` to use `plot_training_metrics()` instead of its own
  `plot_training_history()` implementation.
- Refactored `VAEPostAnalyzer` in `vae_post_analysis.py` to inherit from
  `BaseVAEPipeline` and reuse its plotting and logging features.
- Consolidated history plotting logic in `VAEPostAnalyzer` to leverage shared
  base functionality.

- Refactored `VAETrainer` in `vae_training.py` to inherit from
  `BaseVAEPipeline`.
- Removed redundant implementations for model saving, logging, and history
  export from `VAETrainer`.

- Replaced procedural training logic with a modular, class-based interface
  (`VAETrainer`).
- Adopted all shared utilities: `VAEDataPreprocessor`, `build_loss_fn`,
  `create_optimizer`, `VAELogger`, and `save_figure`.
- Improved logging, plotting, and output management consistency across training
  runs.

- Updated `encode_latent()` in `AutoEncoder` to return `(mu, logvar)` explicitly
  and reflect correct type annotation.
- Refactored `reconstruct()` method to:
  - Split input into processed and identity features, if applicable.
  - Concatenate identity features after decoding for full reconstruction.
  - Return both reconstructed data and latent mean vectors (`mu`) for later use.
- Added assertion to validate input dimensionality in `reconstruct()` to catch
  configuration/data mismatch issues.
- Improved method docstrings and added type hints for consistency and developer usability.

- Refactored `fit()` method of the `AutoEncoder` class to take
  `val_loss_threshold` as a standalone argument rather than reading it from
  `loss_hyperparams`.
- Updated the Optuna optimization pipeline to remove `val_loss_threshold` from
  the `loss_hyperparams` dictionary.
- Improved separation of concerns by decoupling validation threshold logic from
  loss function construction.

- Standardized optimizer parameter keys to:
  - `"optimizer"` (formerly `"optimizer_type"`)
  - `"learning_rate"` (formerly `"learning_rate"`)
  - `"weight_decay"` (formerly `"l2_penalty"` or `"l2_reg"`)
  - These keys are now compatible across both `AutoEncoder` and
    `create_optimizer()`.

- Made `optimizer_type` lookup case-insensitive in `create_optimizer()` (in
  `training_utils.py`), allowing more flexible config inputs like `"Adam"` or
  `"adam"`.

- Modified `plot_train_test_val_distribution_df` in
  `src/my_plot_pkg/plot_utils.py` to return the generated
  `matplotlib.figure.Figure` object instead of displaying or saving internally.
- Improves reusability by allowing external scripts to control figure rendering
  and export (e.g., via `save_figure()`).

- Replaced absolute configuration file paths with relative paths in:
  - `vae_generate.py`
  - `vae_optimization.py`
  - `vae_optimization_analysis.py`
  - `vae_post_analysis.py`
- This improves machine portability and enables consistent execution across
  environments.
- Scripts now resolve config files relative to the repository structure
  (`src/configs/`), avoiding machine-specific paths.

- Refactored `vae_generate.py` to use the shared `VAEDataPreprocessor` from
  `core.preprocessing.data_preprocessor`.
- Removed redundant local preprocessing classes and replaced them with modular,
  reusable utilities.
- Updated `vae_optimization.py` to use the centralized `VAEDataPreprocessor`.
- Standardized dataset preparation across optimization and generation pipelines.

- Refactored figure saving in main scripts (e.g.,
  `vae_optimization_analysis.py`, `vae_post_analysis.py`, etc.) to use the
  shared `save_figure` utility.
- Reduced redundant `_save_figure` method definitions across the project.
- Standardized figure export formatting (tight bounding box, 300 DPI for EPS).

- Refactored `VAEDataPreprocessor` to improve flexibility, robustness, and
  logging consistency.
- Prepared for full integration of shared preprocessing utilities across
  training, post-analysis, and generation workflows.

- Replaced local `VAEDataPreprocessor` in `vae_post_analysis.py` with shared
  version from `core.preprocessing.data_preprocessor`.
- Updated data preparation in post-analysis to use standardized, centralized
  logic.

- Refactored `vae_optimization.py` to:
  - Inject `opt` and `loss_fn` into `AutoEncoder` using new `set_optimizer()`
    and `set_loss_function()` methods
  - Log hyperparameter blocks via shared `log_params_dict()` utility
- Updated `AutoEncoder` class to support:
  - Clean logger initialization with optional prefixing
  - External assignment of optimizer and loss function via setter methods
- Refactored `model_builder.py` to remove internal loss/optimizer setup
- Refined `training_utils.py` to centralize and standardize optimizer/loss
  creation logic

- Refactored `vae_optimization.py`:
  - Uses `create_optimizer()` and `build_loss_fn()` from `training_utils.py`
  - Injects optimizer and loss function into `AutoEncoder` externally
  - Improves separation of concerns for training and model architecture

- Refactored `AutoEncoder` in `core/models/autoencoder.py`:
  - Removed internal construction of optimizer and loss function
  - Model now accepts `opt` and `loss_fn` via constructor or external assignment
  - Enables more flexible integration with training loops and Optuna workflows

- Refactored `AutoEncoder` in `core/models/autoencoder.py`:
  - Delegates optimizer and loss function creation to shared `training_utils`
  - Improves modularity and Optuna compatibility

- Restructured model architecture under `src/core/`:
  - Moved VAE model classes to `core/models/autoencoder.py`
  - Extracted reusable activation functions to `core/models/activation.py`
  - Moved MLP layer-building utilities to `core/models/helpers.py`
  - Added `core/losses.py` for custom training/evaluation losses
  - Centralized model instantiation logic in `core/model_builder.py`

- Updated `vae_optimization_analysis.py`:
  - Now loads configuration dynamically using `load_config_from_json()`
  - Supports externalized JSON configs (e.g.,
    `optimization_analysis_config.json`)

- Updated `vae_post_analysis.py`:
  - Refactored to use shared `VAELogger` for consistent and contextual logging
  - Uses shared `load_model()` and `load_model_config()` from `model_io`
  - Loads Optuna studies using `load_optuna_study()` from `study_utils`
  - Parses configuration via `load_config_from_json()` to enable dataclass-based
    setup

- Refactored `load_model()` in `model_io.py`:
  - Supports flexible input via `params` or `config_path`
  - Logs and raises clear errors when config loading fails or inputs are missing

- Updated `vae_optimization_analysis.py` to:
  - Use `load_optuna_study()` and `summarize_best_trial()` from `study_utils`
  - Improve modularity and reduce code duplication across the pipeline

- Enhanced `load_model_config()` in `model_io.py`:
  - Added error handling for missing files and malformed JSON.
  - Logs success and failure messages using `VAELogger`.

- Improved `load_model()` in `model_io.py`:
  - Added input validation for required hyperparameters.
  - Added error handling for weight loading errors.
  - Supports optional `params` argument to avoid redundant config loading.

- Updated `vae_optimization_analysis.py`:
  - Now uses `VAELogger` and utility functions from `utils` package.
  - Benefits from cleaner logs, reusable config loading, and reduced duplication.

- Refactored `vae_optimization_analysis.py` to:
  - Use the new VAELogger for improved logging structure and modularity.
  - Log model loading, study processing, and analysis steps with tagged messages.
  - Adopt consistent logging format across modules.

#### 🔧 Fixed

- Resolved Pylance type inference issues related to method access and structural
  typing by explicitly typing protocols and return values.

- Resolved Pylance warning in
  `VAEGeneratorAnalysis.compute_mean_variance_difference()` by explicitly
  casting variance values to `float` after handling potential NaNs from
  `pandas.Series.var()`.

- Ensures type-safe subtraction operations and reliable metric reporting when
  working with sparse or edge-case input data.

- Resolved Pylance warnings caused by unsafe use of `dict = None` and `key not
  in params` checks in `load_model()` by enforcing a non-None check before
  accessing parameters.

- Pylance type warning caused by passing `np.floating` instead of `float`.

#### 📦 Structural

- Refactored `AutoEncoder` (in `core/models/autoencoder.py`):
  - Now supports receiving an `opt` argument as either a `torch.optim.Optimizer`
    instance or a dictionary.
  - If a dictionary is provided, the optimizer is constructed using
    `create_optimizer()` and assigned internally.

- Refactored `build_vae_model_from_params()` in `model_builder.py`:
  - Now constructs the model architecture only (optionally optimizer, no loss
    assignment).
  - Updated docstring to reflect the builder’s limited scope.
  - Ensures clearer separation of construction and training responsibilities.

- Improved project modularity by centralizing preprocessing workflows.
- Ensured consistency between training, validation, inference, and generation
  steps using shared utilities.

- Moved `src/ml_prep/data_utils.py` to `src/core/preprocessing/data_utils.py`:
  - Consolidates all dataset splitting, scaling, and inverse transforms into the
    preprocessing module.
  - Enables easier maintenance and unified access to preprocessing utilities.

### Added

- **[generative-workflow] - 2025-07-28**
  - CLI argument parsing with `argparse` for specifying config path via
    `--config`.
  - Type hints for `main(config_path: Optional[Union[str, Path]] = None)` to
    improve clarity and tool support.
  - Support for interactive execution via direct function call with
    parameterized config path.
  - Dual-mode execution block under `if __name__ == "__main__"` for
    compatibility with both command-line and VSCode Interactive Window.

- **[generative-workflow] - 2025-04-16**
  - New post-generation evaluation methods inside `VAEGeneratorAnalysis`:
    - `compute_wasserstein()`: Earth Mover's Distance
    - `compute_ks_test()`: Kolmogorov-Smirnov test
    - `compute_mean_variance_difference()`: Mean and variance comparison
    - `evaluate_generated()`: Aggregates all metrics into a summary table.

- **[generative-workflow] - 2025-04-15**
  - Support for functional scaler definitions with parameterized control (e.g.,
    `StandardScaler(with_mean=False)`).
  - Latent space interpolation and high-resolution output generation pipeline.
  - Output comparison plotting and gap filler evaluation via VAE performance
    metrics.

- **[generative-workfolw] - 2025-04-14**
  - Support for functional scaler configuration in `VAEOptimizationConfig`.
    - Accepts either string (e.g., `"standard"`) or dictionary format:

      ```python
      {"type": "standard", "with_mean": False}
      ```

- **[post_analysis] - 2025-04-10**
  - Training history plotting via `plot_training_history()`, showing loss trends
    and logging total training time.
  - `_load_history()` method with file existence and JSON parsing checks.
  - Support for inverse transformation and plotting of reconstructed features.

- **[optuna_optimization] - 2025-04-10**
  - Save training history (`history` dictionary) for the best-performing Optuna
    trial.
  - Logged tracked trial number and training metrics to aid reproducibility.

  - Introduced new hyperparameter `use_dropout` (categorical: [True, False]):
    - When set to `False`, `dropout_rate` is automatically overridden to 0.0.
    - Both values are tracked in trial user attributes and passed to the model
      configuration.

  - Configuration flags (`use_default_*`) for key hyperparameters:
    - `use_dropout`
    - `use_exit_activation`
    - `use_negative_penalty`
  - If flags are enabled, default values are enforced instead of being suggested
    by Optuna.
  - All final values (forced or suggested) are logged and stored via
    `trial.set_user_attr()`.
  - Enhanced reproducibility and control over optimization behavior from the
    configuration layer.
  - Added structured logging and metadata capture for better post-study
    analysis.

  - New method `log_best_trial_hyperparameters()` in `VAEOptimizationAnalyzer`:
    - Logs best trial number and objective value.
    - Prints all suggested hyperparameters and user-defined attributes.
    - Provides a clear overview of the best model configuration after
      optimization.

- **[optuna_optimization] - 2025-04-09**
  - Enhanced model saving logic during Optuna optimization:
    - Track best validation loss and save the `state_dict()` of the best-trained
      model using `deepcopy`.
    - Save best model weights to `best_model_weights_trial_<n>.pth`.
    - Log the trial number of the best-performing model.
    - Ensure no untrained models are saved by checking that weights exist.

- **[optuna_optimization] - 2025-04-08**
  - Integrated dropout support in VAE encoder architecture:
  - Added `dropout_rate` to `concat_lin_layers` in `helpers.py`
  - Enabled `dropout_rate` parameter in `autoencoder.py` for encoder layers
    only.

  - New standalone script `vae_optimization_analysis.py` to perform Optuna study
    analysis independently of model evaluation.
    - Includes visualization of optimization history, parameter importance, EDF,
      timeline, slice plots, and contour plots.
    - Supports optional capping for contour plots using 95th percentile of
      objective values.
    - Generates high-res `.png` and `.eps` figures for reports.
    - Logs model summary after loading best checkpoint and hyperparameters.

- **[optuna_optimization] - 2025-04-03**
  - New methods `plot_timeline()`, and `plot_edf()` to VAEOptimizationAnalyzer:
    - Plots trial timeline using matplotlib.
    - Plots Empirical Distribution Function (EDF) to show the cumulative
      distribution of the objective values.

- **[optuna_optimization] - 2025-04-02**
  - `plot_contour` method to generate contour plots for all parameter pairs
    using Matplotlib.
  - Automatic z-value capping based on the 95th percentile of objective values
    via `cap_z` flag.
  - `_create_capped_study` utility to clone study with capped objective values
    for plotting compatibility.
  - Automatic logging of model config and dynamic class name integration in
    logs.
  - Support for:
    - Torch model summaries using `torchinfo`.

- **[optuna_optimization] - 2025-04-01**
  - New `vae_post_analysis.py` script for post-training model analysis.
  - Modular class structure:
    - `VAEPostAnalysisConfig`, `VAEPostAnalysisLogger`, `VAEPostAnalyzer`
    - `VAEDataPreprocessor` for test data loading
  - Optuna visualizations:
    - Optimization history (log-scaled)
    - Parameter importance
    - Slice plots (grouped by importance, saved individually)
  - Plot output in both `.png` and high-resolution `.eps` formats.

- **New `vae_builder.py` Module**
  - Added **`vae_builder.py`** in `vae_module/` to **encapsulate model**
    **construction**:
    - `build_vae_model_from_params()` now builds a **VAE model from a full**
      **hyperparameter dictionary**.
    - Ensures **cleaner separation between hyperparameter optimization and**
      **model instantiation**.
    - Allows easy **reconstruction of the best model** after optimization.
  - Updated **`vae_module/__init__.py`** to include `vae_builder.py`.

- Created `src/vae_module/losses.py` to modularize loss functions:
  - Moved `CustomLossWithNegPenalty` from `autoencoder.py` to this new file.
  - Introduced two new dynamic KL loss variants:
    - `InverseBetaLoss`: scales β as `β = beta_scale / (recon_loss + ε)`
    - `SigmoidBetaLoss`: computes β with a sigmoid function centered around a
      target recon loss.
- Added `_build_loss_fn()` helper method to `AutoEncoder` class for clean loss
  function selection.
- Integrated loss function configuration (`loss_type`, `beta_scale`,
  `recon_target`, etc.) into Optuna optimization.
- Introduced helper function `log_vae_config()` in `vae_optuna.py` to log model,
  loss, and optimizer configuration.

- Introduced `src/vae_module/activations.py` module:
  - Includes modular custom activations: `ShiftedSoftplus`,
    `ELUWithLearnableOffset`, and `PELU`.
  - Provides reusable and tunable output activations for the decoder.

- Integrated tunable output activations into `vae_optuna.py`:
  - Supports `ShiftedSoftplus`, `ELUWithLearnableOffset`, and `PELU` with
    parameter tuning.
  - Selected via Optuna with associated init parameters (`beta_init`,
    `offset_init`, `a_init`, `b_init`).
  - Activation configuration stored in trial metadata.

- Support for tunable decoder activation functions via Optuna
  (`ShiftedSoftplus`, `ELUWithLearnableOffset`, `PELU`).
- Recorded activation configurations and their hyperparameters using Optuna's
  user attributes.
- Post-training validation of reconstructed data: checks for negative values
  (below tolerance) with detailed per-feature logging.

- Introduced `vae_performance_metrics.py` in `src/`:
  - Implements evaluation metrics for generative model performance.
  - Includes:
    - **Reconstruction Metrics:** MAE, RMSE, Pearson Correlation.
    - **Generative Quality Metrics:** Wasserstein Distance, KS-Test, Mean &
      Variance Difference.
    - **Evaluation Functions:** `evaluate_reconstructed()` and
      `evaluate_generated()`.

- Created a new utility module `src/vae_module/activations.py` for activation
  functions.

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

- **Decoder Final Layer Bias:**
  Initialized the final layer bias to -2.0 in the decoder when
  `use_final_activation` is enabled. This change helps drive the pre-activation
  values into a more negative range, allowing the Softplus activation to output
  values closer to zero and mitigating the lower clamp issue.  

- **Custom Loss with Negative Penalty:**  
  Added a new loss function `customLossWithNegPenalty` that applies an extra
  penalty on negative outputs in the processed (non-identity) features. This
  loss function now optionally computes the loss only on the processed part when
  `proc_dim` is specified, ensuring that identity features are not penalized.

- **Added vae_module/autoencoder.py:**
  - This new file include an updated version of the AutoEncoder class to
    integrate the new `customLossWithNegPenalty` in both training and validation
    steps.  
  - Extended the input arguments of the `fit` method to accept a dictionary
    `loss_hyperparams` and then create the loss function accordingly, within the
    `train_step`.

- **Dedicated Data Engineering Workflow:**  
  Refactored `dataeng_main.py` to focus solely on data import, cleaning, and
  outlier handling by removing all exploratory data analysis (EDA) and plotting
  code that was present in the previous `eda_main.py` version.

- **Primary Particle Naming Consistency:**  
  Added logic to ensure the primary particle identifier ends with `"_1"` (using
  `primary_search_pattern`), avoiding redundant suffixes.
  `primary_search_pattern`), avoiding redundant suffixes.
- **Improved Documentation:**  
  Updated docstrings and inline comments to better explain the functions and
  data engineering workflow.

### Changed

- **[generative-workflow] - 2025-07-28**
  - Replaced hardcoded config path with dynamic loading based on user input or
    default.
  - Improved structure of `main()` for better modularity and testability.

- **[generative-workflow] - 2025-04-16**
  - Made the validation loss threshold for penalizing unstable trials
    configurable via `val_loss_threshold` in `loss_hyperparams` (AutoEncoder
    `fit()` method).

  - Updated `plot_latent_space()` method in `VAEPostAnalyzer`:
    - Added support for dimensionality reduction using PCA, t-SNE, and UMAP.
    - Includes informative logging and fallback handling if UMAP is not
      installed.
    - Saves 2D latent embeddings with method-specific filenames for easy
      reference.

- **[generative-workflow] - 2025-04-15**
  - Reimplemented `vae_generate.py` in a fully object-oriented design:
    - Introduced `VAEGenerateConfig`, `VAEDataPreprocessor`, `VAEGenerate`, and
      `VAEGeneratorAnalysis` classes.
    - Modularized data loading, scaling, model inference, and post-analysis.
    - Added consistent logger integration with class-level prefixes.
    - Latent space interpolation now uses the latent mean (no reparameterization
      sampling) for reproducibility in generation.

- **[generative_workflow] - 2025-04-14**
  - `split_and_scale_dataset()` now propagates scaler configuration parameters
    to `train_val_test_scale()`.
  - `train_val_test_scale()` dynamically applies parameters to `StandardScaler`,
    `MinMaxScaler`, `QuantileTransformer`, and `PowerTransformer`.
  - Improved flexibility in defining preprocessing pipelines via config
    structure.

- **[generative_workflow] - 2025-04-10**
  - Refactored `get_network_hyperparams()` to enforce structured and scalable
    layer design:
    - `layer_0_size` now depends on `processed_dim` and grows proportionally
      with input complexity.
    - Subsequent layers taper in size to promote compression.
    - `latent_dim` is constrained to be ≤ `processed_dim` to ensure consistent
      encoding.

- **[post_analysis] - 2025-04-10**
  - Enhanced logger behavior for clearer message formatting and Optuna verbosity
    suppression.
  - Configuration and logger setup refactored for cleaner instantiation and
    reusability.

- **[optuna_optimization] - 2025-04-10**
  - Conditional logic to disable negative penalty loss when using `minmax`
    scaler:
    - `use_neg_penalty` is forcibly set to `False` and `neg_penalty_weight` to
      `0.0`.
    - Ensures consistent configuration and avoids unnecessary penalty term when
      inputs are strictly non-negative.

  - Replaced old optimization pipeline `src/vae_optuna.py` with the finalized
    version `src/vae_optimization.py`:
    - Improved configurability, logging, model saving, and training history
      tracking.
    - File name changed to reflect its purpose more clearly.

- **[optuna_optimization] - 2025-04-08**
  - Updated `vae_optuna_claude.py`:
    - Included `dropout_rate` as a tunable hyperparameter in the Optuna study
    - Applied dropout only to encoder (not decoder) to preserve generative
      quality.

  - Modularized analysis pipeline by separating optimization metrics from latent
    space and reconstruction tasks.

- **[optuna_optimization] - 2025-04-07**
  - Renamed `output_activation`, and `use_out_activation` to
    `exit_activation_type`, and `use_exit_activation` in:
    - `vae_module/autoencoder.py`
    - `vae_module/helpers.py`
    - `vae_optuna_claude.py`
  - Renamed `model_config` to `hyperparameters_config` in `vae_optuna_claude.py`
    for improved clarity.
  - Moved device setup responsibility from `VAEDataPreprocessor` to
    `VAEOptimizer` for better cohesion.
  - Added `input_dim` and `process_dim` to `trial` metadata using
    `trial.set_user_attr()` to aid post-analysis and reproducibility.

  - Adjusted `latent_dim` search space in Optuna to depend on `net_size[-1]`
    with step size 4 for tighter architectural alignment.
  - Updated dataset preparation in `VAEOptimizer` to use
    `split_and_scale_dataset()` with:
    - `scaler_type='minmax'`.

- **[optuna_optimization] - 2025-04-02**
  - Split analysis responsibilities into `VAEOptimizationAnalyzer` and
    `VAEPostAnalyzer`.

- **Optimization Pipeline Bug Fixes (`vae_optuna_claude.py`):**
  - Finalized and debugged `vae_optuna_claude.py`, the refactored optimization
    script.
  - Fixed issue where `logger` used in `VAETrialHandler` was not a native logger
    instance.
  - Ensured debug-level logging is enabled and properly displayed for all
    hyperparameter configurations.
  - Merged `trial.params` with `trial.user_attrs` for correct model
    reconstruction in `rebuild_model()`.
  - Improved logger configuration to prevent duplicate handlers and support both
    script and notebook contexts.

- **Dedicated Post-Optimization Evaluation Script**
  - **Planned:** A new script (`vae_post_analysis.py`) to **evaluate and**
    **visualize** the best trained model.
    - **This avoids running evaluation within `vae_optuna_claude.py`**, keeping
      it optimized for batch execution.
    - The evaluation script will:
      - **Reload the dataset** and **re-split using the same `random_seed`**
        (ensuring the test set remains identical).
      - **Load the best model** from saved hyperparameters.
      - **Run performance metrics and visualization** for model evaluation.
  - **`vae_post_analysis.py` will be implemented after validation of the new**
    **optimization pipeline.**

- **Refactoring the Optimization Pipeline**
  - **`vae_optuna_claude.py`**: 
    - Introduced a **refactored version of the VAE optimization pipeline**,
      improving **modularity**, **logging**, and **batch system compatibility**.
    - Code now follows **object-oriented design**, separating:
      - **Hyperparameter optimization (`VAEOptimizer`)**
      - **Logging (`VAEOptimizationLogger`)**
      - **Model handling (`VAETrialHandler`)**
    - **This refactored script will replace `vae_optuna.py` after testing &**
      **debugging.**

- **vae_optuna.py updates**:
  - Updated VAE model creation and optimization to support custom activations
    from `vae_module.activations`.
  - Improved reproducibility and tracking with structured trial attributes.

  - Modular activation selection logic using `functools.partial` to pass preset
    parameters.
  - Improved training configuration logging for model, optimizer, and loss
    settings.
  - Updated handling of `output_activation` logging for better transparency.

  - Decoder activation selection now respects `use_out_activation` correctly:
    - Fixed a bug where `ELUWithLearnableOffset` was included even when
      `use_out_activation=False`.
    - Resolved by making `output_activation_layer=None` the default and
      conditionally instantiating the layer inside `Decoder`.
  - Upgraded logging configuration in `autoencoder.py` to allow DEBUG-level logs
    independently of the global logging level.
  - Enhanced `vae_optuna.py` to support conditional activation and loss type
    configuration via Optuna search space.

- **Loss function now applies negative output penalty in scaled space**:
  - Avoids numerical instability from inverse-scaling during training.
  - Ensures model outputs remain non-negative after inverse transform.
- `CustomLossWithNegPenalty` in `autoencoder.py` updated to compute penalty
  based on scaled zero threshold derived from `scaler.mean_` and
  `scaler.scale_`.
- `vae_optuna.py` updated to pass new loss hyperparameters (`scaler_mean`,
  `scaler_scale`, `proc_dim`) to the model during optimization and final
  training.

- Updated `autoencoder.py` to use `ELUWithLearnableOffset` as decoder output
  activation.
- Enabled clean switching between custom activations for improved numerical
  control.

- **`vae_performance_metrics.py`**:
  - Debugged `compute_ks_test()` function.

- **`vae_generate.py`**:
  - Removed redundant call to `.reconstruct()`; now reconstruction is extracted
    from interpolated latent decoding.
  - Cleanly separated reconstructed low-resolution data from newly generated
    samples using `RESAMPLE_FACTOR`.
  - Enhanced logging and visualization of LET spatial distributions.

- **Updated `vae_generate.py` to integrate `vae_performance_metrics.py`:**
  - Calls `evaluate_reconstructed()` to assess model reconstruction quality.
  - Uses `evaluate_generated()` to compare generated and real data distributions.
  - Outputs structured performance reports using `tabulate` for readability.
  
- **Generative Workflow Script Update:**
- Improved visualization:
  - Modified the **plot title** to include `RESAMPLE_FACTOR` dynamically.
  - Changed x-axis label from `"Sample Index"` to `"x [mm]"`.
  - Changed y-axis label from `"LET [units]"` to `"LTT [keV μm⁻¹]"` for improved
    clarity.

- **Generative Workflow Script Update:**
- Refactored `vae_generate.py` to use a single global variable `RESAMPLE_FACTOR`
  for both downsampling and upsampling.
- Removed redundant manual assignments of `downsample_factor` and
  `upsample_factor`, ensuring consistent resampling behavior.
- Improved code maintainability by centralizing resampling factor control.

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

- Updated `split_and_scale_dataset` to include a `shuffle` parameter for
  controlling dataset shuffling.
- Updated function docstring for better clarity on new parameters.

- Updated `src/vae_module/__init__.py` to include `activations.py`.

- **Generative Workflow Script Update:**
  - Upscaling the identity features via linear interpolation (instead of simple
    replication).
  - Concatenating the decoded processed features with the upsampled identity
    features.
  - Inverse transforming the generated data and sorting by the 'x' column before
    plotting.

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

- **Data Scaling Improvements (ml_prep/data_utils.py):**
  - Updated the `train_val_test_scale` function (and indirectly
    split_and_scale`_dataset) to handle empty validation and test splits (i.e.
    when `val_size` or `test_size` is 0.0). Now, before applying
    `scaler.transform()`, the code checks whether the input split is non-empty,
    ensuring that empty splits are returned without errors.

- **VAE Optimization Script Update:**
  - **Visualization Enhancements:**
    - Added a latent space visualization step before generating test vs.
      reconstructed plot comparison.
    - The generated latent space figure is saved as a PDF in the designated plot
      directory.

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

- **AutoEncoder Module Update (autoencoder.py):**
  - **Enhanced Type Annotations & Docstrings:**
   Added detailed type hints and improved docstrings across the module to
   clarify method behaviors and support static analysis.
  - **Robust Encoder Construction:**
    Modified the Encoder class to determine the output dimension for the latent
    space layers using the last value in `hidden_layers_dim` instead of indexing
    into the Sequential container.
  - **Naming Conventions:**
    Renamed `customLossWithNegPenalty` to `CustomLossWithNegPenalty` to adhere
    to CamelCase conventions.
  - **Optimizer Refactoring:**
  - **Optimizer Refactoring:**
    Extracted optimizer creation into a helper function (`create_optimizer`) to
    simplify the AutoEncoder constructor and streamline future optimizer
    additions.
    additions.
  - **Logging Enhancements:**
    Replaced print statements in the training loop with logging calls for
    improved production-readiness and configurability.
    improved production-readiness and configurability.
  - **Improved Error Handling:**
    Added assertions to ensure that the processed dimension is non-negative,
    catching configuration errors early.
    catching configuration errors early.
  - **Code Cleanup:**
    Removed commented-out code and improved overall readability and
    maintainability.
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

- **vae_optuna Integration:**  
  Updated `vae_optuna.py` to pass the new `loss_hyperparams` parameter  into the
  model initialization.
  Adjusted the hyperparameter search space and documentation accordingly.

- **Net Parameters and Model Instantiation in vae_optuna.py:**
  - Modified the construction of `net_params` in both the `opt_objective` and
    `build_vae_model_from_params` functions to include a new key
    "`use_final_activation`", allowing users to control the behavior of the
    final decoder layer.
  - Optionally, the `use_final_activation` parameter tuned as a hyperparameter.
  - Updated logging and variable naming for clarity.

- **Decoder Final Layer Configurability in models.py:**
  - Added a new parameter `use_final_activation` to the `Decoder` class.
  - When `use_final_activation` is set to True, the decoder applies
    normalization and the specified activation (e.g. Softplus) on its final
    layer. When False, the final layer is a plain linear projection.
  - Updated the `AutoEncoder` class to pass the new `use_final_activation`
    parameter from net_params to the `Decoder` constructor.

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

- **Output Activation Update:**
  - Switched the VAE decoder's output activation to use Softplus to enforce
    non-negative outputs.
  - Centralized this setting by introducing a global constant
    `DEFAULT_OUTPUT_ACTIVATION` (set to `nn.Softplus`), which is now used in
    optimization pipeline.

- **VAE Optimization Pipeline Enhancements:**
  - Introduced a helper function `build_vae_model_from_params` that encapsulates
    the logic for reconstructing the network architecture (e.g. generating
    `net_size`), building the `vae_params`, `net_params`, and `optimizer_params`
    dictionaries, and instantiating the `AutoEncoder` model.
  - Updated the main function in `vae_optuna.py` to re-train the best model
    using the best trial's hyperparameters, export the best model's state
    dictionary (e.g., `best_vae_model.pt`), and generate a reconstruction
    comparison figure on test data.

- **Dynamic Identity Dimension in VAE Optimization:**
  - Updated the `prepare_data()` function to store the identity features list in
    the `VAEData` container.
  - Modified the `opt_objective` function so that `"identity_dim"` is set as
    `len(data.identity_features)` instead of a hard-coded value.
  - Updated the main function to similarly set `vae_params["identity_dim"]` based
    on the length of `data.identity_features`.
  - These changes ensure that the model configuration automatically adjusts if
    the identity features list is modified.

- **Contour Plot Logic Improvement in VAE Optimization:**
  - Revised the contour plot generation in `vae_optuna.py` to create unique
    parameter pairs and generate separate contour figures for each pair.

- **VAE Optimization Objective Update:**
  - Updated the `opt_objective` function in `vae_optuna.py` to include optimizer
    selection in the hyperparameter search.
  - The function now suggests an optimizer type (choosing among "Adam", "SGD",
    and "RMSprop") and, if needed, additional hyperparameters such as momentum
    for SGD.
  - The key optimizer selection has been updated to `optimizer_type` to ensure
    consistency with the updated `AutoEncoder` class.

- **AutoEncoder Optimizer Refactoring:**
  - Introduced a new key `optimizer_type` in the `optimizer_params` dictionary
    to allow selecting different optmizers (e.g. Adam, SGD, RMSprop) during
    training and hyperparamter optimization.
  - Updated the `__init__` method in the `AutoEncoder` class to choose the
    optimized based on `optimizer_type`, including fetching optimizer-specific
    paramters such as momentum for SGD.
  - Fixed the device assignment condition to correctly check if a device was
    provided.

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

- **AutoEncoder Fit Function Update:**
  - Modified the `fit` function the `AutoEncoder` class to support early pruning
    during training:
    - An optional `trial` parameter has been added so that, when provided (e.g.,
      during Optuna hyperparameter optimization), the function reports the
      validation loss at each epoch and checks for pruning via `trial.report()`
      and `trial.should_prune()`. When no trial is supplied, the function
      behaves as before, allowing for standard training runs.

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

- **VAE Optimization Script (vae_optuna.py):**
  - Added a pruning strategy using Optuna's MedianPruner (n_warmup_steps=5) to
    stop unpromising trials early.
  - Removed `training_epochs` from the hyperparameter search and set it as a fixed
    value (50 epochs) to reduce the search space.
  - Updated the study creation to use a fixed study name ("vae_optimization") with
    SQLite storage, ensuring persistent storage of trial results.

- **VAE Optimization Script (vae_optuna.py):**
  - Added code examples for generating visualizations using Optuna's built-in
    functions (e.g., `plot_optimization_history`, `plot_param_importances`, and
    `plot_slice`).
  - These visualizations offer a detailed view of the study progress and
    parameter effects, improving interpretability of the hyperparameter search.

- **VAE Optimization Script (vae_optuna.py):**
  - Added persistent storage using SQLite so that each trial is saved
    immediately upon completion. The study is created with
    `storage="sqlite:///optuna_study.db"` and `load_if_exists=True`,
    enabling the optimization process to resume from previous runs.

- **VAE Optimization Script (vae_optuna.py):**
  - Removed plotting-related parameters (`vae_plot_dir` and `vae_history_dir`)
    from the call to `train_vae` in the objective function.
  - Updated the objective function to unpack the tuple returned by `train_vae`
    (model, history) and extract the final validation loss from
    `history['val_losses']`.
  - This change further decouples training from visualization and simplifies the
    optimization loop.

- **VAE Training Script (vae_training.py):**
  - Updated `train_vae` to return the trained model alongside the training
    history.
  - Main function now uses the returned model to generate and save plots (latent
    space, test vs. reconstructed data), decoupling training from visualization.

- **VAE Optimization Script (vae_optuna.py):**
  - Extended hyperparameter optimization to include network architecture.
  - The `net_size` parameter is now dynamically optimized by allowing Optuna to
    determine the number of hidden layers (from 1 to 5) and the size of each
    layer.
  - This enhancement allows the VAE training pipeline to explore a wider range
    of model architectures.

- **VAE Optimization Script (vae_optuna.py):**
  - Removed directory configuration from the objective function by defining
    global constants for `vae_plot_dir` and `vae_history_dir`. The objective
    now uses these global settings, reducing redundancy and ensuring consistent
    output paths.

- **VAE Optimization Script (vae_optuna.py):**
  - Removed data import and cleaning from the Optuna objective function.
  - Introduced `prepare_data()`, which preloads, cleans, splits, and scales data,
    storing results in global variables.
  - The objective function now uses these precomputed DataLoaders for training,
    reducing overhead and ensuring consistency across trials.

- **New Feature (Optuna Optimization):**
  - Introduced a new script `vae_optuna.py` for hyperparameter optimization
    of the Variational Autoencoder using Optuna.
  - Developed in a dedicated branch (`optuna_optimization`) to allow for
    independent development and testing before merging into the main codebase.

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

- **vae_training.py:**
  - Updated the data loader creation pipeline by changing
    create_data_loaders_from_dataframes to return a namedtuple (`DataLoaders`)
    instead of a dictionary. This improves readability and makes it easier to
    access the train, validation, and test loaders.  

- **autoenc module:**
  - Updated `plot_test_and_reconstructed`:
    - Now supports a `plot_cols` parameter that accepts either column indices
      or column names.
    - If column names are provided, the function retrieves the column mapping
      from the DataLoader (via its `col_mapping` attribute) and converts them
      to indices.
    - This enhancement allows for more flexible selection of features for
      plotting.

- **ml_prep module:**
  - Added a new function `plot_train_test_val_distribution_df` to plot the
    distribution of training, validation, and test datasets when provided as
    pandas DataFrames. This function uses the DataFrame's column names to select
    features for plotting, removing the need for a separate original DataFrame
    argument.

- **ml_prep module:**
  - Updated `train_val_test_scale` to preserve DataFrame structure when inputs
    are DataFrames:
    - The function now retains the original index and columns by converting
      scaled numpy arrays back to DataFrames.
    - This allows column names and mappings to be preserved for downstream
      tasks.

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
  - Added a `col_mapping` attribute to the DataBuilder class when the input
    data is a pandas DataFrame, storing a mapping from column names to indices.
  - Updated the `create_data_loader` function to attach this column mapping to
    the returned DataLoader object, facilitating downstream tasks that require
    access to column names.

- **autoenc module:**
  - Improved `plot_latent_space` to support the new identity feature bypass.
    - The function now checks if the model has an identity bypass and processes
      only the processed features.
    - Added type annotations and error handling (e.g., checking for an empty
      DataLoader and proper label extraction).
    - Changed `label_column` to `color_column` to better indicate that the
      values are used for coloring; changed `labels` to `color_values` for the
      same reason.
    - Updated the docstring to document these changes.

- **VAE Training Script:**
  - Update `train_vae` to support identity feature bypass
    - Added a new parameter `identity_dim` (default 0) in `vae_params_opt`.
    - Passed `identity_dim` to `net_params` so the AutoEncoder can split the
      input into processed and identity features.
    - Verified that the training pipeline remains unchenged apart from the new
      identity feature ahndling.

- **VAE Module:**
  - Updated the AutoEncoder class to support an identity feature bypass.
    - A new parameter `identity_dim` is added to `net_params` (default 0) to
      define the number of identity features.
    - The input is split into a processed part (to be encoded and decoded) and
      an identity part (bypassing the latent space).
    - The final output concatenates the reconstructed processes features with
      the unchanged identity features.

- **VAE Module:**
  - Updated the AutoEncoder loss computation in __build_loss to support the
    identity feature bypass:
    - When identity features are present (using the new identity_dim parameter),
      the reconstruction loss (MSE) is computed only on the processed features.
    - This prevents penalizing the identity features that bypass the encoder.

- **VAE Training Script:**
  - Added `reorder_identity_features` to reposition identity features
    before dataset splitting, supporting the new AutoEncoder architecture.
  - **Reminder:** Further updates are needed in the autoenc module:
    - Refactor `plot_latent_space` to better visualize the latent space with
      identity features.
    - Refactor `plot_test_and_reconstructed` to correctly reattach or visualize
      identity features.
    - These updates will be implemented in a dedicated branch.

- **preprocessing module:**
  - Fixed an error where an incorrect function was included instead of
    `cut_dataframe_with_primary`. The correct implementation is now restored,
    ensuring that data truncation based on the primary particle stopping point
    works as expected.

- **VAE Training Script:**
  - Integrated a new `import_and_clean_data` function that leverages the
    preprocessing module to import and clean LET data for VAE training.
  - Updated the data preparation pipeline to use `split_and_scale_dataset` from
    the ml_prep module for splitting the dataset into training, validation, and
    test sets and scaling them.
  - These changes streamline the VAE training workflow and ensure consistency
    in data preprocessing across the project.

- **ml_prep module: Add split_and_scale_dataset function**
  - Added and optimized the split_and_scale_dataset function:
    - Added type hints and changed the verbose parameter to a boolean.
    - Added validation to ensure train, validation, and test sizes sum to 1.
    - Improved default parameter handling and error messaging.

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

- **VAE Training Script:**
  - Created a dedicated script for Variational Autoencoder (VAE) training by
    extracting and retaining only the VAE design, training, and visualization code
    from main_conditional.py.
  - Removed all EDA and Data Engineering code to focus on VAE-specific routines.
  - Updated module imports, configuration, and helper functions to support a
    streamlined VAE training workflow.

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

- **eda_main.py: Rename plot_and_export to plot_extendend_comparison**
  - Renamed the function to better reflect its role in generating extended
    exploratory visualizations (e.g., single-element LET plots, fluence
    distributions, and secondary data comparisons).

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

- **data_analysis_utils.py: Add find_bragg_peak_start function**
  - Added a function that computes the first derivative of an input array or
    Series (approximating a Bragg curve) and returns the index where the
    derivative exceeds a specified fraction (threshold_ratio) of the maximum
    derivative.
  - The function supports both NumPy arrays and pandas Series as input.

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

- **eda_main.py: Replace print statements with logging**
  - All print calls in the main function have been replaced with logger.info and
    logger.debug calls for improved output control and consistency.
  - Verbose output is now managed via the logging configuration.
  - Existing EDA plotting and processing functionality remains unchanged.

- **dataeng_main.py: Enhance import_and_clean_data function**
  - Added fallback extraction of `cut_in_um` and `voxel_in_um` from the last
    directory name in `data_dir` using regex patterns when the filename does not
    provide these values.
  - Converts values from mm to micrometers when needed and logs the extraction
    process.
  - Raises a `ValueError` if neither the filename nor the directory yields valid
    cut and voxel values.

- **data_analysis_utils.py: Refactor get_cut_voxel_from_filename**
  - Add error handling for filename parsing in get_cut_voxel_from_filename.

- **dataeng_main.py: Refactor main function into smaller helper functions**
  - Split the main function into:
    - `import_and_clean_data` for data import and cleaning.
    - `process_outliers` for running outlier detection and replacement,
      including automatic candidate selection based on a quality metric.
    - `save_processed_data` for saving the processed data in the same *.out
      format as the original.
  - This refactoring improves code readability, maintainability, and testing.

- **dataeng_main.py: Replace print statements with logging and add entry point**
  - Replaced all print statements with logging calls (INFO/DEBUG/ERROR) for
    better configurability and debugging.
  - Introduced the standard "if __name__ == '__main__'" entry point pattern.
  - Updated the saving procedure for processed data to generate an output file
    with a _replaced.out suffix in the same directory as the original input.
  - Key processing steps, such as data import and saving, are now logged.

- **dataprocessing.py / dataeng_main.py: Enhance outlier search and auto-save**
  **functionality**
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

- **plot_utils.py: Add auto-save capability to plot_let_outliers_replacements**
  - Introduced `auto_save` and `save_path` parameters to automatically save
    figures.
  - The function now calls `fig.savefig(save_path, dpi=300)` when `auto_save` is
    True.
  - If the directory does not exist, it is created before saving the figure.
  - Maintains non-blocking display using `_plt.show(block=False)` and
    `_plt.pause(0.001)`.

- **dataeng_main.py: Implement metric-based automatic selection for outlier detection**
  - Added a replacement quality metric (mean absolute error between replacement
    values and the local baseline from k-nearest non-outlier values) via a
    helper function.
  - Updated the `outliers_search` function to automatically select the candidate
    method with the lowest quality metric, eliminating the need for manual user
    input.

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

- **eda_main.py: Enhance extraction of cut and voxel values and run_identifier**
  - Added fallback extraction for `cut_in_um` and `voxel_in_um` from the last
    directory in `data_dir` using regex when extraction from the filename fails.
  - Converts values from mm to µm when necessary and logs extraction details.
  - Raises a `ValueError` if the cut and voxel values cannot be obtained.

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

- **plot_utils.py: Enhance `plot_let_outliers_replacements` function**
  - Ensure that each call creates a new figure so that a separate window
    is opened for each plot.
  - Use non-blocking display with `_plt.show(block=False)` and
    `_plt.pause(0.001)` to avoid halting execution.
  - Add `_plt.tight_layout()` to improve the layout and prevent clipping of
    titles and labels.
  - Return the figure object for further processing if needed.

- **dataprocessing.py: Refactor `search_neighbors` function**
  - Optimized the removal of query points from the reference dataset using
    boolean indexing.
  - Updated comments to clarify that `search_neighbors` now returns a list where
    each element is a NumPy array of neighbor indices (with shape `(1,
    n_neighbors)`) if the reference dataset is a DataFrame.

- **dataprocessing.py: Enhance `local_statistic_outlier_replacement` function**
  - Updated the docstring to clearly describe the expected structure of the
    input data and the role of the LET value column.
  - Modified the logic to flatten the neighbor indices (if returned as a NumPy
    array) so that they can be used consistently for indexing.
  - Added handling for cases where no neighbors are found by appending `NaN` as
    the replacement value.
  - Revised comments to reflect the updated behavior in light of changes to
    `search_neighbors`.

- **dataprocessing.py: Update `replace_outliers` function**
  - Improved type handling for aggregate calculations by checking if non-outlier
    data is a Series or a DataFrame before extracting the median or mean value.
  - Enhanced error handling by explicitly raising a `ValueError` for unsupported
    `replace_method` values.
  - Cleaned up the code and added clarifying comments for better readability.

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

- **Code Refactoring:**  
  Removed unnecessary plotting functions and parameters related to EDA. The
  script now concentrates on data engineering tasks.
- **Variable Naming:**  
  Updated variable names (e.g., using `primary_search_pattern` derived from
  `primary_particle`) for consistency and clarity.
- **Function Parameters:**  
  Adjusted function signatures (especially in `main`) to reflect the dedicated
  focus on data engineering.

### Removed

- Removed redundant code in `src/vae_optuna.py`.
- Removed redundant code in `src/vae_module/autoencoder

### Fixed

- **[generative-workflow] - 2025-07-29**
  - Corrected `ks_2samp` result handling by explicitly unpacking p-value to
    comply with Ruff linting.
  - Resolved marker type warning by wrapping marker symbols with `MarkerStyle()`
    in matplotlib plots.
  - Added explicit `ValueError` when `main()` is called without `config_path`,
    addressing static analysis warning from Pylance.

- **Redundant Suffix Issue:**  
  Resolved an issue where the primary particle identifier could end up with a
  redundant `"_1"` by checking if it already ends with that suffix before
  appending.

- Prevented unconditional instantiation of decoder output activation by
  deferring creation until runtime.
- Resolved logging inconsistency where debug messages from `autoencoder.py` were
  not appearing unless global logging level was set to DEBUG.

- **[post_analysis] - 2025-04-10**
  - Graceful handling of missing or malformed training history files.

### Moved

- **[optuna_optimization] - 2025-04-10**
  - Archived the old pipeline to `backup/vae_optuna.py`.
  - Copied the last development version of `vae_optuna_claude.py` to
    `dev/older_version/` for future reference.

---

## [1.0.0] - 2025-02-04

- **Initial Release:**  
  This is the first release of `dataeng_main.py`, providing a dedicated data
  engineering script that:
  - Imports and cleans LET and dose data.
  - Determines the primary particle stopping point.
  - Implements outlier detection and replacement strategies.
