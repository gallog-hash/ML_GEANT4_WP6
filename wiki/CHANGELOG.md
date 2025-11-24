# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Command-line argument for lowres_data_file:**
  Added `--lowres_data_file` command-line argument to `vae_generate.py`
  to allow overriding the low-resolution data filename specified in the
  configuration file. This enables quick testing with different input
  files without modifying the config. The argument accepts a filename
  (not a full path) which is combined with `lowres_data_dir` from the
  config to construct the full file path.

- **README.md directory structure update:**
  Added `timing.py` entry to the utils/ section of the directory
  structure, documenting the optional execution timing utility
  (OptionalTimer) in the project overview.

### Changed

- **Downsampling output directory structure:**
  Modified `vae_downsample.py` to use a consistent output directory
  structure where the downsampling factor is included in the filename
  instead of the directory name. Output files are now saved to
  `vae_downsample_output/<source_dir>/` with filenames like
  `Let_downsampled_20x.out` instead of using separate directories for
  each downsampling factor. This simplifies directory management and
  makes it easier to compare different downsampling factors for the
  same source data.

- **Enhanced timing precision in OptionalTimer:**
  Improved time formatting in `src/utils/timing.py` to automatically
  select appropriate units based on duration: seconds (≥1s) with
  millisecond precision, milliseconds (≥1ms) with microsecond
  precision, or microseconds (<1ms) with nanosecond precision. This
  provides more readable output for operations of varying durations.

- **Total execution timing in vae_generate.py:**
  Wrapped the entire generation pipeline in an outer `OptionalTimer`
  context to measure total script execution time from data
  preprocessing through file export. This provides a comprehensive
  timing measurement alongside individual operation timings.

## [Previous]

### Added

- **Optional execution timing utility:**
  Added `OptionalTimer` context manager in `src/utils/timing.py` for
  measuring execution time across pipeline components. The timer is
  disabled by default and can be enabled via the `enable_timing`
  configuration parameter. Features include: config-controlled
  activation, zero overhead when disabled, graceful handling of missing
  dependencies, automatic detection and unwrapping of `VAELogger`
  wrapper objects, and integration with existing logging infrastructure.
  Compatible with all pipeline scripts (training, generation,
  optimization, etc.).

- **Timing control in GenerationConfig:**
  Added `enable_timing` boolean parameter to `GenerationConfig` to
  control execution time measurement during VAE generation. Defaults to
  `false` for zero overhead in production workflows. When enabled,
  reports timing for: data preprocessing, model initialization, data
  loader preparation, high-resolution generation, data post-processing,
  and file export operations.

- **Plot generation control parameter:**
  Added `enable_plots` boolean parameter to `GenerationConfig` to
  control whether plots and statistical evaluations are generated
  during the VAE generation process. Defaults to `false` to optimize
  for production workflows where only the upsampled textual output is
  needed. When disabled, skips VAEGeneratorAnalysis initialization,
  comparison plots, and statistical metrics (Wasserstein distance,
  KS-test, mean/variance differences).

- **Consolidated documentation for profile-based configuration:**
  Merged GENERATION_INPUT_MODES.md content into PROFILE_CONFIG_GUIDE.md
  to create a comprehensive single-source guide covering input modes,
  profile system mechanics, usage examples, configuration parameters,
  and migration from legacy resample_factor. The consolidated guide
  includes detailed sections on symmetric vs asymmetric workflows,
  adding custom profiles, and troubleshooting common issues.

- **Standalone data downsampling script:**
  Created `src/vae_downsample.py` - a dedicated tool for downsampling
  high-resolution LET data without requiring a trained VAE model. The 
  script loads high-resolution data, downsamples it by selecting every 
  nth element, and exports to Let.out format. Useful for creating 
  low-resolution datasets for testing or for simulating lower-density 
  data collection scenarios.

- **Downsample configuration file:**  
  Added `src/configs/downsample_config.json` with sensible defaults for 
  data downsampling operations including downsample_factor=1000 (um to 
  mm sampling), data directory paths, and preprocessing flags.

- **VAEDownsampler class:**  
  Implemented dedicated class in `vae_downsample.py` that reuses 
  existing preprocessing infrastructure while providing standalone 
  downsampling functionality. Includes methods for downsampling data 
  and exporting to Let.out format with proper coordinate-to-index 
  conversion.


- **Profile-based configuration system:**  
  Implemented a flexible profile-based configuration architecture to 
  eliminate duplicate config files and improve maintainability. Base 
  settings are defined in `generation_config.json` while mode-specific 
  overrides are stored in `generation_profiles.json`. Profiles can be 
  selected via the `profile` field in config or overridden via CLI with 
  `--profile` argument. Available profiles: `downsample` (internally 
  downsample high-res data) and `direct` (use externally provided 
  low-res data).

- **Profile configuration loader:**  
  Added `load_config_with_profile()` function in 
  `src/utils/config_loader.py` that merges base configuration with 
  profile-specific settings. Supports both config-file profile 
  selection and CLI override via `profile_override` parameter.

- **Profile configuration documentation:**  
  Created `PROFILE_CONFIG_GUIDE.md` with comprehensive documentation on 
  the profile system including usage examples, available profiles, and 
  instructions for adding custom profiles.

- **Export to Let.out file functionality:**  
  Added `export_to_let_file` method to VAEGenerate class for exporting 
  upsampled LET data to textual files in the original Let.out format. 
  The method handles conversion from x (mm) coordinates back to i 
  (index) values and writes tab-separated values with proper formatting.

- **Enhanced data preprocessing tracking:**  
  Extended VAEDataPreprocessor to include `cut_in_um` and `voxel_in_um` 
  parameters in the returned data components dictionary for downstream 
  use in generation and export processes.

- **Complete upsampled data export:**  
  Main generation workflow now automatically exports the complete 
  upsampled dataset to "Let_upsampled.out" file in the 
  vae_generate_output directory.

### Changed
- **Explicit resolution factor parameters:**  
  Replaced ambiguous `resample_factor` parameter with explicit 
  `downsample_factor` and `upsample_factor` parameters for improved 
  code clarity:
  - `downsample_factor`: How much to downsample input data (e.g., 20 
    means 1 µm → 20 µm resolution). Used only in `downsample` mode.
  - `upsample_factor`: How many interpolation points to generate 
    between adjacent input points during generation. Used in both modes.
  
  This separation enables asymmetric workflows (e.g., downsample by 10×, 
  upsample by 20× for super-resolution beyond original data).

- **Resolution factors moved to profiles:**  
  Moved `downsample_factor` and `upsample_factor` from base 
  configuration to profile-specific settings in 
  `generation_profiles.json`. Each profile defines its own factors 
  appropriate for its input mode.

- **Updated GenerationConfig dataclass:**  
  Modified `src/configs/task_config.py:GenerationConfig` to replace 
  `resample_factor: int` with `downsample_factor: Optional[int]` and 
  `upsample_factor: int` with clear documentation of their purposes.

- **Updated data preprocessing:**  
  Modified `src/core/preprocessing/data_preprocessor.py` to use 
  `downsample_factor` instead of `resample_factor` with improved 
  logging that displays the actual factor value being used.

- **Updated generation pipeline:**  
  Updated `src/vae_generate.py` to use `upsample_factor` throughout 
  (latent interpolation, identity interpolation, plot titles, 
  reconstructed data extraction) for consistency with new naming scheme.

- **Enhanced CLI for vae_generate.py:**  
  Added `--profile` command-line argument to override profile selection 
  at runtime without editing configuration files.

- **Timing integration in vae_generate.py:**
  Integrated `OptionalTimer` context managers around six major
  operations in the generation pipeline: data preprocessing, model
  initialization, data loader preparation, high-resolution generation,
  data post-processing, and file export. Timing output includes
  operation descriptions and elapsed time in seconds with two decimal
  precision.

- **Improved code readability in vae_generate.py:**  
  - Enhanced code formatting with proper line breaks and improved 
    comment structure throughout the file
  - Split long conditional expressions and function calls across 
    multiple lines for better readability
  - Added descriptive comments for complex operations like latent space 
    interpolation and identity vector upsampling
  - Improved variable naming and method documentation

- **Enhanced error handling and argument parsing:**  
  - Improved config path handling with better default path construction
  - Enhanced exception handling for config loading with more specific 
    error messages
  - Better integration of project root resolution and directory 
    utilities

### Removed
- **Duplicate configuration file:**
  Deleted `src/configs/generation_config_direct.json` as it is no
  longer needed with the profile-based system. All mode-specific
  settings now live in `generation_profiles.json`.

- **Deprecated documentation file:**
  Removed `GENERATION_INPUT_MODES.md` after merging its content into
  PROFILE_CONFIG_GUIDE.md. Updated README.md to reference only the
  consolidated documentation. This eliminates duplication and ensures
  users are directed to current, authoritative documentation for the
  profile-based configuration system.

### Fixed
- **Data component access:**
  Fixed potential KeyError issues when accessing data components by
  adding proper default values and existence checks for `voxel_in_um`
  parameter.

- **Output directory management:**
  Ensured proper creation of output directories using utility functions
  before file export operations.

- **Conditional output directory creation in generation:**
  Modified `VAEGenerate.__init__()` to skip creating the `output_dir`
  directory when `enable_plots` is `False`. Previously, the directory
  was always created regardless of whether plots would be generated.
  Now temporarily removes `output_dir` attribute before
  `BaseVAEPipeline` initialization when plots are disabled, preventing
  unnecessary filesystem operations while preserving the attribute for
  potential programmatic access.

- **Configuration parameter usage in training:**
  Fixed `features_to_plot` parameter in `trainer_config.json` not being
  used during evaluation. The `plot_reconstruction` method was using a
  hardcoded list `['LTT']` instead of reading from the config file. Now
  properly uses `self.config.features_to_plot` to respect user-defined
  plotting preferences.

- **Negative values clamping in training:**
  Implemented proper handling of negative values through the
  `clamp_negatives` configuration parameter to ensure data integrity
  during VAE reconstruction processes in training pipeline.

- **Consistent negative values handling in generation:**
  Added `clamp_negatives` parameter support to generation pipeline to
  ensure consistency with training behavior. The generation process now
  applies the same negative value clamping logic after decoding
  interpolated latents, controlled by the `clamp_negatives` 
  configuration parameter in `generation_config.json`.

- **Plot generation in direct profile mode:**
  Fixed plotting functionality to work in both profile modes. 
  Previously, plots were only generated in downsample mode (when ground 
  truth data was available). Now, plots are always generated: in 
  downsample mode they compare against ground truth (`R_input`), while 
  in direct mode they compare generated output against the original 
  input data (`X_input`). This ensures visualization is available 
  regardless of the data input mode used.

- **OptionalTimer logger compatibility:**
  Fixed `OptionalTimer` to properly handle both `VAELogger` wrapper
  objects and standard `logging.Logger` instances. Previously, passing
  a `VAELogger` wrapper would cause silent failures because the timer
  tried to call `.info()` on the wrapper instead of the underlying
  logger. Now automatically detects wrappers using `hasattr()` check
  for `get_logger()` method and extracts the actual logger instance.
  This fix enables early-stage timing measurements (data preprocessing,
  model initialization) to appear in logs correctly.
