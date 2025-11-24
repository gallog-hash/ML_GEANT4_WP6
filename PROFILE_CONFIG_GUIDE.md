# VAE Generation: Profile-Based Configuration & Input Modes

This guide explains the profile-based configuration system for VAE
generation, including the two input modes available for loading and
processing data.

## Table of Contents

1. [Overview](#overview)
2. [Input Modes](#input-modes)
3. [Profile System](#profile-system)
4. [Usage Examples](#usage-examples)
5. [Configuration Parameters](#configuration-parameters)
6. [Understanding Resolution Factors](#understanding-resolution-factors)
7. [Adding Custom Profiles](#adding-custom-profiles)
8. [Implementation Details](#implementation-details)

---

## Overview

The VAE generation pipeline (`src/vae_generate.py`) uses a profile-based
configuration system to support different operation modes without
duplicating configuration files.

**Key concepts:**
- **Base config**: `src/configs/generation_config.json` - Contains all
  shared settings
- **Profile definitions**: `src/configs/generation_profiles.json` -
  Contains mode-specific overrides
- **Input modes**: Two ways to load data (`downsample` or `direct`)
- **Resolution factors**: Separate parameters for downsampling and
  upsampling operations

---

## Input Modes

The generation pipeline supports two modes for loading input data:

### 1. Downsample Mode (Default)

Loads high-resolution data and downsamples it to create synthetic
low-resolution input. The remaining points serve as ground truth for
validation.

**Workflow:**
```
High-res Let.out → Downsample (every Nth point)
                        ↓
              [Low-res input] + [Ground truth]
                        ↓
            VAE Generate → High-res output
                        ↓
              Compare with ground truth
```

**Features:**
- Enables validation against known ground truth
- Generates comparison plots showing original vs generated data
- Computes statistical metrics (Wasserstein distance, KS-test,
  mean/variance)
- Useful for model validation and hyperparameter tuning

**When to use:**
- Model development and validation
- Testing VAE performance on synthetic data
- Comparative studies with known ground truth
- Hyperparameter optimization

### 2. Direct Mode

Loads low-resolution data directly from a low-resolution Geant4
simulation. No downsampling is performed, and no ground truth is
available for comparison.

**Workflow:**
```
Low-res Let_lowres.out (from G4 simulation)
                ↓
        VAE Generate → High-res output
                ↓
        Export to Let_upsampled.out
```

**Features:**
- Uses real low-resolution simulation data
- Skips comparison plots (no ground truth available)
- Skips statistical evaluation (no ground truth available)
- Directly generates super-resolved output for production use

**When to use:**
- Production inference with real low-res simulation data
- Generating high-resolution predictions for downstream analysis
- Scenarios where ground truth is unavailable
- Real-world deployment

---

## Profile System

Instead of maintaining separate config files for each mode, the profile
system enables easy switching between configurations.

### Available Profiles

#### `downsample` (default)

Uses internally downsampled data from high-resolution input.

```json
{
  "input_mode": "downsample",
  "lowres_data_dir": null,
  "lowres_data_file": null,
  "downsample_factor": 20,
  "upsample_factor": 20
}
```

**Parameters:**
- `downsample_factor`: How much to downsample input (e.g., 20 means
  1 µm → 20 µm)
- `upsample_factor`: How much to interpolate during generation (e.g.,
  20 means 20 µm → 1 µm)

#### `direct`

Uses externally provided low-resolution data.

```json
{
  "input_mode": "direct",
  "lowres_data_dir": "vae_downsample_output/thr96_1e8_v1um_cut1mm_ver_11-2-2",
  "lowres_data_file": "Let_downsampled_20x.out",
  "downsample_factor": null,
  "upsample_factor": 20
}
```

**Parameters:**
- `downsample_factor`: Not used (set to null)
- `upsample_factor`: How much to interpolate between input points
- `lowres_data_dir`: Directory containing low-resolution data
- `lowres_data_file`: Filename of low-resolution data

**Note:** If `lowres_data_dir` or `lowres_data_file` are not specified
in direct mode, the pipeline will fall back to using the base config's
`data_dir` and `data_file`.

### Benefits of Profile System

- **DRY principle**: Shared settings defined once in base config
- **Easy switching**: Change modes without editing large config files
- **Scalability**: Add new profiles without duplication
- **Maintainability**: Update common settings in one place
- **Flexibility**: Override any setting per profile

---

## Usage Examples

### Method 1: Edit the Base Config File

In [src/configs/generation_config.json](src/configs/generation_config.json),
change the profile field:

```json
{
  ...
  "profile": "downsample"  // or "direct"
}
```

Then run normally:
```bash
python src/vae_generate.py
```

### Method 2: Override via Command-Line

Keep the base config unchanged and override the profile at runtime:

```bash
# Use downsample mode (validation with ground truth)
python src/vae_generate.py --profile downsample

# Use direct mode (production inference)
python src/vae_generate.py --profile direct

# Override low-resolution data file (useful for testing different
# downsampling factors)
python src/vae_generate.py --profile direct \
  --lowres_data_file Let_downsampled_10x.out

# Override upsampling factor
python src/vae_generate.py --upsample_factor 50

# Combine multiple overrides
python src/vae_generate.py --profile direct \
  --upsample_factor 50 \
  --lowres_data_file Let_downsampled_10x.out
```

### Method 3: Programmatic (in notebooks/scripts)

```python
from pathlib import Path
from configs.task_config import GenerationConfig
from utils import load_config_with_profile

# Load with default profile from config file
config = load_config_with_profile(
    config_path='src/configs/generation_config.json',
    config_class=GenerationConfig
)

# Or override the profile
config = load_config_with_profile(
    config_path='src/configs/generation_config.json',
    config_class=GenerationConfig,
    profile_override='direct'
)
```

### Complete Examples

#### Example 1: Downsample Mode (Validation)

Use this mode when you want to validate the VAE's super-resolution
performance against known high-resolution data.

```bash
python src/vae_generate.py --profile downsample
```

**Output:**
- `vae_generate_output/Let_upsampled.out` - High-resolution upsampled
  data
- `vae_generate_plots/` - Comparison plots showing original vs generated
- Statistical metrics (Wasserstein distance, KS-test, mean/variance
  differences)

#### Example 2: Direct Mode (Production)

Use this mode when you have real low-resolution simulation data and want
to generate high-resolution predictions.

```bash
python src/vae_generate.py --profile direct
```

**Output:**
- `vae_generate_output/Let_upsampled.out` - High-resolution upsampled
  data
- Log messages indicating ground truth is unavailable
- No comparison plots or metrics

---

## Configuration Parameters

### Base Configuration (`generation_config.json`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile` | str | Profile name to load ("downsample" or "direct") |
| `data_dir` | str | Path to high-resolution data directory |
| `data_file` | str | High-resolution data filename |
| `optuna_io_dir` | str | Directory containing Optuna study outputs |
| `database` | str | Optuna database filename |
| `study_name` | str | Name of Optuna study |
| `output_dir` | str | Directory for plots and outputs |
| `features_to_plot` | list | Features to visualize |
| `enable_plots` | bool | Whether to generate plots |
| `clamp_negatives` | bool | Clamp negative values to zero |

### Profile-Specific Parameters (`generation_profiles.json`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_mode` | str | "downsample" or "direct" |
| `lowres_data_dir` | str/null | Path for direct mode data (null in
downsample mode) |
| `lowres_data_file` | str/null | Direct mode filename (null in
downsample mode) |
| `downsample_factor` | int/null | Downsample input by this factor
(null in direct mode) |
| `upsample_factor` | int | Interpolate by this during generation
(both modes) |

### CLI Override Parameters

All profile-specific parameters can be overridden via command-line
arguments:

| CLI Argument | Overrides | Example |
|--------------|-----------|---------|
| `--profile` | `profile` field | `--profile direct` |
| `--lowres_data_file` | `lowres_data_file` | `--lowres_data_file
Let_downsampled_10x.out` |
| `--upsample_factor` | `upsample_factor` | `--upsample_factor 50` |

---

## Understanding Resolution Factors

### Downsample Factor

- **Purpose**: Reduces the resolution of high-res input data
- **Example**: If input is at 1 µm resolution and
  `downsample_factor = 20`, the downsampled data will be at 20 µm
  resolution
- **When used**: Only in `downsample` mode (set to null in `direct`
  mode)
- **Typical values**: 10, 20, 50, 100 depending on desired resolution
  reduction

### Upsample Factor

- **Purpose**: Determines how many points to interpolate between
  adjacent input points during generation
- **Example**: If `upsample_factor = 20`, the VAE generates 19
  intermediate points between each pair of input points (total 20
  segments)
- **When used**: Both modes use this during generation
- **Typical values**: Match the downsample factor for symmetric
  workflows, or differ for super-resolution beyond original data

### Symmetric vs Asymmetric Workflows

**Symmetric (typical):**
```json
{
  "downsample_factor": 20,
  "upsample_factor": 20
}
```
- Downsample: 1 µm → 20 µm
- Upsample: 20 µm → 1 µm
- Result: Same resolution as original

**Asymmetric (experimental):**
```json
{
  "downsample_factor": 10,
  "upsample_factor": 20
}
```
- Downsample: 1 µm → 10 µm
- Upsample: 10 µm → 0.5 µm
- Result: Super-resolution beyond original data (2× higher)

This enables testing VAE's capability to generate finer-grained data
than the original resolution.

---

## Adding Custom Profiles

To add a new profile (e.g., for a different dataset or experimental
mode):

### Step 1: Edit `src/configs/generation_profiles.json`

```json
{
  "downsample": { ... },
  "direct": { ... },
  "my_custom_profile": {
    "input_mode": "direct",
    "lowres_data_dir": "data/my_custom_data",
    "lowres_data_file": "Let_custom.out",
    "downsample_factor": null,
    "upsample_factor": 50
  }
}
```

### Step 2: Use the Profile

```bash
python src/vae_generate.py --profile my_custom_profile
```

### Step 3: Document the Profile

Add a comment in `generation_profiles.json` explaining the profile's
purpose:

```json
{
  "_comment_my_custom_profile": "Experimental profile for testing 50×
upsampling on custom dataset",
  "my_custom_profile": { ... }
}
```

**Note**: Profiles can override ANY setting from the base config, not
just mode-specific ones. This includes `output_dir`, `features_to_plot`,
`enable_plots`, etc.

---

## Implementation Details

### Modified Files

1. **[src/configs/task_config.py](src/configs/task_config.py)**
   - Added `input_mode`, `lowres_data_dir`, `lowres_data_file` to
     `GenerationConfig`
   - Replaced `resample_factor` with `downsample_factor` and
     `upsample_factor`

2. **[src/core/preprocessing/data_preprocessor.py](src/core/preprocessing/data_preprocessor.py)**
   - Modified `load_and_preprocess_data()` to support both modes
   - Returns `None` for ground truth data in direct mode
   - Uses `downsample_factor` for downsampling operations

3. **[src/vae_generate.py](src/vae_generate.py)**
   - Updated `VAEGeneratorAnalysis` to handle missing ground truth
   - Modified `main()` to skip plots/metrics when ground truth is
     unavailable
   - Uses `upsample_factor` for latent interpolation and reconstruction
   - Added `--profile` CLI argument for runtime override
   - Added `--lowres_data_file` CLI argument to override low-resolution
     data filename

4. **[src/configs/generation_config.json](src/configs/generation_config.json)**
   - Added `profile` parameter with default value
   - Removed mode-specific parameters (moved to profiles)

5. **[src/configs/generation_profiles.json](src/configs/generation_profiles.json)** (new)
   - Contains mode-specific overrides for each profile
   - Defines `downsample` and `direct` profiles

6. **[src/utils/config_loader.py](src/utils/config_loader.py)**
   - Implements `load_config_with_profile()` function
   - Handles profile loading and merging logic

### Profile Loading Logic

When loading a config:

1. Base config is loaded from JSON
   ```python
   base_config = json.load(open('generation_config.json'))
   ```

2. Profile name is determined (from config file or CLI override)
   ```python
   profile = args.profile or base_config.get('profile', 'downsample')
   ```

3. Profile settings are loaded from profiles file
   ```python
   profiles = json.load(open('generation_profiles.json'))
   profile_settings = profiles[profile]
   ```

4. Profile settings are merged over base settings (profile wins)
   ```python
   merged = {**base_config, **profile_settings}
   ```

5. Final config dataclass is instantiated
   ```python
   config = GenerationConfig(**merged)
   ```

### Backward Compatibility

The changes are **fully backward compatible**. Existing configurations
will continue to work:
- Default `input_mode="downsample"` if not specified
- Default `profile="downsample"` if not specified
- Legacy `resample_factor` replaced with explicit factors

### Error Handling

The pipeline includes validation to ensure proper configuration:

- **Invalid `input_mode`**: Raises `ValueError` if not "downsample" or
  "direct"
- **Invalid profile**: Raises `KeyError` if profile not found in
  profiles file
- **Missing ground truth in downsample mode**: Raises errors as expected
- **Missing ground truth in direct mode**: Logs warnings but continues
  execution
- **Missing profile parameters**: Falls back to base config values

---

## Recommendations

### For Model Development
- **Use**: `downsample` profile
- **Why**: Provides ground truth for validation and metrics
- **Output**: Comparison plots and statistical evaluation

### For Production Inference
- **Use**: `direct` profile
- **Why**: Works with real low-resolution simulation data
- **Output**: High-resolution predictions for downstream analysis

### For Comparative Studies
- **Use**: Both profiles sequentially
- **Why**: Compare synthetic vs real low-res inputs
- **Output**: Insights into model robustness

### For Hyperparameter Tuning
- **Use**: `downsample` profile with various factor combinations
- **Why**: Test different resolution ratios systematically
- **Output**: Optimal downsample/upsample factor pairs

---

## Migration from Legacy Configuration

If you have old configuration files using `resample_factor`:

### Old Configuration
```json
{
  "resample_factor": 20,
  "data_dir": "data/...",
  "data_file": "Let.out"
}
```

### New Configuration
```json
{
  "profile": "downsample",
  "data_dir": "data/...",
  "data_file": "Let.out"
}
```

**In `generation_profiles.json`:**
```json
{
  "downsample": {
    "downsample_factor": 20,
    "upsample_factor": 20
  }
}
```

The old `resample_factor` is now split into:
- `downsample_factor`: Controls input downsampling
- `upsample_factor`: Controls generation interpolation

This separation enables asymmetric workflows and clearer semantics.

---

## Troubleshooting

### Issue: Profile not found

**Error:** `KeyError: 'my_profile'`

**Solution:** Check that the profile exists in
`src/configs/generation_profiles.json`

### Issue: Downsample factor ignored in direct mode

**Expected behavior:** `downsample_factor` should be null in direct mode

**Why:** Direct mode loads already-downsampled data, so no downsampling
is performed

### Issue: Missing ground truth warnings

**Expected behavior in direct mode:** Pipeline logs warnings but
continues

**Expected behavior in downsample mode:** Pipeline raises errors

**Solution:** Ensure you're using the correct profile for your use case

### Issue: Resolution mismatch

**Symptom:** Output resolution doesn't match expectations

**Check:**
- Verify `upsample_factor` in your active profile
- Verify input data resolution matches profile assumptions
- Check logs for actual factor values being used

---

## See Also

- [VAE Generate Script Guide](wiki/vae_generate_guide.md) - Detailed
  script documentation
- [CHANGELOG](wiki/CHANGELOG.md) - History of configuration system
  changes
- [GenerationConfig Dataclass](src/configs/task_config.py) - Config
  parameter definitions
