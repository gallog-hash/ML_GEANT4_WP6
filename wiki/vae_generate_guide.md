# Script Guide: `vae_generate.py`

This guide provides a modular summary of the VAE generation script and
how to use it for inference and evaluation of synthetic data.

- **Title:** VAE Generation Script Guide  
- **Script:** `src/vae_generate.py`  
- **Uses:** config-driven, fully modular, supports output plots  
- **View:** CLI or interactive (e.g., VSCode)

---

## 🧠 Structure Overview

- `GenerationConfig`: Loads from JSON with parameters.
- `VAEGenerate` class:
  - `prepare_data_loaders()`: Loads and prepares test data.
  - `get_latents()`: Encodes input into latent space.
  - `interpolate_latents_codes()`: Generates new latent vectors via
    interpolation.
  - `interpolate_identity()`: (If applicable) interpolates identity
    features.
  - `generate_high_res()`: Decodes latent vectors into synthetic
    samples.
  - `export_to_let_file()`: Exports upsampled data to Let.out format.
- `VAEGeneratorAnalysis` class:
  - `plot_input_and_generated()`: Compares original and generated
    features visually.
  - `evaluate_generated()`: Computes metrics such as Wasserstein
    distance and KS-test.
- `main()`: Entry point coordinating the full generation pipeline.

---

## 🎯 Input Modes & Profile System

The generation pipeline supports two modes for loading input data,
configured via a **profile-based system**. See
[PROFILE_CONFIG_GUIDE.md](../PROFILE_CONFIG_GUIDE.md) for complete
details.

### 1. Downsample Mode (Default)

Loads high-resolution data and downsamples it to create synthetic
low-resolution input. The remaining points serve as ground truth for
validation.

**Configuration** ([generation_config.json](../src/configs/generation_config.json)):
```json
{
  "profile": "downsample",
  "data_dir": "data/thr96_1e8_v1um_cut1mm_ver_11-2-2",
  "data_file": "Let.out"
}
```

**Profile settings** ([generation_profiles.json](../src/configs/generation_profiles.json)):
```json
{
  "downsample": {
    "input_mode": "downsample",
    "downsample_factor": 20,
    "upsample_factor": 20
  }
}
```

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
- Enables validation against ground truth
- Generates comparison plots
- Computes statistical metrics (Wasserstein, KS-test, mean/variance)

### 2. Direct Mode

Loads low-resolution data directly from a low-resolution Geant4
simulation. No downsampling is performed, and no ground truth is
available for comparison.

**Configuration** ([generation_config.json](../src/configs/generation_config.json)):
```json
{
  "profile": "direct",
  "data_dir": "data/thr96_1e8_v1um_cut1mm_ver_11-2-2",
  "data_file": "Let.out"
}
```

**Profile settings** ([generation_profiles.json](../src/configs/generation_profiles.json)):
```json
{
  "direct": {
    "input_mode": "direct",
    "lowres_data_dir": "vae_downsample_output/thr96_1e8_v1um_cut1mm_ver_11-2-2",
    "lowres_data_file": "Let_downsampled_20x.out",
    "upsample_factor": 20
  }
}
```

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
- Skips comparison plots (no ground truth)
- Skips statistical evaluation (no ground truth)
- Directly generates super-resolved output

**Note:** If `lowres_data_dir` or `lowres_data_file` are not specified
in the profile, the pipeline will fall back to using base config's
`data_dir` and `data_file`.

---

## 🚀 CLI and Entry Point

Run the script from the command line:

**Method 1: Using default profile from config:**
```bash
python src/vae_generate.py
```

**Method 2: Override profile via CLI:**
```bash
# Downsample Mode (Validation)
python src/vae_generate.py --profile downsample

# Direct Mode (Production)
python src/vae_generate.py --profile direct
```

**Method 3: Custom config path:**
```bash
python src/vae_generate.py --config_path src/configs/generation_config.json
```

**Method 4: Override low-resolution data file:**
```bash
# Use a different low-res data file without editing config
python src/vae_generate.py --profile direct --lowres_data_file Let_downsampled_10x.out
```

> 💡 If using Jupyter or VSCode Interactive Window, run:
>
> ```python
> main("configs/generation_config.json")
> ```

**CLI Arguments:**
- `--config_path`: Path to config JSON file (default: uses built-in
  config)
- `--profile`: Override profile selection (e.g., "downsample", "direct")
- `--lowres_data_file`: Filename of low-resolution data file (overrides
  value in config file)

---

## 📦 Outputs

### Downsample Mode Outputs

- **`vae_generate_output/Let_upsampled.out`** - High-resolution
  upsampled data in Let.out format
- **`vae_generate_plots/`** - Comparison plots showing original vs
  generated features
- **Console output** - Statistical metrics (Wasserstein distance,
  KS-test, mean/variance differences)

### Direct Mode Outputs

- **`vae_generate_output/Let_upsampled.out`** - High-resolution
  upsampled data in Let.out format
- **Console output** - Log messages indicating ground truth is
  unavailable
- **No comparison plots or metrics** (due to absence of ground truth)

---

## 🛠 Common Tasks with This Script

### Model Validation (Downsample Mode)
- Use a trained VAE model with high-resolution data
- Validate super-resolution performance against ground truth
- Generate comparison plots and statistical metrics
- Tune `downsample_factor` and `upsample_factor` in profiles to test
  different resolution ratios

### Production Inference (Direct Mode)
- Load real low-resolution simulation data from Geant4
- Generate high-resolution predictions without ground truth
- Export results to Let.out format for downstream analysis
- Skip validation steps when ground truth is unavailable

### Configuration Customization

**Base config** (`src/configs/generation_config.json`):
- `profile`: "downsample" or "direct"
- `data_dir` / `data_file`: Input data location
- `features_to_plot`: Features to visualize
- `output_dir`: Where to save plots
- `enable_plots`: Whether to generate plots

**Profile settings** (`src/configs/generation_profiles.json`):
- `input_mode`: "downsample" or "direct"
- `lowres_data_dir` / `lowres_data_file`: Direct mode data paths
- `downsample_factor`: Downsampling ratio (downsample mode only)
- `upsample_factor`: Upsampling ratio (both modes)

---

## 🧩 Developer Task Reference

| Component | Responsibility |
|-----------|----------------|
| `main()` | Orchestrates data loading, generation, and evaluation |
| `GenerationConfig` | Loads configuration from JSON with input mode |
| `VAEGenerate` | Performs latent encoding, interpolation, decoding |
| `prepare_data_loaders()` | Creates DataLoaders for inference |
| `get_latents()` | Extracts latent codes from encoder |
| `interpolate_latent_codes()` | Interpolates between latent vectors |
| `interpolate_identity()` | Interpolates identity features (e.g., x) |
| `generate_high_res()` | Generates high-resolution samples |
| `export_to_let_file()` | Exports to Let.out format |
| `VAEGeneratorAnalysis` | Analyzes and visualizes generated data |
| `plot_input_and_generated()` | Plots comparison (if ground truth) |
| `evaluate_generated()` | Computes metrics (if ground truth) |
| `compute_wasserstein()` | Wasserstein distance metric |
| `compute_ks_test()` | Kolmogorov-Smirnov test |
| `compute_mean_variance_difference()` | Mean/variance comparison |

---

## 🔧 Configuration Parameters

### Base Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile` | str | Profile to load ("downsample" or "direct") |
| `data_dir` | str | Path to high-resolution data directory |
| `data_file` | str | High-resolution data filename |
| `features_to_plot` | list | Features to visualize |
| `output_dir` | str | Directory for plots |
| `enable_plots` | bool | Whether to generate plots |
| `clamp_negatives` | bool | Clamp negative values to zero |

### Profile-Specific Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_mode` | str | "downsample" or "direct" |
| `lowres_data_dir` | str/null | Path for direct mode data |
| `lowres_data_file` | str/null | Direct mode filename |
| `downsample_factor` | int/null | Downsampling factor (downsample mode) |
| `upsample_factor` | int | Upsampling factor (both modes) |

---

## ⚠️ Error Handling

The pipeline includes validation and graceful error handling:

- **Invalid `input_mode`**: Raises `ValueError` if not "downsample" or
  "direct"
- **Missing ground truth in downsample mode**: Raises errors as
  expected
- **Missing ground truth in direct mode**: Logs warnings but continues
  execution
- **Configuration fallback**: Uses `data_dir`/`data_file` if direct
  mode paths not specified

---

## 💡 Recommendations

- **For model validation**: Use downsample mode with high-resolution
  simulation data
- **For production inference**: Use direct mode with real low-resolution
  simulation data
- **For comparative studies**: Run both modes and compare results
- **Backward compatibility**: Existing configurations work without
  changes (default: downsample mode)
