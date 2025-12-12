
# 🧠 VAE-Based Super-Resolution for Geant4 Hadrontherapy

This repository implements a **Variational Autoencoder (VAE)** pipeline for
enhancing the resolution of simulation outputs from the **Geant4 Hadrontherapy
example**. It targets particle interaction data (e.g., Bragg peak patterns) and
reconstructs high-resolution outputs using generative modeling techniques.

---

## 🚀 Super-Resolution Task Overview

The **super-resolution task** involves:

- Using low-resolution simulation data from **Geant4 Hadrontherapy**
- Training a **Variational Autoencoder** to model latent representations
- Upsampling or reconstructing **high-resolution distributions**
- Optimizing model parameters via **Optuna**

---

## 🧪 Key Features

- 🔧 Modular training/optimization via `vae_training.py` and
  `vae_optimization.py`
- 🎯 Study of various loss functions and network configurations
- 📈 Performance visualization and EDA (via Matplotlib and Optuna)
- 📊 Comprehensive metrics evaluation with point-wise spatial fidelity
  metrics (MAE, RMSE, MAPE, R²) and domain-specific Bragg peak metrics
  (peak position, height, FWHM, distal falloff) for reconstruction and
  interpolation quality
- 📦 Utilities for dataset I/O, logging, and evaluation included in `utils/`
- 🔽 Data downsampling tool for creating low-resolution datasets from
  high-resolution simulations
- ⚙️ Profile-based configuration system for flexible generation workflows
- 🎛️ Independent control of analysis and plotting for optimized workflows

---

## 📁 Directory Structure

```bash
src/
├── vae_training.py                    # Main VAE training script
├── vae_generate.py                    # Generate super-resolved outputs
├── vae_generate_multi_factor.py       # Batch generation across multiple upsampling factors
├── vae_downsample.py                  # Downsample high-resolution data
├── vae_optimization.py                # Hyperparameter optimization (Optuna)
├── vae_optimization_analysis.py       # Analysis of optimization runs
├── vae_post_analysis.py               # Post-training evaluation and visualization
├── eda_main.py                        # Exploratory data analysis
├── dataeng_main.py                    # Data engineering and preprocessing
├── core/                              # Core VAE pipeline components
│   ├── base_pipeline.py               # Base class for VAE operations
│   ├── model_builder.py               # Factory for building VAE models
│   ├── losses.py                      # Loss functions (VAE, beta-VAE variants)
│   ├── metrics.py                     # Evaluation metrics (MAE, RMSE, MAPE, R²)
│   ├── bragg_peak_metrics.py          # Bragg peak metrics for hadrontherapy
│   ├── training_utils.py              # Training utilities
│   ├── models/                        # Custom PyTorch modules
│   │   ├── autoencoder.py             # AutoEncoder implementation
│   │   └── activations.py             # Custom activations (PELU, ShiftedSoftplus)
│   └── preprocessing/                 # Data preprocessing pipeline
│       ├── data_preprocessor.py       # VAEDataPreprocessor class
│       └── preprocessing_utils.py     # Preprocessing utilities
├── utils/                             # Shared utilities
│   ├── config_loader.py               # JSON configuration loading
│   ├── data_loader_utils.py           # PyTorch DataLoader creation
│   ├── filesystem_utils.py            # File I/O and directory management
│   ├── logger.py                      # VAELogger for consistent logging
│   ├── model_io.py                    # Model saving/loading
│   ├── latent_utils.py                # Latent space analysis
│   ├── plot_utils.py                  # Training metrics visualization
│   └── timing.py                      # Optional execution timing (OptionalTimer)
└── configs/                           # Configuration files
    ├── trainer_config.json            # Main training configuration
    ├── generation_config.json         # Model generation settings
    ├── generation_profiles.json       # Generation profiles (downsample/direct)
    ├── downsample_config.json         # Data downsampling configuration
    ├── optuna_config.json             # Hyperparameter optimization
    ├── optimization_analysis_config.json # Optimization analysis settings
    └── post_training_config.json      # Post-training evaluation
```

---

## 🧰 Usage

### 1. Downsample High-Resolution Data

Before training, you can downsample high-resolution LET data to create
low-resolution datasets:

```bash
# Downsample with default settings from config
python src/vae_downsample.py --config_path src/configs/downsample_config.json

# Downsample with custom factor and output filename
python src/vae_downsample.py \
  --config_path src/configs/downsample_config.json \
  --downsample_factor 20 \
  --output_filename Let_downsampled.out
```

Output is saved to `vae_downsample_output/<source_dir>/` with the
downsampling factor included in the filename (e.g.,
`Let_downsampled_20x.out`)

### 2. Train a VAE

```bash
python src/vae_training.py --config_path src/configs/trainer_config.json
```

### 3. Generate Super-Resolved Output

```bash
python src/vae_generate.py --config_path src/configs/generation_config.json
```

#### Generation Profiles

The generation pipeline supports two input modes via
`generation_profiles.json`:

- **`downsample` mode**: Automatically downsamples high-resolution data during
  generation
- **`direct` mode**: Uses pre-downsampled low-resolution data from a specified
  directory

Specify the profile in your generation config or use command-line override.

#### Command-Line Options

- `--config_path`: Path to generation config JSON file
- `--profile`: Profile name to use (overrides profile in config file)
- `--lowres_data_file`: Filename of low-resolution data file (overrides
  value in config file)
- `--upsample_factor`: Upsampling factor (overrides config value)

#### Examples

```bash
# Use default settings from config
python src/vae_generate.py --config_path src/configs/generation_config.json

# Override profile to direct mode
python src/vae_generate.py --profile direct

# Override upsampling factor
python src/vae_generate.py --upsample_factor 50

# Combine multiple overrides
python src/vae_generate.py --profile direct --upsample_factor 50 \
  --lowres_data_file Let_downsampled_10x.out
```

#### Batch Generation Across Multiple Factors

For systematic comparison across different upsampling factors, use the
multi-factor batch script:

```bash
# Run generation with factors: 10, 20, 50, 100
python src/vae_generate_multi_factor.py
```

This script automatically:
- Runs `vae_generate.py` sequentially for each upsampling factor
- Updates the generation profile configuration between runs
- Saves factor-specific outputs to `vae_generate_output/`:
  - `Let_upsampled_factor_10.out`, `Let_upsampled_factor_20.out`, etc.
  - `vae_generate_factor_10.txt`, `vae_generate_factor_20.txt`, etc.
    (execution logs)
- Restores original configuration after completion

Edit the `FACTORS` list in the script to customize which factors to test.

### 4. Optimize Hyperparameters

```bash
python src/vae_optimization.py --config_path src/configs/optuna_config.json
```

### 5. Analyze Optimization Results

```bash
python src/vae_optimization_analysis.py \
  --config_path src/configs/optimization_analysis_config.json
```

### 6. Post-Training Analysis

```bash
python src/vae_post_analysis.py \
  --config_path src/configs/post_training_config.json
```

### 7. Exploratory Data Analysis

```bash
python src/eda_main.py
```

### 8. Data Engineering and Preprocessing

```bash
python src/dataeng_main.py
```

---

## ⏱️ Performance Profiling

To measure execution time of pipeline operations, enable timing in your
configuration file and filter logs to show only timing measurements:

```bash
# View only timing output from generation pipeline
python src/vae_generate.py \
  --config_path src/configs/generation_config.json \
  2>&1 | grep "completed in"
```

**Configuration requirement:**
Set `"enable_timing": true` in your config file (e.g.,
`generation_config.json`)

**Example output:**
```
[INFO] [VAEGenerate] Data preprocessing completed in 0.15s
[INFO] [VAEGenerate] Model initialization completed in 0.63s
[INFO] [VAETrainer] Data loader preparation completed in 0.00s
[INFO] [VAETrainer] High-resolution generation completed in 0.96s
[INFO] [VAETrainer] Data post-processing completed in 0.01s
[INFO] [VAETrainer] Export to Let.out format completed in 0.83s
```

**Supported scripts:**
- `vae_generate.py` (currently implemented)
- Other pipeline scripts can be extended with `OptionalTimer` as needed

### 📘 Script Guides

- [vae_downsample.py](wiki/vae_downsample_guide.md) – Data downsampling
  tool for creating low-resolution datasets
- [vae_training.py](wiki/vae_training_guide.md) – Training pipeline
  workflow and configuration
- [vae_generate.py](wiki/vae_generate_guide.md) – Super-resolution
  generation from trained models
- [vae_optimization.py](wiki/vae_optimization_guide.md) – Hyperparameter
  optimization with Optuna
- [vae_optimization_analysis.py](wiki/vae_optimization_analysis_guide.md) –
  Analysis and visualization of optimization results
- [vae_post_analysis.py](wiki/vae_post_training_analysis_guide.md) –
  Post-training evaluation and metrics

### 📚 Advanced Guides

- [PROFILE_CONFIG_GUIDE.md](PROFILE_CONFIG_GUIDE.md) – Comprehensive
  guide on the profile-based configuration system, input modes
  (downsample vs direct), and generation workflows

---

## 🔄 Typical Workflow

A typical super-resolution workflow consists of:

1. **Data Preparation** (optional): Downsample high-resolution data to create
   low-resolution training datasets
   ```bash
   python src/vae_downsample.py --config_path src/configs/downsample_config.json
   ```

2. **Model Training**: Train VAE on the dataset
   ```bash
   python src/vae_training.py --config_path src/configs/trainer_config.json
   ```

3. **Super-Resolution Generation**: Generate high-resolution outputs from
   low-resolution inputs
   ```bash
   python src/vae_generate.py --config_path src/configs/generation_config.json
   ```

4. **Evaluation**: Analyze model performance and visualize results
   ```bash
   python src/vae_post_analysis.py \
     --config_path src/configs/post_training_config.json
   ```

For hyperparameter tuning, use `vae_optimization.py` before final training.

---

## 📊 Input Data

The VAE model is trained using Geant4 simulation output stored in the `Let.out`
file located in dataset directories such as:

```bash
data/thr96_1e8_v1um_cut1mm_ver_11-2-2/
├── Let.out   ← used for training (LET profiles)
├── Dose.out  ← used for visualization (dose profile overlay)
```

- **`Let.out`** contains voxelized **Linear Energy Transfer (LET)** values used
  directly as input data for training and generation.
- **`Dose.out`** contains the **energy deposition profile**, which is *not* used
  during training but is optionally overlaid in analysis plots for comparison
  with LET reconstructions.

The data typically represents high-resolution (∼1 μm voxel size) 3D
distributions from **Geant4 Hadrontherapy** simulations. These are parsed into
3D arrays, normalized, and used to learn latent mappings for the
super-resolution task.

---

## 🛠 Dependencies

- Python ≥ 3.8
- PyTorch
- Optuna
- Numpy, Matplotlib, Seaborn
- [Geant4 simulation output](https://geant4.web.cern.ch/)

---

## 📄 License

This project is licensed under the terms of the **GNU General Public License v3.0**.  
See the [LICENSE](./LICENSE) file for full details.
