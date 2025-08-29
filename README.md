
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

- 🔧 Modular training/optimization via `vae_training.py` and `vae_optimization.py`
- 🎯 Study of various loss functions and network configurations
- 📈 Performance visualization and EDA (via Matplotlib and Optuna)
- 📦 Utilities for dataset I/O, logging, and evaluation included in `utils/`

---

## 📁 Directory Structure

```bash
src/
├── vae_training.py                    # Main VAE training script
├── vae_generate.py                    # Generate super-resolved outputs
├── vae_optimization.py                # Hyperparameter optimization (Optuna)
├── vae_optimization_analysis.py       # Analysis of optimization runs
├── vae_post_analysis.py               # Post-training evaluation and visualization
├── eda_main.py                        # Exploratory data analysis
├── dataeng_main.py                    # Data engineering and preprocessing
├── core/                              # Core VAE pipeline components
│   ├── base_pipeline.py               # Base class for VAE operations
│   ├── model_builder.py               # Factory for building VAE models
│   ├── training_utils.py              # Training utilities and loss functions
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
│   └── plot_utils.py                  # Training metrics visualization
└── configs/                           # Configuration files
    ├── trainer_config.json            # Main training configuration
    ├── generation_config.json         # Model generation settings
    ├── optuna_config.json             # Hyperparameter optimization
    ├── optimization_analysis_config.json # Optimization analysis settings
    └── post_training_config.json      # Post-training evaluation
```

---

## 🧰 Usage

### 1. Train a VAE

```bash
python src/vae_training.py --config_path src/configs/trainer_config.json
```

### 2. Generate Super-Resolved Output

```bash
python src/vae_generate.py --config_path src/configs/generation_config.json
```

### 3. Optimize Hyperparameters

```bash
python src/vae_optimization.py --config_path src/configs/optuna_config.json
```

### 4. Analyze Optimization Results

```bash
python src/vae_optimization_analysis.py --config_path src/configs/optimization_analysis_config.json
```

### 5. Post-Training Analysis

```bash
python src/vae_post_analysis.py --config_path src/configs/post_training_config.json
```

### 6. Exploratory Data Analysis

```bash
python src/eda_main.py
```

### 7. Data Engineering and Preprocessing

```bash
python src/dataeng_main.py
```

### 📘 Script Guides

- [vae_training.py](wiki/vae_training_guide.md) – Training pipeline workflow and configuration
- [vae_generate.py](wiki/vae_generate_guide.md) – Super-resolution generation from trained models
- [vae_optimization.py](wiki/vae_optimization_guide.md) – Hyperparameter optimization with Optuna
- [vae_optimization_analysis.py](wiki/vae_optimization_analysis_guide.md) – Analysis and visualization of optimization results
- [vae_post_analysis.py](wiki/vae_post_training_analysis_guide.md) – Post-training evaluation and metrics

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
