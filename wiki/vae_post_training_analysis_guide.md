# 🧩 VAE Post-Training Analysis Guide

## 📌 Overview

This guide explains how to run **post-training analysis** for the VAE-based
super-resolution pipeline in the `98_wp6_vae_pytorch` project.
The post-analysis step lets you:

- Reconstruct simulation outputs from trained models.
- Evaluate reconstruction quality.
- Visualize latent space structure.
- Review training history and metrics.

The workflow uses:

- **`vae_post_analysis.py`** – the main script for post-analysis.
- A **PostTrainingConfig JSON** – configuration for loading data, models, and output paths.

---

## 📂 Directory Structure

```plaintext
src/
 ├── vae_post_analysis.py      # Main post-training analysis script
 ├── core/                     # Base pipeline and processing logic
 ├── configs/                  # Configuration JSONs
 ├── utils/                    # Logging, plotting, model loading
 └── ...
```

---

## ⚙️ Configuration

All analysis is driven by a **PostTrainingConfig** JSON file, e.g.:

```json
{
  "hparams_config_path": "../vae_opt_output/best_hyperparameters_config.json",
  "model_path": "../vae_opt_output/best_model_weights.pth",
  "history_path": "../vae_opt_output/best_model_training_history.json",
  "output_dir": "../vae_opt_plots",
  "features_to_plot": ["LTT"],
  "metrics_to_plot": [
    ["train_loss", "val_loss"],
    "train_loss_mse",
    "train_loss_kld",
    "train_neg_penalty"
  ],
  "test_size": 0.1,
  "identity_features": ["x"],
  "drop_zero_cols": true
}
```

---

## 🚀 Running the Analysis

You can run the analysis in **two ways**:

### 1️⃣ Command-line

```bash
python src/vae_post_analysis.py     --config_path src/configs/post_training_config.json
```

### 2️⃣ Interactive / Notebook Mode

```python
from src.vae_post_analysis import main
main("src/configs/post_training_config.json")
```

If running interactively (e.g., in Jupyter), plots will be displayed inline and
also saved to the configured output directory.

---

## 🔍 What the Script Does

When executed, `vae_post_analysis.py`:

1. **Loads Config & Logger**
   - Reads your `PostTrainingConfig` JSON.
   - Initializes a detailed debug logger.

2. **Preprocesses Data**
   - Loads datasets (train, validation, test) via `VAEDataPreprocessor`.
   - Applies scaling and formatting.

3. **Loads Model & Study**
   - Loads trained VAE weights using `hparams_config_path` for architecture.
   - Loads associated Optuna study for hyperparameter metadata.

4. **Prepares Data Loaders**
   - Creates PyTorch DataLoader objects for train/val/test sets.

5. **Reconstructs Test Data**
   - Uses the model to generate reconstructed outputs from test inputs.

6. **Plots Reconstructions**
   - Plots original vs reconstructed features (e.g., `LTT`).
   - Inverse transforms to physical units before plotting.

7. **Extracts & Visualizes Latent Space**
   - Extracts latent vectors from training data.
   - Generates PCA/TSNE plots (UMAP optional).

8. **Plots Training History**
   - Reads saved history JSON.
   - Plots metrics like loss curves over epochs.

---

## 📊 Output Files

Typical outputs saved to `output_dir`:

| File / Folder | Description |
|---------------|-------------|
| `reconstruction_<feature>.png` | Original vs reconstructed feature plots |
| `latent_pca.png` | Latent space PCA visualization |
| `latent_tsne.png` | Latent space t-SNE visualization |
| `training_history.png` | Loss/metric curves |

---

## 🛠 Developer Notes

- **Extending Features**
  - Add new analysis plots by extending `VAEPostAnalyzer`.
  - Latent space methods can be expanded (add `'umap'` for example).

- **Data Format Requirements**
  - `X_true` in `plot_reconstruction` **must be a Pandas DataFrame** with an `'x'` column.
  - Features are assumed to be normalized — inverse transform is applied for
    interpretability.

- **Dependencies**
  - Python ≥ 3.8
  - PyTorch, Optuna, Matplotlib, Pandas, NumPy
  - `torchinfo` for model summary

- **Error Handling**
  - The script will log errors if:
    - Model or history files are missing.
    - Feature columns requested for plotting are not found.
    - Metrics listed in `metrics_to_plot` do not exist in the training history.
