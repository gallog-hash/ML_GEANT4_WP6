
# 🧪 `vae_optimization.py` — VAE Hyperparameter Optimization with Optuna

This script automates hyperparameter tuning for Variational Autoencoders (VAEs) using [Optuna](https://optuna.org/), with support for persistent studies, custom architecture definitions, and modular configuration via JSON.

---

## 📂 Structure Overview

| Component | Description |
|----------|-------------|
| `VAEOptimizer` | Core class extending `BaseVAEPipeline`; orchestrates optimization. |
| `main()` | Entry point that supports CLI and interactive execution (e.g. Jupyter, VSCode). |
| `configs/optuna_config.json` | Main configuration file for trials, paths, defaults. |
| `AutoEncoder` | Model class built with flexible architecture parameters. |
| `utils/*` | Includes data loading, logging, and helper functions. |

---

## 🚀 CLI and Entry Point

### 🧩 Entry Options

- **CLI usage:**

   ```bash
   python src/vae_optimization.py --config_path configs/optuna_config.json
   ```

- **Jupyter/VSCode Interactive usage:**

   ```python
   # Automatically uses "configs/optuna_config.json"
   %run src/vae_optimization.py
   ```

### 🏗️ Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--config_path` | Path to JSON config file (see example config) | `configs/optuna_config.json` |

---

## 📤 Outputs

Depending on your configuration (`OptunaStudyConfig`), the following files are saved to the output directory:

| File | Description |
|------|-------------|
| `model_summary.txt` | Summary of final model architecture |
| `best_model.pth` | Best model weights after optimization |
| `best_hyperparameters_config.json` | Full set of best hyperparameters (suggested + derived) |
| `history.json` | Training history: losses, metrics, etc. |
| `optuna_study.db` | Persistent study storage (if enabled) |

Output folders can be timestamped for isolation between runs.

---

## 🧰 Common Tasks with This Script

### 🔍 Run a full optimization study

```bash
python src/vae_optimization.py --config_path configs/optuna_config.json
```

### 📊 Log best hyperparameters

After trials complete, the best configuration is saved to `best_hyperparameters_config.json` — includes architecture, loss, optimizer, and dropout info.

### 💾 Save model for reuse

Best weights (`.pth`) and model summary are saved automatically. You can load them in downstream tasks (e.g., generation or evaluation scripts).

---

## ⚙️ Developer Task Reference

| Method | Purpose |
|--------|---------|
| `prepare_data()` | Loads and preprocesses training and validation sets. |
| `create_study()` | Sets up Optuna study with optional SQLite storage. |
| `get_network_hyperparams(trial)` | Suggests model architecture (layers, dropout, latent dim). |
| `get_exit_activation(trial)` | Suggests output activation (e.g., `ShiftedSoftplus`, `PELU`). |
| `get_loss_hyperparams(trial, loss_type)` | Suggests custom loss function params. |
| `get_optimizer_hyperparams(trial)` | Suggests optimizer, learning rate, weight decay. |
| `objective(trial)` | Main optimization function used by Optuna. |
| `train_vae_model(...)` | Executes model training loop with pruning support. |
| `save_model_state_dict(...)` | Saves best model weights. |
| `save_hparams_config(...)` | Saves best trial’s hyperparameters to JSON. |
| `save_history(...)` | Dumps full training loss history. |

---

## 🧠 Notes & Tips

- ✅ **Custom activations:** Output layer supports learnable activations: `ShiftedSoftplus`, `ELUWithLearnableOffset`, `PELU`.
- ✅ **Pruning**: Uses `MedianPruner` to skip underperforming trials early.
- 🧩 **Extensible**: The script is modular and easy to extend with new activation types, loss functions, or model variants.
- 🧪 **Reproducibility**: All critical configuration and results are saved for reruns and analysis.
