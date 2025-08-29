# 📊 VAE Optimization Analysis Script

A utility for visualizing Optuna hyperparameter optimization studies for VAE models. Generates diagnostic plots such as optimization history, parameter importance, slice plots, EDF plots, trial timelines, and parameter contour plots.

---

## 🧱 Structure Overview

### Main Components

| Component | Description |
|----------|-------------|
| `VAEOptimizationAnalyzer` | Core class that encapsulates study loading and plotting methods. |
| `main()` | Entry point that parses config, initializes the analyzer, and generates plots. |
| `utils.py` (imported) | Handles logging, loading config, study loading, and figure saving. |
| `OptimizationAnalysisConfig` | Typed config class for validating and passing JSON-based settings. |

### Supported Plot Types

- Optimization history
- Parameter importance
- Trial timeline
- EDF (Empirical Distribution Function)
- Parameter slice plots
- Parameter contour plots

---

## 🚀 CLI and Entry Point

### Command-Line Execution

```bash
python src/vae_optimization_analysis.py --config_path path/to/config.json
```

### Arguments

| Argument       | Description                                     |
|----------------|-------------------------------------------------|
| `--config_path`| (Optional) Path to the configuration JSON file. |

### Jupyter/VSCode Interactive usage

```python
# Automatically uses "configs/vae_optimization_analysis.json"
%run src/vae_optimization_analysis.py
```

---

## 📁 Outputs

The script saves plots as `.png` files in the `plot_dir` defined in the configuration file.

| Plot File Name                    | Description                                      |
|----------------------------------|--------------------------------------------------|
| `optimization_history.png`       | Objective value across trials                   |
| `param_importance.png`           | Importance of hyperparameters                   |
| `trial_timeline.png`             | Trial durations and scheduling overview         |
| `edf_plot.png`                   | Empirical distribution of objective values      |
| `slice_plot_<batch>.png`         | Slice plots of hyperparameters in batches       |
| `param_contour_<p1>_<p2>.png`    | Contour plots for hyperparameter interactions   |

---

## 🔧 Common Tasks with This Script

### Visualize a VAE hyperparameter tuning study

```bash
python src/vae_optimization_analysis.py --config_path configs/my_custom_config.json
```

Make sure `study_name` and `study_path` point to a valid Optuna study (e.g., SQLite or PostgreSQL).

### Customize Plots

- Change the number of parameters shown per batch in slice plots by editing the `batch_size` value inside `plot_parameters_slice()`.
- Enable/disable value capping in contour plots using the `cap_z` flag in `plot_contour(cap_z=True)`.

---

## 🛠 Developer Task Reference

| Method | Description |
|--------|-------------|
| `__init__(self, config)` | Initializes logger, device, and loads Optuna study |
| `plot_basic_study_analysis()` | Plots optimization history and parameter importance |
| `plot_timeline()` | Plots a Gantt-style timeline of trials |
| `plot_edf()` | Plots empirical CDF of objective values |
| `plot_parameters_slice(batch_size=3)` | Generates slice plots of top parameters in batches |
| `plot_contour(cap_z=False)` | Plots contour maps of hyperparameter pairs (optionally capped) |
| `_create_capped_study(cap)` | Returns a study with capped objective values |

---