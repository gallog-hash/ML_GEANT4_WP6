# Script Guide: `vae_generate.py`

This guide provides a modular summary of the VAE generation script and how to use it for inference and evaluation of synthetic data.

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
  - `interpolate_latents_codes()`: Generates new latent vectors via interpolation.
  - `interpolate_identity()`: (If applicable) interpolates identity features.
  - `generate_high_res()`: Decodes latent vectors into synthetic samples.
- `VAEGeneratorAnalysis` class:
  - `plot_input_and_generated()`: Compares original and generated features visually.
  - `evaluate_generated()`: Computes metrics such as Wasserstein distance and KS-test.
- `main()`: Entry point coordinating the full generation pipeline.

---

## 🚀 CLI and Entry Point

Run the script from the command line:

```bash
python src/vae_generate.py --config_path configs/generation_config.json
```

> 💡 If using Jupyter or VSCode Interactive Window, run:
>
> ```python
> main("configs/generation_config.json")
> ```

- The script accepts a `--config_path` argument.
- This points to a JSON file that defines model parameters, data path, and
  generation settings.

---

## 📦 Outputs

The script itself does not save outputs by default such as
`generated_samples.csv`, or `generation_log.txt`.

By default, the following applies:

- **Generated data** (from `generate_high_res()`) is returned as a DataFrame,
  but must be explicitly saved by the user.
- **Plots** (from `plot_input_and_generated()`) of the features specified in the
  config file (both input and synthetic data) are saved in the specified output
  directory.
- **Metrics** are logged on the standard output.

---

## 🛠 Common Tasks with This Script

- Use a trained VAE model with `load_model()` and `GenerationConfig`.
- Interpolate latent vectors to create new synthetic data.
- Save output plots and metrics with `evaluate_generated()` and
  `plot_input_and_generated()`.
- Modify `configs/generation_config.json` to change:
  - input data
  - latent sampling method
  - output paths

---

## 🧩 Developer Task Reference

| Component                  | Responsibility                                                 |
|---------------------------|-----------------------------------------------------------------|
| `main()`                  | Orchestrates data loading, generation, and evaluation pipeline |
| `GenerationConfig`        | Loads configuration from JSON                                  |
| `VAEGenerate`             | Performs data loading, latent encoding, interpolation, decoding|
| `generate_high_res()`     | Generates synthetic high-resolution samples                    |
| `interpolate_latents_codes()` | Interpolates between latent vectors                     |
| `VAEGeneratorAnalysis`    | Analyzes and visualizes generated data                         |
| `plot_input_and_generated()` | Plots comparison of input vs. generated features         |
| `evaluate_generated()`    | Runs statistical tests on synthetic vs. real data              |
| `features_to_plot` (config) | Specifies which features to visualize                         |
