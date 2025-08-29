import argparse
import json
import os

# import sys
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from matplotlib.markers import MarkerStyle
from torch.utils.data import DataLoader

from configs.task_config import PostTrainingConfig
from core.base_pipeline import BaseVAEPipeline
from core.preprocessing.data_preprocessor import VAEDataPreprocessor
from utils import (
    VAELogger,
    create_data_loaders,
    display_plot,
    is_interactive_environment,
    load_config_from_json,
    load_model,
    load_model_config,
    load_optuna_study,
    plot_training_metrics,
    save_figure,
    summarize_best_trial,
)
from utils.latent_utils import LatentSpaceMixin


class VAEPostAnalyzer(BaseVAEPipeline, LatentSpaceMixin):
    def __init__(
        self, 
        config: PostTrainingConfig, 
        logger_obj: VAELogger,
        data_components: dict,
    ):
        super().__init__(config, logger_obj)
        
        if data_components is not None:
            self.data_components = data_components
        else:
            self.data_components = \
                VAEDataPreprocessor(config, logger_obj).load_and_preprocess_data()

        # Set Optuna logging level
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        self.color_column: Optional[str] = None
        self.latent_space_color_vals: Optional[np.ndarray] = None
        
        self.logger.info(f"Optuna I/O directory set to: {self.optuna_io_dir}")
        self.logger.info(f"Optuna study path set to: {self.study_path}")
        self.study = load_optuna_study(
            study_name=self.config.study_name,
            storage=self.study_path
        )
        summarize_best_trial(self.study, self.logger)
        
        self.hparam_config = load_model_config(
            config_path=self.config.hparams_config_path
        )
        self.model = load_model(
            weights_path=self.config.model_path,
            device=self.device,
            params=self.hparam_config
        )
        
        if self.model is None:
            self.logger.error("Model have not been loaded correctly.")
            raise ValueError("Model must be loaded before proceeding with analysis.")
        
        self.model.eval()
            
    def _load_history(self, history_path=None):
        if history_path is None:
            history_path = self.config.history_path

        if not os.path.isfile(history_path):
            self.logger.error(f"History file '{history_path}' not found.")
            return

        try:
            with open(history_path, "r") as f:
                self.history = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse history file: {e}")
            return
        
        self.logger.info("History successfully loaded.")
        self.logger.info("History contains information on: ")
        for key in self.history.keys():
            self.logger.debug(f"  - {key}")

        
    def model_summary(self):
        from torchinfo import summary
        
        if self.model is None:
            self.logger.error("Model is not defined, cannot generate model summary.")
            raise ValueError("Model must be setup before generating summary.")
        
        self.logger.info("Generating model summary...")
        model_summary = summary(
            model=self.model,
            input_size=(1, self.hparam_config['input_dim']),
            depth=4
        )
        self.logger.info(f"\n{model_summary}")
        
    def prepare_data_loaders(self):
        self.logger.info("Creating data loaders...")
        loaders = create_data_loaders(
            train_data=self.data_components['train_data'],
            val_data=self.data_components['val_data'],
            test_data=self.data_components['test_data'],
            batch_size=self.hparam_config['batch_size'],
            shuffle={"train": True, "val": True, "test": True},
            data_type={"train": "training", "val": "validation", "test": "test"}        
        )
        self.X_train_loader = loaders.get("train")
        self.X_val_loader = loaders.get("val")
        self.X_test_loader = loaders.get("test")
        
    def generate_reconstructed_data(self, X_loader):
        if self.model is None:
            self.logger.error("Model is not defined, cannot reconstruct data.")
            raise ValueError("Model must be setup before data reconstruction.")
        
        self.logger.info("Generating reconstructed data...")
        X_rec, _ = self.model.reconstruct(X_loader)
        return X_rec
    
    def _inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            x_hat = X.detach().cpu().numpy()
        elif isinstance(X, DataLoader):
            all_data = []
            with torch.no_grad():
                for batch in X:
                    all_data.append(batch.detach().cpu().numpy())
            x_hat = np.concatenate(all_data, axis=0)
        else:
            x_hat = X

        return self.data_components['scaler'].inverse_transform(x_hat)

    def plot_reconstruction(
        self, 
        X_true, 
        X_pred, 
        inverse_transform: bool = True,
        features_to_plot: Optional[List[str]] = None,
    ):
        # X_true should be a DataFrame
        if not isinstance(X_true, pd.DataFrame):
            raise ValueError("X_true must be a pandas DataFrame")
        
        if inverse_transform:
            self.logger.info("Inverse transforming data...")
            X_true_inverted = self._inverse_transform(X_true)
            X_pred_inverted = self._inverse_transform(X_pred)
            
        X_true = pd.DataFrame(X_true_inverted, columns=X_true.columns)
        X_pred = pd.DataFrame(X_pred_inverted, columns=X_true.columns)
        
        # Sort both DataFrames by 'x' column
        if 'x' in X_true.columns:
            X_true = X_true.sort_values(by='x')
            X_pred = X_pred.sort_values(by='x')
        else:
            raise ValueError("'x' column not found in DataFrame")
        
        features_to_plot = features_to_plot or []
        if not features_to_plot:
            self.logger.warning("No features specified for plotting.")
            return
        
        self.logger.info("Plotting reconstruction...")
        
        for col in features_to_plot:
            if col not in X_true.columns:
                self.logger.warning(f"Feature '{col}' not found in DataFrame")
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(
                X_true['x'], X_true[col], label='Original', alpha=0.8, 
                s=10, color='tab:orange', marker=MarkerStyle('o')
            )
            ax.scatter(
                X_pred['x'], X_pred[col], label='Reconstructed', alpha=0.4,
                s=10, color='tab:blue', marker=MarkerStyle('x')
            )
            ax.set_xlabel('x [mm]')
            ax.set_ylabel(rf"{col} [keV $\mu$m$^{-1}$]")
            ax.set_title(f"Reconstruction of {col}")
            ax.legend()
            ax.grid(True)
            display_plot(fig)
            
            save_figure(fig, self.config.output_dir, f"reconstruction_{col}")
            
    def plot_training_history(self, history: Optional[dict] = None):
        if history is None:
            if not self.history:
                self._load_history()
            history = self.history

        if not history:
            self.logger.warning("No training history available for plotting.")
            return
        
        self.logger.info("Plotting training history...")
        fig = plot_training_metrics(
            history=self.history,
            output_dir=self.config.output_dir,
            metrics=getattr(self.config, "metrics_to_plot", None)
        )
        display_plot(fig)
        
        save_figure(fig, self.config.output_dir, "training_history")
        self.logger.info(
            f"Training metrics plot saved to: {self.config.output_dir}"
        )

def main(config_path: Optional[Union[str, Path]] = None):
    # Load from CLI or fallback default
    if config_path is None:
        parser = argparse.ArgumentParser(
            description="VAE Post Training Analysis Script"
        )
        parser.add_argument(
            "--config_path",
            type=str,
            default=str(Path(__file__).resolve().parent / 
                        "configs/post_training_config.json"),
            help="Path to generation config JSON file",
        )
        args = parser.parse_args()
        config_path = args.config_path
        
    # Initialize the logger
    logger_obj = VAELogger(name="VAEPostTrainingAnalysis", log_level='debug')
    
    # Initialize configuration
    if config_path is None:
        raise ValueError(
            "config_path cannot be None. Please provide a valid path."
        )
    try:
        config = load_config_from_json(
            config_path=Path(config_path), config_class=PostTrainingConfig
        )
    except(FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load config: {e}") 
    
    # Preprocess data
    data_preprocessor = VAEDataPreprocessor(config, logger_obj)
    data_components = data_preprocessor.load_and_preprocess_data()
        
    # Initialize the post-analyzer
    post_analyzer = VAEPostAnalyzer(
        config=config,
        logger_obj=logger_obj,
        data_components=data_components,
    )
    
    post_analyzer.prepare_data_loaders()
    X_rec = post_analyzer.generate_reconstructed_data(post_analyzer.X_test_loader)
    
    post_analyzer.plot_reconstruction(
        X_true=data_components['test_data'],
        X_pred=X_rec,
        inverse_transform=True,
        features_to_plot=['LTT'],
    )
    
    latents = post_analyzer.extract_latents(
        X_loader=post_analyzer.X_train_loader, color_column='x'
    )
    
    # Only display figures in interactive environments, close them in CLI mode
    interactive_mode = is_interactive_environment()
    figures = post_analyzer.plot_and_save_multiple_latents(
        X_latents=latents,
        methods=['pca', 'tsne'], # add 'umap' if needed
        close_after_save=not interactive_mode
    )
    
    if interactive_mode:
        display_plot(figures)
    
    post_analyzer.plot_training_history()
    
if __name__ == "__main__":
    if "get_ipython" in globals():
        # Running in interactive mode (e.g., VSCode Interactive Window or
        # Jupyter)
        main("configs/post_training_config.json")
    else:
        # Standard command-line execution
        main()