import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.markers import MarkerStyle
from torch.utils.data import DataLoader

from configs.task_config import TrainerConfig
from core.base_pipeline import BaseVAEPipeline
from core.model_builder import build_vae_model_from_params
from core.preprocessing.data_preprocessor import VAEDataPreprocessor
from core.training_utils import build_loss_fn
from utils import (
    create_data_loaders,
    display_plot,
    ensure_directory_exists,
    load_config_from_json,
    log_params_dict,
    plot_train_test_val_distribution_df,
    plot_training_metrics,
    save_figure,
)
from utils.latent_utils import LatentSpaceMixin


class VAETrainer(BaseVAEPipeline, LatentSpaceMixin):
    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        
        self.loss_fn = None
        self.optimizer = None
        self.model_state_dict = None

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.test_rec = None
        self.latent_space = None
        
        self.color_column: Optional[str] = None
        self.latent_space_color_vals: Optional[np.ndarray] = None
        
        # Output Directory Timespamping - use resolved output_dir from BaseVAEPipeline
        if self.config.use_timestamp_output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_output_dir = self.output_dir / f"run_{timestamp}"
            ensure_directory_exists(timestamped_output_dir)
            self.output_dir = timestamped_output_dir
            # Update config to reflect the timestamped directory
            self.config.output_dir = timestamped_output_dir
        
    def prepare_data(self):
        self.logger.info("Preparing training/validation/test data...")
        preprocessor = VAEDataPreprocessor(
            config=self.config, logger_obj=self.logger_obj, mode="training"
        )
        data = preprocessor.load_and_preprocess_data()

        self.train_data = data["train_data"]
        self.val_data = data["val_data"]
        self.test_data = data["test_data"]
        self.scaler = data["scaler"]
        self.input_dim = data["n_features"]   
        self.cut_in_um = data["cut_in_um"]
        self.voxel_in_um = data['voxel_in_um']
        
        if getattr(self.config, "plot_data_splitting", False):
            # Plot how the dataset was split
            if self.config.let_type == "track":
                let_label = 'LTT'
            elif self.config.let_type == "dose":
                let_label = 'LDT'

            fig = plot_train_test_val_distribution_df(
                X_train=self.train_data,
                X_test=self.test_data,
                X_val=self.val_data,
                feature_names=['x', let_label],
            )
            display_plot(fig)
            
            save_figure(fig, self.config.output_dir, "data_scaled_splits")
        
    def setup_model(self):
        self.logger.info("Building VAE model from config parameters...")
        
        if self.input_dim is None:
            self.logger.error(
                "input_dim is not set. Cannot define network hyperparameters."
            )
            raise ValueError(
                "Missing input_dim: ensure prepare_data() is called before "
                "get_network_hyperparams()."
            )
        
        processed_dim = self.input_dim - len(self.config.identity_features)
                
        architecture_params = {
            "input_dim": self.input_dim,
            "latent_dim": self.config.latent_dim,
            "num_layers": len(self.config.net_size),
            "hidden_layers_dim": self.config.net_size,
            "processed_dim": processed_dim,
            "identity_dim": len(self.config.identity_features),
            "normalization": torch.nn.BatchNorm1d,
            "activation": torch.nn.ReLU,
            "use_dropout": self.config.use_dropout,
            "dropout_rate": self.config.dropout_rate if self.config.use_dropout else 0.0, 
        }
        log_params_dict("Network Architecture", architecture_params, self.logger)
        
        exit_activation_params = {
            "use_exit_activation": self.config.use_exit_activation,
            "exit_activation_type": self.config.exit_activation.get("type", None),
            **self.config.exit_activation.get("params", {}),
            "skip_norm_in_final": self.config.skip_norm_in_final,
        }
        log_params_dict("Exit Activation Function", exit_activation_params, self.logger)
        
        optimizer_params = {
            "optimizer": self.config.optimizer.get("optimizer", "adam"),
            "learning_rate": self.config.optimizer.get("learning_rate", 1e-3),
            "weight_decay": self.config.optimizer.get("weight_decay", 1e-5)
        }      
        log_params_dict("Optimizer", optimizer_params, self.logger)
        
        self.model = build_vae_model_from_params(
            input_dim=self.input_dim,
            n_processed_features=processed_dim,
            hyperparams={
                **architecture_params, 
                **exit_activation_params, 
                **optimizer_params},
            device=self.device
        )
        
        # Set the loss function
        loss_type = self.config.loss.get("type", "standard")
        loss_params = self.config.loss.get("params", {})
        loss_params["proc_dim"] = processed_dim
        self.loss_fn = build_loss_fn(loss_type, loss_params)
        log_params_dict("Loss", loss_params, self.logger)      
        self.model.set_loss_function(self.loss_fn)

    def prepare_data_loaders(self):
        self.logger.info("Creating data loaders...")
        loaders = create_data_loaders(
            train_data=self.train_data,
            val_data=self.val_data,
            test_data=self.test_data,
            batch_size=self.config.batch_size,
            shuffle={"train": True, "val": True, "test": False},
            data_type={"train": "training", "val": "validation", "test": "test"}        
        )
        self.train_loader = loaders.get("train")
        self.val_loader = loaders.get("val")
        self.test_loader = loaders.get("test")
        
    def train(self):
        if self.model is None:
            self.logger.error("Model is not defined, cannot start training.")
            raise ValueError("Model must be setup before training.")
        
        self.logger.info("Training the VAE model...")
        history = self.model.fit(
            trainloader=self.train_loader,
            num_epochs=self.config.training_epochs,
            valloader=self.val_loader,
            verbose=True,
            show_every=self.config.training_log_step,
            val_loss_threshold=self.config.val_loss_threshold
        )
        self.logger.info("Training completed.")
        self.history = history
        self.model_state_dict = deepcopy(self.model.state_dict())
            
    def save_training(
        self, 
        output_dir: Optional[Path] = None,
        state_dict_filename: Optional[str] = None,
        history_filename: Optional[str] = None
    ) -> None:
        if output_dir is None:
            output_dir = Path("training_results")
        if state_dict_filename is None:
            state_dict_filename = "trained_model_weights.pth"
        if history_filename is None:
            history_filename = "training_history.json"

        if self.model_state_dict is None:
            self.logger.warning(
                "No model state dict available to save. "
                "Ensure the model has been trained before saving."
            )
            return
        
        self.logger.info("Saving training results...")
        self.save_model_state_dict(
            state_dict=self.model_state_dict, 
            path=output_dir / state_dict_filename
        )
        self.save_history(path=output_dir / history_filename)
        
    def evaluate(self):
        if self.model is None:
            self.logger.error("Model is not defined, cannot evaluate.")
            raise ValueError("Model must be setup before evaluation.")
        
        self.logger.info("Evaluating model on test set...")
        self.test_rec, _ = self.model.reconstruct(self.test_loader)
        self.plot_reconstruction(features_to_plot=['LTT'])
        latents = self.extract_latents(
            X_loader=self.train_loader, color_column='x'
        )
        self.plot_and_save_multiple_latents(
            X_latents=latents,
            methods=['pca', 'tsne'], # add 'umap' if needed
        )
        
        self.logger.info("Evaluation completed.")
        
    def plot_reconstruction(
        self, 
        inverse_transform: bool = True,
        features_to_plot: Optional[List[str]] = None,
    ):
        if not isinstance(self.test_data, pd.DataFrame):
            raise ValueError("X_true must be a pandas DataFrame")
        
        if inverse_transform:
            self.logger.info("Inverse transforming data...")
            X_true_inverted = self._inverse_transform(self.test_data)
            X_rec_inverted = self._inverse_transform(self.test_rec)
            
        features_to_plot = features_to_plot or []
            
        X_true = pd.DataFrame(X_true_inverted, columns=self.test_data.columns)
        X_rec = pd.DataFrame(X_rec_inverted, columns=self.test_data.columns)
        
        # Sort both DataFrames by 'x' column
        if 'x' in X_true.columns:
            X_true = X_true.sort_values(by='x')
            X_rec = X_rec.sort_values(by='x')
        else:
            raise ValueError("'x' column not found in DataFrame")
        
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
                X_rec['x'], X_rec[col], label='Reconstructed', alpha=0.4,
                s=10, color='tab:blue', marker=MarkerStyle('x')
            )
            ax.set_xlabel('x [mm]')
            ax.set_ylabel(rf"{col} [keV $\mu$m$^{-1}$]")
            ax.set_title(f"Reconstruction of {col}")
            ax.legend()
            ax.grid(True)
            
            save_figure(fig, self.config.output_dir, f"reconstruction_{col}")
            
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
            
        if self.scaler is None:
            raise ValueError(
                "Scaler is not defined. Ensure data is processed with a scaler."
            )

        return self.scaler.inverse_transform(x_hat)

def main(config_path: Optional[Union[str, Path]]= None, verbose: int = 1):  
    # Load from CLI or fallback default
    if config_path is None:
        parser = argparse.ArgumentParser(description="VAE Training Script")
        parser.add_argument(
            "--config_path",
            type=str,
            default=str(Path(__file__).resolve().parent / "configs/trainer_config.json"),
            help="Path to training config JSON file",
        )
        args = parser.parse_args()
        config_path = args.config_path
    
    # Initialize configuration
    if config_path is None:
        raise ValueError(
            "config_path cannot be None. Please provide a valid path."
        )
    try:
        config = load_config_from_json(
            config_path=Path(config_path), config_class=TrainerConfig
        )
    except(FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load config: {e}") 

    trainer = VAETrainer(config)
    trainer.prepare_data()
    trainer.prepare_data_loaders()
    trainer.setup_model()
    trainer.train()
    trainer.save_training(
        output_dir=config.output_dir,
        state_dict_filename=config.model_weights_filename,
        history_filename=config.history_filename
    )
    plot_training_metrics(
        history=trainer.history,
        output_dir=config.output_dir,
        metrics=config.metrics_to_plot
    )
    trainer.model_summary(config.output_dir / config.model_summary_filename)
    trainer.evaluate()
        
if __name__ == '__main__':
    if "get_ipython" in globals():
        # Running in interactive mode (e.g., VSCode Interactive Window or Jupyter)
        main("configs/trainer_config.json")
    else:
        # Standard command-line execution
        main()