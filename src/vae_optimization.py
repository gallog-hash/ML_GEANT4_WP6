import argparse
import json
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Union

import optuna
import torch.nn as nn

from configs.task_config import OptunaStudyConfig
from core.base_pipeline import BaseVAEPipeline
from core.models import PELU, AutoEncoder, ELUWithLearnableOffset, ShiftedSoftplus
from core.preprocessing.data_preprocessor import VAEDataPreprocessor
from core.training_utils import build_loss_fn, create_optimizer
from utils import (
    create_data_loaders,
    ensure_directory_exists,
    load_config_from_json,
    log_params_dict,
)


class VAEOptimizer(BaseVAEPipeline):
    """
    Handles Optuna optimization for VAE.
    """

    def __init__(self, config: OptunaStudyConfig):
        super().__init__(config)

        self.best_model_state_dict = None
        self.best_trial_number = None
        self.best_loss = float("inf")

        # Output Directory Timespamping - use resolved optuna_io_dir from BaseVAEPipeline
        self.output_dir_parent = self.optuna_io_dir
        if self.config.use_timestamp_output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_output_dir = self.optuna_io_dir / f"run_{timestamp}"
            ensure_directory_exists(timestamped_output_dir)
            self.optuna_io_dir = timestamped_output_dir
            # Update config to reflect the timestamped directory
            self.config.optuna_io_dir = timestamped_output_dir

    def prepare_data(self):
        preprocessor = VAEDataPreprocessor(
            config=self.config, logger_obj=self.logger_obj, mode="training"
        )
        self.data_components = preprocessor.load_and_preprocess_data()
        self.input_dim = self.data_components["n_features"]

    def create_study(self) -> optuna.Study:
        """Create Optuna study with persistent storage."""
        study_name = self.config.study_name or "vae_optimization"
        db_name = self.config.database or "optuna_study.db"

        # Default path (if study_path not set)
        default_study_path = (
            f"sqlite:///{(self.config.optuna_io_dir / db_name).resolve()}"
        )

        # Start with what’s in config or fallback to default
        storage_path = self.config.study_path or default_study_path

        # Ensure storage path is valid for Optuna
        if not storage_path.startswith("sqlite:///"):
            self.logger.error(
                "Storage path must start with 'sqlite:///' for Optuna studies."
            )
            raise ValueError("Invalid storage path for Optuna study.")

        if not storage_path.endswith(".db"):
            self.logger.warning(
                "Storage path should end with '.db' to indicate a SQLite database."
            )

        # Handle potential mismatch between timestamping and hardcoded path
        if self.config.study_path and self.config.use_timestamp_output_dir:
            self.logger.warning(
                "Using timestamped output directory may conflict with study storage path."
            )
            # Replace parent with timestamped path (as string)
            storage_path = str(self.config.study_path).replace(
                str(self.output_dir_parent), str(self.config.optuna_io_dir)
            )
            self.config.study_path = storage_path
            self.logger.info(f"Updated storage path to: {storage_path}")

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        return optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
            direction="minimize",
            pruner=pruner,
        )

    def get_network_hyperparams(self, trial):
        num_layers = trial.suggest_int("num_layers", 1, 5)

        if self.input_dim is None:
            self.logger.error(
                "input_dim is not set. Cannot define network hyperparameters."
            )
            raise ValueError(
                "Missing input_dim: ensure prepare_data() is called before "
                "get_network_hyperparams()."
            )

        processed_dim = self.input_dim - len(self.config.identity_features)
        net_size = []

        for i in range(num_layers):
            if i == 0:
                # First layer: between processed_dim and processed_dim^2 / 2
                layer_size = trial.suggest_int(
                    "layer_0_size", processed_dim, processed_dim**2 // 2, step=16
                )
            else:
                # Subsequent layers: Between half of the previous layer and the
                # previous layer
                lower_bound = max(16, net_size[-1] // 2)
                upper_bound = net_size[-1]
                layer_size = trial.suggest_int(
                    f"layer_{i}_size", lower_bound, upper_bound, step=16
                )

            net_size.append(layer_size)

        latent_dim = trial.suggest_int(
            "latent_dim", 4, min(processed_dim, net_size[-1]), step=4
        )

        trial.set_user_attr("input_dim", self.input_dim)
        trial.set_user_attr("hidden_layers_dim", net_size)
        trial.set_user_attr("processed_dim", processed_dim)

        # Suggest dropout usage
        if self.config.use_default_use_dropout:
            use_dropout = self.config.default_use_dropout
        else:
            use_dropout = trial.suggest_categorical("use_dropout", [True, False])
        dropout_rate = (
            trial.suggest_float("dropout_rate", 0.01, 0.5) if use_dropout else 0.0
        )

        # Set dropout control as user attributes
        trial.set_user_attr("use_dropout", use_dropout)
        trial.set_user_attr("dropout_rate", dropout_rate)

        log_params_dict(
            "Network Architecture",
            {
                "input_dim": self.input_dim,
                "latent_dim": latent_dim,
                "num_layers": num_layers,
                "hidden_layers_dim": net_size,
                "processed_dim": processed_dim,
                "use_dropout": use_dropout,
                "dropout_rate": dropout_rate,
            },
            logger=self.logger,
        )

        return {
            "input_dim": self.input_dim,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "hidden_layers_dim": net_size,
            "processed_dim": processed_dim,
            "use_dropout": use_dropout,
            "dropout_rate": dropout_rate,
        }

    def get_exit_activation(self, trial):
        if self.config.use_default_use_exit_activation:
            use_exit_activation = self.config.default_use_exit_activation
            trial.set_user_attr("use_exit_activation", use_exit_activation)
        else:
            use_exit_activation = trial.suggest_categorical(
                "use_exit_activation", [True, False]
            )
        activation_choice = trial.suggest_categorical(
            "exit_activation_type", ["shifted_softplus", "elu_offset", "pelu"]
        )

        activation_params = {}  # To store parameters for logging

        if activation_choice == "shifted_softplus":
            beta_softplus = trial.suggest_float("beta_softplus", 0.5, 5.0, log=True)
            trial.set_user_attr("beta_softplus", beta_softplus)
            activation_params["beta_softplus"] = beta_softplus
            activation_type = partial(ShiftedSoftplus, beta=beta_softplus)

        elif activation_choice == "elu_offset":
            offset_init = trial.suggest_float("offset_init", 0.0, 2.0)
            trial.set_user_attr("offset_init", offset_init)
            activation_params["offset_init"] = offset_init
            activation_type = partial(ELUWithLearnableOffset, offset_init=offset_init)

        elif activation_choice == "pelu":
            a_init = trial.suggest_float("a_init", 0.1, 5.0)
            b_init = trial.suggest_float("b_init", 0.1, 5.0)
            trial.set_user_attr("a_init", a_init)
            trial.set_user_attr("b_init", b_init)
            activation_params["a_init"] = a_init
            activation_params["b_init"] = b_init
            activation_type = partial(PELU, a_init=a_init, b_init=b_init)

        log_params_dict(
            "Output Activation",
            {
                "use_exit_activation": use_exit_activation,
                "exit_activation_type": activation_type,
                **activation_params,
            },
            logger=self.logger,
        )

        trial.set_user_attr(
            "skip_norm_in_final", self.config.default_skip_norm_in_final
        )

        return {
            "use_exit_activation": use_exit_activation,
            "exit_activation_type": activation_type,
            "skip_norm_in_final": self.config.default_skip_norm_in_final,
        }

    def get_loss_hyperparams(self, trial, loss_type):
        # Priority: if using MinMax, disable neg penalty regardless of default
        # flag
        if isinstance(self.config.scaler, str):
            scaler_type = self.config.scaler
        else:
            scaler_type = self.config.scaler.get("type", "unknown")

        if scaler_type.lower() == "minmax":
            use_neg_penalty = False
        elif self.config.use_default_use_negative_penalty:
            use_neg_penalty = self.config.default_use_negative_penalty
        else:
            use_neg_penalty = trial.suggest_categorical(
                "use_neg_penalty", [True, False]
            )

        neg_penalty_weight = (
            trial.suggest_float("neg_penalty_weight", 0.1, 5.0, log=True)
            if use_neg_penalty
            else 0.0
        )

        trial.set_user_attr("use_neg_penalty", use_neg_penalty)
        trial.set_user_attr("neg_penalty_weight", neg_penalty_weight)

        loss_hyperparams = {
            "loss_type": loss_type,
            "use_neg_penalty": use_neg_penalty,
            "neg_penalty_weight": neg_penalty_weight,
            "proc_dim": self.data_components["n_features"]
            - len(self.config.identity_features),
        }

        if scaler_type == "standard" and use_neg_penalty:
            loss_hyperparams["scaler_mean"] = self.data_components["scaler"].mean_
            loss_hyperparams["scaler_scale"] = self.data_components["scaler"].scale_

        if loss_type == "standard":
            loss_hyperparams["beta"] = trial.suggest_float("beta", 0.1, 5.0, log=True)

        if loss_type in ["inverse", "sigmoid"]:
            loss_hyperparams["beta_scale"] = trial.suggest_float(
                "beta_scale", 0.1, 50.0, log=True
            )

        if loss_type == "sigmoid":
            loss_hyperparams["recon_target"] = trial.suggest_float(
                "recon_target", 1e-4, 1e-2, log=True
            )

        log_params_dict("Loss", loss_hyperparams, self.logger)
        return loss_hyperparams

    def get_optimizer_hyperparams(self, trial):
        optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)

        log_params_dict(
            "Optimizer",
            {
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            logger=self.logger,
        )

        return {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }

    def train_vae_model(
        self, model, train_loader, val_loader, trial, val_loss_threshold
    ):
        history = model.fit(
            trainloader=train_loader,
            num_epochs=self.config.default_training_epochs,
            valloader=val_loader,
            show_every=self.config.default_training_log_step,
            verbose=True,
            trial=trial,
            val_loss_threshold=val_loss_threshold,
        )
        return history

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optimization objective for VAE hyperparameters.

        Args:
            trial (optuna.Trial): A trial object for parameter suggestions.

        Returns:
            float: Final validation loss for the trial.
        """
        # Tune batch size as a hyperparameter
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

        # Create DataLoaders dynamically based on tuned batch size
        loaders = create_data_loaders(
            train_data=self.data_components["train_data"],
            val_data=self.data_components["val_data"],
            batch_size=batch_size,
            shuffle={"train": True, "val": True},
            data_type={"train": "training", "val": "validation"},
        )
        train_loader = loaders.get("train")
        val_loader = loaders.get("val")

        network_hparams = self.get_network_hyperparams(trial)
        exit_layer_hparams = self.get_exit_activation(trial)
        net_params = {
            **network_hparams,
            "identity_dim": len(self.config.identity_features),
            "normalization": nn.BatchNorm1d,
            "activation": nn.ReLU,
            **exit_layer_hparams,
        }

        optimizer_hparams = self.get_optimizer_hyperparams(trial)

        loss_type = trial.suggest_categorical(
            "loss_type", ["standard", "inverse", "sigmoid"]
        )
        loss_hyperparams = self.get_loss_hyperparams(trial, loss_type)

        model = AutoEncoder(architecture_params=net_params, device=self.device).to(
            self.device
        )

        optimizer = create_optimizer(model.parameters(), optimizer_hparams)
        loss_fn = build_loss_fn(loss_type=loss_type, loss_params=loss_hyperparams)

        model.set_optimizer(optimizer)
        model.set_loss_function(loss_fn)

        # Train the model, passing the trial for pruning.
        history = self.train_vae_model(
            model, train_loader, val_loader, trial, self.config.val_loss_threshold
        )

        # Use the final validation loss as the objective
        final_val_loss = (
            history["val_loss"][-1] if history["val_loss"] else float("inf")
        )

        if final_val_loss < self.best_loss:
            self.history = history
            self.best_loss = final_val_loss
            self.best_trial_number = trial.number
            self.best_model_state_dict = deepcopy(model.state_dict())
            self.model = model
            self.logger.info(
                f"Best model state dictionary updated at trial {trial.number}"
            )

        return final_val_loss

    def save_hparams_config(self, trial, save_path="best_hyperparameters_config.json"):
        complete_hparams = trial.params.copy()
        user_attrs = trial.user_attrs
        complete_hparams.update(user_attrs)
        with open(save_path, "w") as f:
            json.dump(complete_hparams, f, indent=2)
        self.logger.info(f"Hyperparameters configuration saved to: {save_path}")


def main(config_path: Optional[Union[str, Path]] = None):
    # Load from CLI or fallback default
    if config_path is None:
        parser = argparse.ArgumentParser(description="VAE Optimization Script")
        parser.add_argument(
            "--config_path",
            type=str,
            default=str(Path(__file__).resolve().parent / "configs/optuna_config.json"),
            help="Path to optimization config JSON file",
        )
        args = parser.parse_args()
        config_path = args.config_path

    # Initialize configuration
    if config_path is None:
        raise ValueError("config_path cannot be None. Please provide a valid path.")
    try:
        config = load_config_from_json(
            config_path=Path(config_path), config_class=OptunaStudyConfig
        )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load config: {e}")

    # Setup optimizer
    optimizer = VAEOptimizer(config)
    optimizer.prepare_data()

    study = optimizer.create_study()

    # Optimize
    study.optimize(optimizer.objective, n_trials=config.n_trials)

    optimizer.model_summary(config.optuna_io_dir / config.model_summary_filename)
    optimizer.save_model_state_dict(
        state_dict=optimizer.best_model_state_dict,  # type: ignore
        path=config.optuna_io_dir / config.model_weights_filename,
    )
    optimizer.save_hparams_config(
        trial=study.best_trial,
        save_path=str(config.optuna_io_dir / config.hparams_config_filename),
    )
    optimizer.save_history(path=config.optuna_io_dir / config.history_filename)


if __name__ == "__main__":
    if "get_ipython" in globals():
        # Running in interactive mode (e.g., VSCode Interactive Window or Jupyter)
        main("configs/optuna_config.json")
    else:
        # Standard command-line execution
        main()
