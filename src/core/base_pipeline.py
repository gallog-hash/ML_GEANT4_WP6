# src/core/base_pipeline.py

import json
from pathlib import Path
from typing import Optional

import torch
from torchinfo import summary

from configs.task_config import PostTrainingConfig
from utils import VAELogger, ensure_directory_exists, resolve_path_with_project_root


class BaseVAEPipeline:
    def __init__(self, config, logger_obj: Optional[VAELogger] = None):
        self.logger_obj = logger_obj or VAELogger("VAETrainer", "debug")
        self.logger = self.logger_obj.get_logger()
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        project_root = getattr(self.config, "project_root", None)
        
        # Resolve output_dir relative to project_root if specified
        if hasattr(self.config, "output_dir"):
            output_dir = getattr(self.config, "output_dir", "outputs")
            self.output_dir = resolve_path_with_project_root(output_dir, project_root)
            ensure_directory_exists(self.output_dir)
            self.config.output_dir = self.output_dir  # Update config with resolved path
        
        self.input_dim: Optional[int] = None
        self.model: Optional[torch.nn.Module] = None
        self.history: dict = {}
        
        # Handle optuna_io_dir only if explicitly defined in config
        if hasattr(self.config, "optuna_io_dir") and self.config.optuna_io_dir:
            self.optuna_io_dir = resolve_path_with_project_root(
                self.config.optuna_io_dir, project_root
            )
            ensure_directory_exists(self.optuna_io_dir)
            # Update config with resolved path
            self.config.optuna_io_dir = self.optuna_io_dir  
        
        # Setup study_path for Optuna if applicable
        if hasattr(self, "optuna_io_dir"):
            self.database = getattr(self.config, "database", "database.db")
            self.study_path = "sqlite:///" + str(
                self.optuna_io_dir / self.database
            )
            self.config.study_path = self.study_path  # Update config with resolved path
        
        if isinstance(self.config, PostTrainingConfig):
            self._expand_optuna_io_to_full_path()
        
    def _expand_optuna_io_to_full_path(self) -> None:
        self.config.hparams_config_path = \
            self.optuna_io_dir / self.config.hparams_config_filename
        self.config.model_path = self.optuna_io_dir / self.config.model_weights_filename
        self.config.history_path = self.optuna_io_dir / self.config.history_filename
        
    def model_summary(self, export_path: Optional[Path] = None):
        if not self.model:
            self.logger.warning("No model defined to generate summary.")
            return
        
        if self.input_dim is None:
            self.logger.warning(
                "Cannot generate model summary: input_dim is not set."
            )
            return
        
        self.logger.info("Generating model summary...")
        model_summary = summary(
            model=self.model,
            input_size=(1, self.input_dim),
            depth=4
        )
        self.logger.info(f"\n{model_summary}")
        
        if export_path:
            self.logger.info(f"Exporting model summary to {export_path}...")
            with open(export_path, 'w', encoding='utf-8') as file:
                file.write(str(model_summary))

    def save_model_state_dict(
        self, 
        state_dict: dict, 
        path: Optional[Path] = None
    ) -> None:
        if state_dict is None:
            state_dict = self.model.state_dict() if self.model else None

        if state_dict is None:
            self.logger.warning(
                "No state dict provided or model not initialized. "
                "Skipping model save.")
            return
        
        path = path or self.output_dir / "model_weights.pth"

        torch.save(state_dict, path)
        self.logger.info(f"Model state dict saved to: {path}")
        
    def load_model_state_dict(self, path: Path) -> None:
        if not self.model:
            raise ValueError("Model must be initialized before loading state dict.")

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f"Model state dict loaded from: {path}")

    def save_history(
        self, 
        history: Optional[dict] = None, 
        path: Optional[Path] = None
    ) -> None:
        history = history or self.history
        if not history:
            self.logger.warning("No training history to save.")
            return

        path = path or self.output_dir / "training_history.json"

        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)
        self.logger.info(f"Training history saved to: {path}")
