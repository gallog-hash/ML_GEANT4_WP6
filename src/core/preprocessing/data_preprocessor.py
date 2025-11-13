# src/core/preprocessing/data_preprocessor.py

from typing import Any, Dict

import pandas as pd
import torch

from core.preprocessing.data_utils import split_and_scale_dataset
from core.preprocessing.preprocessing_utils import (
    change_default_settings,
    import_and_clean_data,
    reorder_identity_features,
)
from utils import VAELogger, resolve_path_with_project_root

VALID_MODES = ["training", "inference"]


class VAEDataPreprocessor:
    """
    Handles data loading, preprocessing, and splitting for VAE tasks.
    """

    def __init__(self, config: Any, logger_obj: VAELogger, mode: str = "training"):
        self.config = config
        self.logger = logger_obj.get_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}' provided to VAEDataPreprocessor. "
                f"Valid modes are {VALID_MODES}."
            )

    def _downsample_data(self, data, factor: int):
        """
        Downsample high-density data by selecting every nth element.

        Args:
            data (numpy.ndarray or pandas.DataFrame): The input high-density data.
            factor (int): The downsampling factor (e.g., 1000 to go from micrometer
            to millimeter sampling).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - downsampled data (every nth element)
                - remainder data (elements not included in the downsampled set)
        """
        downsampled = data[::factor]
        remainder = data.drop(index=downsampled.index)
        return downsampled, remainder

    def load_and_preprocess_data(self) -> Dict[str, Any]:
        """
        Load and preprocess the dataset based on the given configuration.

        Returns:
            dict: Dictionary containing 'train_data', 'val_data', 'test_data',
            'scaler', 'n_features'.
        """
        # Seed setting
        change_default_settings(self.config.random_seed)

        # Determine which data directory and file to use
        input_mode = getattr(self.config, "input_mode", "downsample")

        if self.mode == "inference" and input_mode == "direct":
            # Use low-res data directory if specified
            data_dir_path = getattr(self.config, "lowres_data_dir", None)
            data_file_name = getattr(self.config, "lowres_data_file", None)

            if data_dir_path is None:
                data_dir_path = self.config.data_dir
            if data_file_name is None:
                data_file_name = self.config.data_file

            self.logger.info(
                f">>> Using direct low-resolution input mode <<<\n"
                f"Data directory: {data_dir_path}\n"
                f"Data file: {data_file_name}"
            )
        else:
            # Use standard data directory and file
            data_dir_path = self.config.data_dir
            data_file_name = self.config.data_file

        # Load and clean data
        data_dir = resolve_path_with_project_root(
            data_dir_path, getattr(self.config, "project_root", None)
        )
        df, cut_in_um, voxel_in_um = import_and_clean_data(
            data_dir=str(data_dir),
            data_file=data_file_name,
            primary_particle=self.config.primary_particle,
            let_type=self.config.let_type,
            cut_with_primary=self.config.cut_with_primary,
            drop_zero_cols=self.config.drop_zero_cols,
            drop_zero_thr=self.config.drop_zero_thr,
            verbose=0,
        )

        # Reorder identity features
        df = reorder_identity_features(df, self.config.identity_features)

        # Handle inference mode data preparation
        if self.mode == "inference":
            if input_mode == "downsample":
                # Existing behavior: downsample high-res data
                downsample_factor = getattr(self.config, "downsample_factor", 20)
                self.logger.info(
                    f">>> Downsampling high-res data for inference "
                    f"(factor={downsample_factor})... <<<"
                )
                low_density_data, remainder_data = self._downsample_data(
                    df, factor=downsample_factor
                )
                self.logger.debug(
                    f"Loaded data shape: {df.shape}\n"
                    f"Downsampled data shape: {low_density_data.shape}\n"
                    f"Remainder data shape: {remainder_data.shape}"
                )
            elif input_mode == "direct":
                # New behavior: use low-res data directly
                self.logger.info(
                    ">>> Using low-resolution data directly (no downsampling) <<<"
                )
                low_density_data = df
                remainder_data = None  # No ground truth for comparison
                self.logger.debug(
                    f"Loaded low-res data shape: {low_density_data.shape}"
                )
            else:
                raise ValueError(
                    f"Invalid input_mode: '{input_mode}'. "
                    f"Must be 'downsample' or 'direct'."
                )

        # Split and scale dataset
        if isinstance(self.config.scaler, str):
            scaler_type = self.config.scaler
            scaler_kwargs = {}
        else:
            scaler_type = self.config.scaler.get("type", "minmax")
            scaler_kwargs = {k: v for k, v in self.config.scaler.items() if k != "type"}

        if self.mode == "training":
            train_data, val_data, test_data, scaler = split_and_scale_dataset(
                let_df=df,
                random_seed=self.config.random_seed,
                train_size=self.config.train_size,
                val_size=self.config.val_size,
                test_size=self.config.test_size,
                single_scaler=True,
                scaler_type=scaler_type,
                **scaler_kwargs,
            )

            self.logger.info(
                f"Data loaded and preprocessed. Train size: {len(train_data)}, "
                f"Val size: {len(val_data)}, Test size: {len(test_data)}"
            )

            return {
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "scaler": scaler,
                "n_features": train_data.shape[1],
                "cut_in_um": cut_in_um,
                "voxel_in_um": voxel_in_um,
            }

        elif self.mode == "inference":
            X_scaled, _, _, scaler = split_and_scale_dataset(
                let_df=low_density_data,
                random_seed=self.config.random_seed,
                train_size=self.config.train_size,
                val_size=self.config.val_size,
                test_size=self.config.test_size,
                shuffle=False,
                single_scaler=True,
                scaler_type=scaler_type,
                **scaler_kwargs,
            )

            # Only transform remainder data if it exists (downsample mode)
            if remainder_data is not None:
                R_scaled = pd.DataFrame(
                    scaler.transform(remainder_data), columns=remainder_data.columns
                )
                self.logger.debug(
                    f"X_scaled data type: {type(X_scaled)}, "
                    f"shape: {X_scaled.shape}\n"
                    f"R_scaled type: {type(R_scaled)}, "
                    f"shape: {R_scaled.shape}",
                )
            else:
                R_scaled = None
                self.logger.debug(
                    f"X_scaled data type: {type(X_scaled)}, "
                    f"shape: {X_scaled.shape}\n"
                    f"R_scaled: None (direct input mode, no ground truth)"
                )

            return {
                "X_input": low_density_data,
                "X_scaled": X_scaled,
                "R_input": remainder_data,
                "R_scaled": R_scaled,
                "scaler": scaler,
                "n_input_features": X_scaled.shape[1],
                "cut_in_um": cut_in_um,
                "voxel_in_um": voxel_in_um,
            }
