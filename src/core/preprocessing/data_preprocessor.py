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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

        if self.mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}' provided to VAEDataPreprocessor. "
                            f"Valid modes are {VALID_MODES}.")

        
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

        # Load and clean data
        data_dir = resolve_path_with_project_root(
            self.config.data_dir, 
            getattr(self.config, "project_root", None)
        )
        df, cut_in_um, voxel_in_um = import_and_clean_data(
            data_dir=str(data_dir),
            data_file=self.config.data_file,
            primary_particle=self.config.primary_particle,
            let_type=self.config.let_type,
            cut_with_primary=self.config.cut_with_primary,
            drop_zero_cols=self.config.drop_zero_cols,
            drop_zero_thr=self.config.drop_zero_thr,
            verbose=0
        )

        # Reorder identity features
        df = reorder_identity_features(df, self.config.identity_features)
        
        # Downsample data if inference mode is set
        if self.mode == "inference":
            self.logger.info(">>> Downsampling data for inference... <<<")
            low_density_data, remainder_data = self._downsample_data(
                df, factor=self.config.resample_factor
            )
            self.logger.debug(
                f"Loaded data shape: {df.shape}\n"
                f"Downsampled data shape: {low_density_data.shape}\n"
                f"Remainder data shape: {remainder_data.shape}"
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
                **scaler_kwargs
            )

            self.logger.info(
                f"Data loaded and preprocessed. Train size: {len(train_data)}, "
                f"Val size: {len(val_data)}, Test size: {len(test_data)}"
            )

            return {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'scaler': scaler,
                'n_features': train_data.shape[1],
                'cut_in_um': cut_in_um,
                'voxel_in_um': voxel_in_um,
            }
            
        elif self.mode == "inference":
            X_scaled, _, _, scaler = \
                split_and_scale_dataset(
                    let_df=low_density_data,
                    random_seed=self.config.random_seed,
                    train_size=self.config.train_size,
                    val_size=self.config.val_size,
                    test_size=self.config.test_size,
                    shuffle=False,
                    single_scaler=True,
                    scaler_type=scaler_type,
                    **scaler_kwargs
                )
            R_scaled = pd.DataFrame(
                scaler.transform(remainder_data),
                columns=remainder_data.columns
            )
            self.logger.debug(
                f"X_scaled data type: {type(X_scaled)}, shape: {X_scaled.shape}\n"
                f"R_scaled type: {type(R_scaled)}, shape: {R_scaled.shape}",
            )
            
            return {
                'X_input': low_density_data,
                'X_scaled': X_scaled,
                'R_input': remainder_data,
                'R_scaled': R_scaled,
                'scaler': scaler,
                'n_input_features': X_scaled.shape[1],
            }
