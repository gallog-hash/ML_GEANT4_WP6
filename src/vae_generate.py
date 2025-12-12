import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.markers import MarkerStyle
from scipy.stats import ks_2samp, wasserstein_distance
from torch.utils.data import DataLoader

from configs.task_config import GenerationConfig
from core.base_pipeline import BaseVAEPipeline
from core.bragg_peak_metrics import BraggPeakMetrics
from core.metrics import PointwiseMetrics
from core.preprocessing.data_preprocessor import VAEDataPreprocessor
from utils import (
    OptionalTimer,
    VAELogger,
    create_data_loaders,
    display_plot,
    ensure_directory_exists,
    load_config_with_profile,
    load_model,
    load_model_config,
    resolve_path_with_project_root,
    save_figure,
)


class VAEGenerate(BaseVAEPipeline):
    def __init__(
        self,
        config: GenerationConfig,
        data_components: dict,
    ):
        # Temporarily hide output_dir if enable_plots is False to prevent
        # directory creation
        output_dir_backup = None
        if not config.enable_plots and hasattr(config, "output_dir"):
            output_dir_backup = config.output_dir
            delattr(config, "output_dir")

        # Initialize BaseVAEPipeline (handles path resolution, logging,
        # etc.)
        super().__init__(config)

        # Restore output_dir attribute (but don't create directory)
        if output_dir_backup is not None:
            config.output_dir = output_dir_backup

        self.data_components = data_components

        self.hparam_config = load_model_config(
            config_path=self.config.hparams_config_path
        )
        self.model = load_model(
            weights_path=self.config.model_path,
            device=self.device,
            params=self.hparam_config,
        )
        self.model.eval()

    def prepare_data_loaders(self):
        self.logger.info("Creating data loaders...")
        loaders = create_data_loaders(
            test_data=self.data_components["X_scaled"],
            batch_size=self.hparam_config["batch_size"],
            shuffle={"test": False},
            data_type={"test": "inference"},
        )
        self.X_scaled_loader = loaders["test"]

    def get_latents(
        self,
        input_data_loader,
    ):
        latent_codes_list = []
        identity_list = []

        # Iterate over the low-resolution DataLoader to extract latent codes and
        # identity features.
        for batch in input_data_loader:
            batch = batch.to(self.device)
            if self.model.identity_dim > 0:
                # Split into processed and identity parts.
                x_proc = batch[:, : self.model.proc_dim]
                x_identity = batch[:, self.model.proc_dim :]
            else:
                x_proc = batch
                x_identity = None

            with torch.no_grad():
                mu, logvar = self.model.encoder(x_proc)
                # latent_code = self.model.reparameterize(mu, logvar)
                latent_code = mu

            latent_codes_list.append(latent_code.cpu())
            if x_identity is not None:
                identity_list.append(x_identity.cpu())

        # Concatenate latent codes from all batches.
        latent_codes = torch.cat(latent_codes_list, dim=0)
        # shape: (M, latent_dim)
        if identity_list:
            identity_uncoded = torch.cat(identity_list, dim=0)
            # shape: (M, identity_dim)
            return latent_codes, identity_uncoded

        return latent_codes

    def interpolate_latent_codes(
        self, latent_codes: torch.Tensor, upsample_factor: int
    ) -> torch.Tensor:
        """
        Interpolate between adjacent latent codes to upsample the latent sequence.

        Args:
            latent_codes (torch.Tensor): Tensor of shape (M, latent_dim)
            representing the original latent codes.
            upsample_factor (int): Total number of segments between each
            pair of latent codes. For example, if upsample_factor = 1000,
            then 999 latent codes are generated between each pair.

        Returns:
            torch.Tensor: A new tensor of latent codes with increased
            resolution.
        """
        m, latent_dim = latent_codes.size()
        # List to store the new latent codes
        new_latents = []

        for i in range(m - 1):
            v1 = latent_codes[i]
            v2 = latent_codes[i + 1]
            # Append the original latent code
            new_latents.append(v1)
            # Generate upsample_factor - 1 interpolated points between v1 and v2.
            # We generate weights between 0 and 1, excluding the endpoints.
            weights = torch.linspace(
                0, 1, steps=upsample_factor + 1, device=latent_codes.device
            )[1:-1]
            # Compute interpolated latent vectors.
            for w in weights:
                interp = (1 - w) * v1 + w * v2
                new_latents.append(interp)

        # Append the last original latent code.
        new_latents.append(latent_codes[-1])

        return torch.stack(new_latents, dim=0)

    def interpolate_identity(
        self, identity_tensor: torch.Tensor, upsample_factor: int
    ) -> torch.Tensor:
        """
        Interpolate between adjacent identity vectors to upsample the identity
        sequence.

        Args:
            identity_tensor (torch.Tensor): Tensor of shape (M, d) containing the
                original identity features.
            upsample_factor (int): Total number of segments between each
                pair of identity vectors. For example, if
                upsample_factor=1000, then 999 intermediate vectors are
                generated between each pair.

        Returns:
            torch.Tensor: A new tensor of identity features with
            increased resolution.
        """
        m, d = identity_tensor.size()
        new_identity_list = []
        for i in range(m - 1):
            v1 = identity_tensor[i]
            v2 = identity_tensor[i + 1]
            # Append the original identity vector
            new_identity_list.append(v1)
            # Generate upsample_factor - 1 interpolated identity vectors
            # between v1 and v2.
            weights = torch.linspace(
                0, 1, steps=upsample_factor + 1, device=identity_tensor.device
            )[1:-1]
            for w in weights:
                interp = (1 - w) * v1 + w * v2
                new_identity_list.append(interp)
        new_identity_list.append(identity_tensor[-1])
        return torch.stack(new_identity_list, dim=0)

    def generate_high_res(
        self,
    ):
        """
        Pass
        """
        result = self.get_latents(self.X_scaled_loader)
        if isinstance(result, tuple):
            latent_codes, identity_uncoded = result
        else:
            latent_codes = result

        # Interpolate between latent codes using the desired upsampling factor.
        if latent_codes.size(0) > 1:
            interpolated_latents = self.interpolate_latent_codes(
                latent_codes, self.config.upsample_factor
            )
            with torch.no_grad():
                decoded_processed = self.model.decoder(
                    interpolated_latents.to(self.device)
                )
                # Clamp negative values in processed features if enabled
                if self.config.clamp_negatives:
                    decoded_processed = torch.clamp(decoded_processed, min=0.0)
                decoded_processed = decoded_processed.cpu()
        else:
            # If only one sample, there's nothing to interpolate.
            decoded_processed = torch.empty(0)

        # Process identity features:
        if isinstance(result, tuple):
            # Upsample identity features using linear interpolation.
            upsampled_identity = self.interpolate_identity(
                identity_uncoded, self.config.upsample_factor
            ).to(decoded_processed.device)
        else:
            upsampled_identity = None

        # Concatenate the decoded processed features with the upsampled identity
        # features.
        if upsampled_identity is not None:
            # Ensure that the number of rows matches.
            high_res_output = torch.cat([decoded_processed, upsampled_identity], dim=1)
        else:
            high_res_output = decoded_processed

        # Convert the generated output (a tensor) into a DataFrame with the same
        # columns as the input.
        high_res_df = pd.DataFrame(
            high_res_output.numpy(),
            columns=(
                self.data_components["X_input"].columns
                if isinstance(self.data_components["X_input"], pd.DataFrame)
                else None
            ),
        )

        return high_res_df

    def inverse_transform(self, X):
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

        return pd.DataFrame(
            self.data_components["scaler"].inverse_transform(x_hat),
            columns=self.data_components["X_input"].columns,
        )

    def export_to_let_file(
        self, data_df: pd.DataFrame, output_filename: Optional[str] = None
    ):
        """
        Export the upsampled LET data to a textual file in the same
        format as the input Let.out file.

        Args:
            data_df (pd.DataFrame): DataFrame containing the upsampled
                data to export
            output_filename (str, optional): Name of the output file. If
                None, generates filename automatically based on
                upsample_factor (e.g., "Let_upsampled_factor_50.out").
                Default: None
        """
        # Generate filename automatically if not provided
        if output_filename is None:
            output_filename = f"Let_upsampled_factor_{self.config.upsample_factor}.out"
            self.logger.info(f"No output filename provided. Using: {output_filename}")
        # Create a copy to avoid modifying the original DataFrame
        export_df = data_df.copy()

        # Recreate the original i, j, k columns that were dropped during import
        if "x" in export_df.columns:
            # Get voxel size from data components
            voxel_size_um = self.data_components.get("voxel_in_um", 1.0)

            # Convert voxel size from um to mm
            voxel_size_mm = voxel_size_um * 1e-3

            # Convert x (mm) back to i (index)
            # The original conversion was: x = voxel_size_mm * i / unit_to_mm[output_size]
            # Where unit_to_mm['mm'] = 1.0, so: x = voxel_size_mm * i
            # Therefore: i = x / voxel_size_mm
            export_df["i"] = (export_df["x"] / voxel_size_mm).round().astype(int)

            # Add j and k columns (always 0 for 1D simulation)
            export_df["j"] = 0
            export_df["k"] = 0

            # Remove the 'x' column since original format uses 'i'
            export_df = export_df.drop(columns=["x"])

            # Reorder columns to match original format: i, j, k, then other columns
            other_columns = [
                col for col in export_df.columns if col not in ["i", "j", "k"]
            ]
            column_order = ["i", "j", "k"] + other_columns
            export_df = export_df[column_order]

        # Use the project root from config and create the
        # vae_generate_output directory as requested
        project_root = getattr(self.config, "project_root", None)
        output_dir = resolve_path_with_project_root("vae_generate_output", project_root)
        output_path = output_dir / output_filename

        # Ensure the output directory exists using the utility function
        ensure_directory_exists(output_dir)

        # Write to file with tab-separated values and no index
        export_df.to_csv(output_path, sep="\t", index=False, float_format="%.5g")

        self.logger.info(f"Upsampled LET data exported to: {output_path}")
        return output_path


class VAEGeneratorAnalysis:
    def __init__(
        self,
        config: GenerationConfig,
        logger,
        X_gen: pd.DataFrame,
        X_rec: pd.DataFrame,
        X_input: pd.DataFrame,
        X_input_gaps: Union[None, pd.DataFrame],
        voxel_size_um: Optional[float] = None,
    ):
        self.config = config
        self.logger = logger
        self.X_gen = X_gen
        self.X_rec = X_rec
        self.X_input = X_input
        self.X_input_gaps = X_input_gaps
        self.voxel_size_um = voxel_size_um

    def get_resolution_info(self) -> Dict[str, Any]:
        """
        Get resolution and sampling information for context.

        Returns:
            dict: Dictionary with resolution metadata including:
                - upsample_factor: Interpolation factor used
                - downsample_factor: Downsampling factor (if applicable)
                - input_mode: "downsample" or "direct"
                - voxel_size_um: Original voxel size in micrometers
                - n_input: Number of low-res input samples
                - n_generated: Number of generated interpolated samples
                - n_ground_truth: Number of ground truth samples (if
                  available)
        """
        info = {
            "upsample_factor": self.config.upsample_factor,
            "downsample_factor": self.config.downsample_factor,
            "input_mode": self.config.input_mode,
            "voxel_size_um": self.voxel_size_um,
            "n_input": len(self.X_input),
            "n_reconstructed": len(self.X_rec),
            "n_generated": len(self.X_gen),
        }

        if self.X_input_gaps is not None:
            info["n_ground_truth"] = len(self.X_input_gaps)

        # Calculate effective resolutions
        if self.voxel_size_um is not None:
            if self.config.downsample_factor is not None:
                info["lowres_voxel_size_um"] = (
                    self.voxel_size_um * self.config.downsample_factor
                )
            info["target_voxel_size_um"] = (
                self.voxel_size_um if self.config.input_mode == "downsample" else None
            )

        return info

    def plot_input_and_generated(
        self,
        X_input,
        X_gen,
        features_to_plot: Optional[List[str]] = None,
    ):
        # X_input and X_gen should be a DataFrame
        if not isinstance(X_input, pd.DataFrame):
            raise ValueError("X_input must be a pandas DataFrame")
        if not isinstance(X_gen, pd.DataFrame):
            raise ValueError("X_gen must be a pandas DataFrame")

        # Sort both DataFrames by 'x' column
        if "x" in X_input.columns:
            X_input = X_input.sort_values(by="x")
        else:
            raise ValueError("'x' column not found in X_input")
        if "x" in X_gen.columns:
            X_gen = X_gen.sort_values(by="x")
        else:
            raise ValueError("'x' column not found in X_gen")

        # Handle default and validate features_to_plot
        if features_to_plot is None:
            features_to_plot = ["LTT"]  # default fallback
            self.logger.info('No features_to_plot provided. Defaulting to ["LTT"].')

        if not isinstance(features_to_plot, list) or not all(
            isinstance(col, str) for col in features_to_plot
        ):
            raise ValueError("features_to_plot must be a non-empty list of strings.")

        if not features_to_plot:
            self.logger.warning(
                "features_to_plot list is empty. No plots will be generated."
            )
            return

        self.logger.info("Plotting inference data...")

        for col in features_to_plot:
            if col not in X_input.columns or col not in X_gen.columns:
                self.logger.warning(f"Feature '{col}' not found in DataFrame")
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(
                X_input["x"],
                X_input[col],
                label="Original",
                alpha=0.8,
                s=10,
                color="tab:orange",
                marker=MarkerStyle("o"),
            )
            ax.scatter(
                X_gen["x"],
                X_gen[col],
                label="Generated",
                alpha=0.4,
                s=10,
                color="tab:blue",
                marker=MarkerStyle("x"),
            )
            ax.set_xlabel("x [mm]")
            ax.set_ylabel(rf"{col} [keV $\mu$m$^{{-1}}$]")
            ax.set_title(f"Generated {col} (upsampling x{self.config.upsample_factor})")
            ax.legend()
            ax.grid(True)
            display_plot(fig)

            save_figure(fig, self.config.output_dir, f"generated_{col}")

    def compute_wasserstein(self):
        """
        Compute the Wasserstein Distance (Earth Mover's Distance) for
        each feature.

        Args:
            real_df (pd.DataFrame): Real dataset.
            generated_df (pd.DataFrame): Generated dataset.

        Returns:
            dict: Wasserstein Distance per feature.
        """
        real_df = self.X_input_gaps
        generated_df = self.X_gen

        if real_df is None:
            self.logger.warning(
                "No ground truth data available. "
                "Skipping Wasserstein Distance computation."
            )
            return None

        wasserstein_scores = {}
        for feature in real_df.columns:
            if feature not in generated_df.columns:
                continue  # Skip features that don't exist in
            # generated data

            # Subsample the larger dataset to match the smaller one
            min_size = min(len(real_df), len(generated_df))
            real_sample = real_df[feature].sample(n=min_size, random_state=42).values
            gen_sample = (
                generated_df[feature].sample(n=min_size, random_state=42).values
            )

            wasserstein_scores[feature] = wasserstein_distance(real_sample, gen_sample)

        return wasserstein_scores

    def compute_ks_test(self):
        """
        Compute the Kolmogorov-Smirnov (KS) test for each feature.

        Args:
            real_df (pd.DataFrame): Real dataset.
            generated_df (pd.DataFrame): Generated dataset.

        Returns:
            dict: KS-test p-values per feature.
        """
        real_df = self.X_input_gaps
        generated_df = self.X_gen

        if real_df is None:
            self.logger.warning(
                "No ground truth data available. Skipping KS-test computation."
            )
            return None

        ks_test_results = {}
        for feature in real_df.columns:
            if feature not in generated_df.columns:
                continue  # Skip missing features
            _, pvalue = ks_2samp(real_df[feature], generated_df[feature])
            ks_test_results[feature] = pvalue

        return ks_test_results

    def compute_mean_variance_difference(self):
        real_df = self.X_input_gaps
        generated_df = self.X_gen

        if real_df is None:
            self.logger.warning(
                "No ground truth data available. "
                "Skipping mean/variance difference computation."
            )
            return None

        mean_var_diff = {}
        for feature in real_df.columns:
            mean_real = real_df[feature].mean()
            var_real = real_df[feature].var()
            mean_gen = generated_df[feature].mean()
            var_gen = generated_df[feature].var()
            if pd.isna(var_real):
                var_real = 0.0
            else:
                var_real = cast(float, var_real)
            if pd.isna(var_gen):
                var_gen = 0.0
            else:
                var_gen = cast(float, var_gen)

            mean_var_diff[feature] = {
                "Mean Difference": abs(mean_real - mean_gen),
                "Variance Difference": abs(var_real - var_gen),
            }
        return mean_var_diff

    def compute_reconstruction_metrics(self):
        """
        Compute pointwise spatial fidelity metrics for reconstruction
        quality.

        Compares the reconstructed data (X_rec) against the original
        input (X_input) to evaluate how well the VAE can reconstruct
        the low-resolution data.

        Returns:
            pd.DataFrame or None: DataFrame with reconstruction metrics
                per feature, or None if reconstruction data is not
                available.
        """
        if self.X_rec is None or self.X_input is None:
            self.logger.warning(
                "Reconstruction data not available. Skipping reconstruction metrics."
            )
            return None

        # Remove spatial coordinate 'x' if present
        features_to_compare = [col for col in self.X_input.columns if col != "x"]

        if not features_to_compare:
            self.logger.warning("No features to compare for reconstruction.")
            return None

        # Extract only the features (exclude 'x' coordinate)
        y_true = self.X_input[features_to_compare]
        y_pred = self.X_rec[features_to_compare]

        # Initialize the PointwiseMetrics calculator
        metrics_calc = PointwiseMetrics(epsilon=1e-8)

        # Compute all metrics per feature
        reconstruction_metrics = metrics_calc.compute_all(
            y_true=y_true,
            y_pred=y_pred,
            per_feature=True,
            feature_names=features_to_compare,
        )

        return reconstruction_metrics

    def compute_interpolation_metrics(self):
        """
        Compute pointwise spatial fidelity metrics for interpolated
        (gap-filling) data quality.

        Compares the generated interpolated data (X_gen) against the
        ground truth gaps (X_input_gaps) to evaluate how well the VAE
        can generate intermediate points.

        Returns:
            pd.DataFrame or None: DataFrame with interpolation metrics
                per feature, or None if ground truth data is not
                available.
        """
        if self.X_input_gaps is None:
            self.logger.warning(
                "No ground truth data available for interpolation. "
                "Skipping interpolation metrics."
            )
            return None

        # Remove spatial coordinate 'x' if present
        features_to_compare = [col for col in self.X_input_gaps.columns if col != "x"]

        if not features_to_compare:
            self.logger.warning("No features to compare for interpolation.")
            return None

        # Extract only the features (exclude 'x' coordinate)
        y_true = self.X_input_gaps[features_to_compare]
        y_pred = self.X_gen[features_to_compare]

        # Handle potential size mismatch due to interpolation rounding
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            self.logger.warning(
                f"Size mismatch in interpolation data: "
                f"ground_truth={len(y_true)}, "
                f"generated={len(y_pred)}. "
                f"Truncating both to {min_len} samples for comparison."
            )
            y_true = y_true.iloc[:min_len]
            y_pred = y_pred.iloc[:min_len]

        # Initialize the PointwiseMetrics calculator
        metrics_calc = PointwiseMetrics(epsilon=1e-8)

        # Compute all metrics per feature
        interpolation_metrics = metrics_calc.compute_all(
            y_true=y_true,
            y_pred=y_pred,
            per_feature=True,
            feature_names=features_to_compare,
        )

        return interpolation_metrics

    def compute_bragg_peak_metrics(self):
        """
        Compute Bragg peak-specific metrics for hadrontherapy
        evaluation.

        Evaluates domain-specific physical accuracy of Bragg peak
        reconstruction:
        - Peak Position Error: Accuracy of dose maximum depth (mm)
        - Peak Height Error: Accuracy of maximum dose value (a.u.)
        - FWHM Error: Accuracy of peak width (mm)
        - Distal Falloff Error: Accuracy of dose decrease beyond peak
          (dimensionless)

        Note:
            Only analyzes features with LET distributions that exhibit
            Bragg peaks (features starting with 'LTT', 'LDT', or
            'proton_'). While all features relate to secondary particles
            and dose distribution, only these show characteristic Bragg
            peak patterns. Other features are excluded from this
            analysis.

            Units: Spatial metrics (position, FWHM) in mm; height in
            feature units (a.u.); falloff is dimensionless (0-1).

        Returns:
            pd.DataFrame or None: DataFrame with Bragg peak metrics
                per feature, or None if ground truth data is not
                available.
        """
        if self.X_input_gaps is None:
            self.logger.warning(
                "No ground truth data available for Bragg peak "
                "analysis. Skipping Bragg peak metrics."
            )
            return None

        # Bragg peak metrics require spatial coordinate
        if "x" not in self.X_gen.columns or "x" not in self.X_input_gaps.columns:
            self.logger.warning(
                "Spatial coordinate 'x' not found. Cannot compute Bragg peak metrics."
            )
            return None

        # Get features to analyze (exclude 'x' coordinate)
        # Only analyze features with LET distributions showing Bragg peaks
        # (LTT, LDT, proton_*). While all features relate to secondary
        # particles, only these exhibit characteristic Bragg peak patterns.
        all_features = [col for col in self.X_input_gaps.columns if col != "x"]

        features_to_compare = [
            col
            for col in all_features
            if col.startswith("LTT")
            or col.startswith("LDT")
            or col.startswith("proton_")
        ]

        if not features_to_compare:
            self.logger.warning(
                "No features with Bragg peak LET distributions found "
                "(LTT, LDT, proton_*). Skipping Bragg peak analysis."
            )
            return None

        self.logger.info(
            f"Computing Bragg peak metrics for {len(features_to_compare)} "
            f"features with LET distributions: {features_to_compare}"
        )

        # Handle potential size mismatch
        y_gen = self.X_gen.copy()
        y_true = self.X_input_gaps.copy()

        if len(y_true) != len(y_gen):
            min_len = min(len(y_true), len(y_gen))
            self.logger.warning(
                f"Size mismatch in Bragg peak data: "
                f"ground_truth={len(y_true)}, "
                f"generated={len(y_gen)}. "
                f"Truncating both to {min_len} samples for comparison."
            )
            y_true = y_true.iloc[:min_len]
            y_gen = y_gen.iloc[:min_len]

        # Initialize BraggPeakMetrics calculator
        bragg_calc = BraggPeakMetrics(
            spatial_coordinate="x",
            falloff_range_percent=0.8,
            min_peak_prominence=0.1,
        )

        # Compute all Bragg peak metrics
        bragg_metrics = bragg_calc.compute_all(
            data_true=y_true,
            data_pred=y_gen,
            features=features_to_compare,
        )

        return bragg_metrics

    def evaluate_generated(self):
        """
        Comprehensive evaluation of VAE generation performance.

        Computes four categories of metrics:
        1. Reconstruction Metrics: How well the VAE reconstructs the
           original low-resolution input
        2. Interpolation Metrics: How well the VAE generates
           intermediate (gap-filling) points
        3. Bragg Peak Metrics: Domain-specific physical accuracy
           metrics for hadrontherapy
        4. Distribution Metrics: Statistical similarity between
           generated and ground truth distributions

        Returns:
            dict: Dictionary containing:
                - 'resolution_info': Dict with sampling/resolution
                  metadata
                - 'reconstruction': DataFrame with reconstruction
                  metrics (MAE, RMSE, MAPE, R²)
                - 'interpolation': DataFrame with interpolation metrics
                  (MAE, RMSE, MAPE, R²)
                - 'bragg_peak': DataFrame with Bragg peak metrics
                  (Peak Position Error, Peak Height Error, FWHM Error,
                  Distal Falloff Error)
                - 'distribution': DataFrame with distribution-based
                  metrics (Wasserstein, KS-test)
        """
        results = {}

        # Add resolution/sampling information
        results["resolution_info"] = self.get_resolution_info()

        # 1. Reconstruction Metrics (always computed if X_rec available)
        self.logger.info("Computing reconstruction metrics...")
        reconstruction_metrics = self.compute_reconstruction_metrics()
        if reconstruction_metrics is not None:
            results["reconstruction"] = reconstruction_metrics
            self.logger.info(
                f"Reconstruction metrics computed for "
                f"{len(reconstruction_metrics)} features"
            )
        else:
            self.logger.warning("Reconstruction metrics not available")

        # 2. Interpolation Metrics (only if ground truth gaps exist)
        self.logger.info("Computing interpolation metrics...")
        interpolation_metrics = self.compute_interpolation_metrics()
        if interpolation_metrics is not None:
            results["interpolation"] = interpolation_metrics
            self.logger.info(
                f"Interpolation metrics computed for "
                f"{len(interpolation_metrics)} features"
            )
        else:
            self.logger.info(
                "Interpolation metrics not available (no ground truth in direct mode)"
            )

        # 3. Bragg Peak Metrics (only if ground truth gaps exist)
        self.logger.info("Computing Bragg peak metrics...")
        bragg_peak_metrics = self.compute_bragg_peak_metrics()
        if bragg_peak_metrics is not None:
            results["bragg_peak"] = bragg_peak_metrics
            self.logger.info(
                f"Bragg peak metrics computed for {len(bragg_peak_metrics)} features"
            )
        else:
            self.logger.info(
                "Bragg peak metrics not available (no ground truth in direct mode)"
            )

        # 4. Distribution Metrics (only if ground truth gaps exist)
        if self.X_input_gaps is not None:
            self.logger.info("Computing distribution metrics...")
            wd = self.compute_wasserstein()
            ks_p = self.compute_ks_test()

            # Build distribution metrics DataFrame
            metric_rows = [
                {
                    "Feature": feature,
                    "Wasserstein Distance": wd.get(feature) if wd else None,
                    "KS-Test p-value": ks_p.get(feature) if ks_p else None,
                }
                for feature in self.X_input_gaps.columns
                if feature != "x"  # Exclude spatial coordinate
            ]

            results["distribution"] = pd.DataFrame(metric_rows)
            self.logger.info(
                f"Distribution metrics computed for {len(metric_rows)} features"
            )
        else:
            self.logger.info(
                "Distribution metrics not available (no ground truth in direct mode)"
            )

        # Return results dictionary
        if not results:
            self.logger.warning("No metrics were computed. Returning None.")
            return None

        return results


def main(
    config_path: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
    lowres_data_file: Optional[Union[str, Path]] = None,
    upsample_factor: Optional[int] = None,
):
    # Load from CLI or fallback default
    if config_path is None:
        parser = argparse.ArgumentParser(description="VAE Generation Script")
        parser.add_argument(
            "--config_path",
            type=str,
            default=str(
                Path(__file__).resolve().parent / "configs/generation_config.json"
            ),
            help="Path to generation config JSON file",
        )
        parser.add_argument(
            "--profile",
            type=str,
            default=None,
            help=(
                "Profile name to use (overrides profile in config file). "
                "Available: 'downsample', 'direct'"
            ),
        )
        parser.add_argument(
            "--lowres_data_file",
            type=str,
            default=None,
            help="Filename of low-resolution data file (overrides value in config file)",
        )
        parser.add_argument(
            "--upsample_factor",
            type=int,
            default=None,
            help="Upsampling factor (overrides config value)",
        )
        args = parser.parse_args()
        config_path = args.config_path
        profile = args.profile
        lowres_data_file = args.lowres_data_file
        upsample_factor = args.upsample_factor

    # Initialize configuration
    if config_path is None:
        raise ValueError("config_path cannot be None. Please provide a valid path.")
    try:
        config = load_config_with_profile(
            config_path=Path(config_path),
            config_class=GenerationConfig,
            profile_override=profile,
        )

        # Override lowres_data_file if provided via command line
        if lowres_data_file is not None:
            config.lowres_data_file = lowres_data_file

        # Override upsample_factor if provided via command line
        if upsample_factor is not None:
            config.upsample_factor = upsample_factor
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load config: {e}")

    # Initialize logger for preprocessor
    logger_obj = VAELogger(name="VAEGenerate", log_level="debug")

    # Start total execution timer
    with OptionalTimer(
        enabled=config.enable_timing,
        logger=logger_obj,
        description="Total script execution",
    ):
        # Preprocess data
        with OptionalTimer(
            enabled=config.enable_timing,
            logger=logger_obj,
            description="Data preprocessing",
        ):
            preprocessor = VAEDataPreprocessor(config, logger_obj, mode="inference")
            data_components = preprocessor.load_and_preprocess_data()

        # Initialise the generator
        with OptionalTimer(
            enabled=config.enable_timing,
            logger=logger_obj,
            description="Model initialization",
        ):
            generator = VAEGenerate(config=config, data_components=data_components)

        logger = generator.logger

        with OptionalTimer(
            enabled=config.enable_timing,
            logger=logger,
            description="Data loader preparation",
        ):
            generator.prepare_data_loaders()

        with OptionalTimer(
            enabled=config.enable_timing,
            logger=logger,
            description="High-resolution generation",
        ):
            upsampled_data = generator.generate_high_res()

        logger.info(f"upsampled_data.shape: {upsampled_data.shape}")

        # Separate the generated high-resolution data into reconstructed data
        # (low-resolution input passed through the autoencoder) and the
        # remaining interpolated data points (gap fill).
        with OptionalTimer(
            enabled=config.enable_timing,
            logger=logger,
            description="Data post-processing",
        ):
            reconstructed_df = upsampled_data.iloc[:: config.upsample_factor]
            gap_filler_data_df = upsampled_data.drop(reconstructed_df.index)
            logger.info(f"reconstructed_df.shape: {reconstructed_df.shape}")
            logger.info(f"gap_filler_data_df.shape: {gap_filler_data_df.shape}")

            # Reverse scaling of high-resolution and reconstructed data output
            reconstructed_df = generator.inverse_transform(reconstructed_df)
            gap_filler_data_df = generator.inverse_transform(gap_filler_data_df)

            # Export the complete upsampled data to textual file in Let.out
            # format
            complete_upsampled_df = generator.inverse_transform(upsampled_data)

        with OptionalTimer(
            enabled=config.enable_timing,
            logger=logger,
            description="Export to Let.out format",
        ):
            generator.export_to_let_file(complete_upsampled_df)

        # Initialize analyzer if plotting or analysis is enabled
        gen_analyzer = None
        if config.enable_plots or config.enable_analysis:
            with OptionalTimer(
                enabled=config.enable_timing,
                logger=logger,
                description="Analysis initialization",
            ):
                # Initialize the generation analyzer
                gen_analyzer = VAEGeneratorAnalysis(
                    config=config,
                    logger=logger,
                    X_gen=gap_filler_data_df,
                    X_rec=reconstructed_df,
                    X_input=data_components["X_input"],
                    X_input_gaps=data_components["R_input"],
                    voxel_size_um=data_components.get("voxel_in_um"),
                )

        # Generate plots if enabled
        if config.enable_plots and gen_analyzer is not None:
            # Plot comparison
            # Use ground truth if available (downsample mode), otherwise use
            # the original input (direct mode)
            with OptionalTimer(
                enabled=config.enable_timing,
                logger=logger,
                description="Plotting comparison",
            ):
                if data_components["R_input"] is not None:
                    gen_analyzer.plot_input_and_generated(
                        X_input=data_components["R_input"],
                        X_gen=gap_filler_data_df,
                        features_to_plot=config.features_to_plot,
                    )
                else:
                    logger.info(
                        "Generating plots using original input data "
                        "(no ground truth in direct mode)"
                    )
                    gen_analyzer.plot_input_and_generated(
                        X_input=data_components["X_input"],
                        X_gen=gap_filler_data_df,
                        features_to_plot=config.features_to_plot,
                    )

        # Run metrics analysis if enabled
        if config.enable_analysis and gen_analyzer is not None:
            # Evaluate generated data (returns dict with multiple
            # metric categories)
            with OptionalTimer(
                enabled=config.enable_timing,
                logger=logger,
                description="Metric evaluation",
            ):
                gen_metrics = gen_analyzer.evaluate_generated()
                if gen_metrics is not None:
                    logger.info("\n" + "=" * 60)
                    logger.info("GENERATION PERFORMANCE METRICS")
                    logger.info("=" * 60)

                    # Display resolution information
                    if "resolution_info" in gen_metrics:
                        res_info = gen_metrics["resolution_info"]
                        logger.info("\n[Resolution & Sampling Information]")
                        logger.info("-" * 60)
                        logger.info(f"  Input Mode: {res_info['input_mode']}")
                        logger.info(
                            f"  Upsample Factor: {res_info['upsample_factor']}x"
                        )
                        if res_info["downsample_factor"] is not None:
                            logger.info(
                                f"  Downsample Factor: {res_info['downsample_factor']}x"
                            )
                        if res_info["voxel_size_um"] is not None:
                            logger.info(
                                f"  Original Voxel Size: "
                                f"{res_info['voxel_size_um']:.3f} μm"
                            )
                            if "lowres_voxel_size_um" in res_info:
                                logger.info(
                                    f"  Low-Res Voxel Size: "
                                    f"{res_info['lowres_voxel_size_um']:.3f} μm"
                                )
                            if res_info.get("target_voxel_size_um"):
                                logger.info(
                                    f"  Target Voxel Size: "
                                    f"{res_info['target_voxel_size_um']:.3f} μm"
                                )
                        logger.info(
                            f"  Samples - Input: {res_info['n_input']}, "
                            f"Generated: {res_info['n_generated']}"
                        )
                        if "n_ground_truth" in res_info:
                            logger.info(
                                f"  Ground Truth Samples: {res_info['n_ground_truth']}"
                            )
                        logger.info("")

                    # Display reconstruction metrics
                    if "reconstruction" in gen_metrics:
                        logger.info("\n[1] Reconstruction Metrics")
                        logger.info("    (Low-res input → VAE → Reconstructed output)")
                        logger.info("-" * 60)
                        logger.info("\n" + gen_metrics["reconstruction"].to_string())
                        logger.info("")

                    # Display interpolation metrics
                    if "interpolation" in gen_metrics:
                        logger.info("\n[2] Interpolation Metrics")
                        logger.info("    (Generated gaps vs. ground truth)")
                        logger.info("-" * 60)
                        logger.info("\n" + gen_metrics["interpolation"].to_string())
                        logger.info("")

                    # Display Bragg peak metrics
                    if "bragg_peak" in gen_metrics:
                        logger.info("\n[3] Bragg Peak Metrics")
                        logger.info("    (Domain-specific hadrontherapy physics)")
                        logger.info(
                            "    Units: Position/FWHM in mm; Height in a.u.; Falloff dimensionless"
                        )
                        logger.info("-" * 60)
                        logger.info("\n" + gen_metrics["bragg_peak"].to_string())
                        logger.info("")

                    # Display distribution metrics
                    if "distribution" in gen_metrics:
                        logger.info("\n[4] Distribution Metrics")
                        logger.info("    (Statistical similarity tests)")
                        logger.info("-" * 60)
                        logger.info("\n" + gen_metrics["distribution"].to_string())
                        logger.info("")

                    logger.info("=" * 60 + "\n")
                else:
                    logger.info("Evaluation skipped: No metrics computed")

        # Log status messages for disabled features
        if not config.enable_plots and not config.enable_analysis:
            logger.info(
                "Plotting and analysis disabled "
                "(enable_plots=False, enable_analysis=False)"
            )
        elif not config.enable_plots:
            logger.info("Plotting disabled (enable_plots=False)")
        elif not config.enable_analysis:
            logger.info("Metrics analysis disabled (enable_analysis=False)")


if __name__ == "__main__":
    import sys

    if "get_ipython" in globals():
        # Running in interactive mode (VSCode Interactive Window or
        # Jupyter)
        main("configs/generation_config.json")
    else:
        # Standard command-line execution
        main()
