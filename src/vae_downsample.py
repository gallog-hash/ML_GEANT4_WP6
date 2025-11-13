import argparse
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from configs.task_config import GenerationConfig
from core.preprocessing.data_preprocessor import VAEDataPreprocessor
from utils import (
    VAELogger,
    ensure_directory_exists,
    load_config_with_profile,
    resolve_path_with_project_root,
)


class VAEDownsampler:
    """
    Downsamples high-resolution LET data and exports to Let.out format.

    This class reuses the downsampling logic from VAEDataPreprocessor
    and the export functionality similar to VAEGenerate, creating a
    standalone tool for data downsampling.
    """

    def __init__(self, config: GenerationConfig, logger: VAELogger):
        self.config = config
        self.logger = logger.get_logger()

    def downsample_data(self, data: pd.DataFrame, factor: int) -> pd.DataFrame:
        """
        Downsample high-density data by selecting every nth element.

        Args:
            data (pd.DataFrame): The input high-density data.
            factor (int): The downsampling factor (e.g., 1000 to go
                from micrometer to millimeter sampling).

        Returns:
            pd.DataFrame: Downsampled data (every nth element).
        """
        downsampled = data[::factor]
        self.logger.info(
            f"Downsampled from {len(data)} to {len(downsampled)} "
            f"samples (factor={factor})"
        )
        return downsampled

    def export_to_let_file(
        self,
        data_df: pd.DataFrame,
        voxel_size_um: float,
        source_data_dir: str,
        downsample_factor: int,
        output_filename: str = "Let_downsampled.out",
    ):
        """
        Export the downsampled LET data to a textual file in the same
        format as the input Let.out file.

        Args:
            data_df (pd.DataFrame): DataFrame containing the
                downsampled data to export.
            voxel_size_um (float): Voxel size in micrometers.
            source_data_dir (str): Source data directory name for
                traceability.
            downsample_factor (int): Downsampling factor used.
            output_filename (str): Name of the output file
                (default: "Let_downsampled.out").

        Returns:
            Path: The path to the exported file.
        """
        # Create a copy to avoid modifying the original DataFrame
        export_df = data_df.copy()

        # Recreate the original i, j, k columns that were dropped
        # during import
        if "x" in export_df.columns:
            # Convert voxel size from um to mm
            voxel_size_mm = voxel_size_um * 1e-3

            # Convert x (mm) back to i (index)
            # Original conversion: x = voxel_size_mm * i
            # Therefore: i = x / voxel_size_mm
            export_df["i"] = (export_df["x"] / voxel_size_mm).round().astype(int)

            # Add j and k columns (always 0 for 1D simulation)
            export_df["j"] = 0
            export_df["k"] = 0

            # Remove the 'x' column since original format uses 'i'
            export_df = export_df.drop(columns=["x"])

            # Reorder columns to match original format: i, j, k,
            # then other columns
            other_columns = [
                col for col in export_df.columns if col not in ["i", "j", "k"]
            ]
            column_order = ["i", "j", "k"] + other_columns
            export_df = export_df[column_order]

        # Create descriptive output directory name that includes
        # source data info and downsampling factor
        # Extract just the directory name from the full path
        source_dir_name = Path(source_data_dir).name

        # Build output directory name:
        # vae_downsample_output/<source_dir>_ds<factor>x
        output_dir_name = (
            f"vae_downsample_output/{source_dir_name}_ds{downsample_factor}x"
        )

        project_root = getattr(self.config, "project_root", None)
        output_dir = resolve_path_with_project_root(output_dir_name, project_root)
        output_path = output_dir / output_filename

        # Ensure the output directory exists
        ensure_directory_exists(output_dir)

        # Write to file with tab-separated values and no index
        export_df.to_csv(output_path, sep="\t", index=False, float_format="%.5g")

        self.logger.info(f"Downsampled LET data exported to: {output_path}")
        return output_path


def main(
    config_path: Optional[Union[str, Path]] = None,
    downsample_factor: Optional[int] = None,
    output_filename: Optional[str] = None,
):
    """
    Main function for downsampling high-resolution LET data.

    Args:
        config_path: Path to the configuration JSON file.
        downsample_factor: Override downsampling factor from config.
        output_filename: Override output filename.
    """
    # Parse command-line arguments
    if config_path is None:
        parser = argparse.ArgumentParser(description="VAE Data Downsampling Script")
        parser.add_argument(
            "--config_path",
            type=str,
            default=str(
                Path(__file__).resolve().parent / "configs/generation_config.json"
            ),
            help="Path to generation config JSON file",
        )
        parser.add_argument(
            "--downsample_factor",
            type=int,
            default=None,
            help="Downsampling factor (overrides config value)",
        )
        parser.add_argument(
            "--output_filename",
            type=str,
            default="Let_downsampled.out",
            help="Output filename for downsampled data",
        )
        args = parser.parse_args()
        config_path = args.config_path
        downsample_factor = args.downsample_factor
        output_filename = args.output_filename

    # Load configuration
    if config_path is None:
        raise ValueError("config_path cannot be None. Please provide a valid path.")

    try:
        config = load_config_with_profile(
            config_path=Path(config_path),
            config_class=GenerationConfig,
            profile_override=None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

    # Override downsample_factor if provided
    if downsample_factor is not None:
        config.downsample_factor = downsample_factor

    # Ensure downsample_factor is set
    if config.downsample_factor is None:
        raise ValueError(
            "downsample_factor must be specified either in config "
            "or via --downsample_factor argument"
        )

    # Initialize logger
    logger_obj = VAELogger(name="VAEDownsampler", log_level="debug")
    logger = logger_obj.get_logger()

    logger.info("=" * 80)
    logger.info("VAE Data Downsampling Script")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Downsample factor: {config.downsample_factor}")
    logger.info(f"Output filename: {output_filename}")
    logger.info("=" * 80)

    # Load and preprocess data
    # Force the preprocessor to load full high-resolution data
    # by temporarily setting mode to training
    logger.info("Loading high-resolution data...")
    preprocessor = VAEDataPreprocessor(config, logger_obj, mode="training")

    # We only need to load the raw data, not split it
    # So we'll directly use the preprocessing utilities
    from core.preprocessing.preprocessing_utils import (
        change_default_settings,
        import_and_clean_data,
        reorder_identity_features,
    )

    # Set random seed
    change_default_settings(config.random_seed)

    # Load and clean data
    data_dir = resolve_path_with_project_root(
        config.data_dir, getattr(config, "project_root", None)
    )
    df, cut_in_um, voxel_in_um = import_and_clean_data(
        data_dir=str(data_dir),
        data_file=config.data_file,
        primary_particle=config.primary_particle,
        let_type=config.let_type,
        cut_with_primary=config.cut_with_primary,
        drop_zero_cols=config.drop_zero_cols,
        drop_zero_thr=config.drop_zero_thr,
        verbose=0,
    )

    # Reorder identity features
    df = reorder_identity_features(df, config.identity_features)

    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Voxel size: {voxel_in_um} um")
    logger.info(f"Cut value: {cut_in_um} um")

    # Initialize downsampler
    downsampler = VAEDownsampler(config=config, logger=logger_obj)

    # Downsample the data
    downsampled_df = downsampler.downsample_data(
        data=df, factor=config.downsample_factor
    )

    # Export to Let.out format
    output_path = downsampler.export_to_let_file(
        data_df=downsampled_df,
        voxel_size_um=voxel_in_um,
        source_data_dir=config.data_dir,
        downsample_factor=config.downsample_factor,
        output_filename=output_filename,
    )

    logger.info("=" * 80)
    logger.info("Downsampling completed successfully!")
    logger.info(f"Output file: {output_path}")
    logger.info(
        f"Original size: {len(df)} samples, "
        f"Downsampled size: {len(downsampled_df)} samples"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys

    if "get_ipython" in globals():
        # Running in interactive mode (VSCode Interactive Window or
        # Jupyter)
        main("configs/generation_config.json")
    else:
        # Standard command-line execution
        main()
