# src/configs/base_config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BaseVAEConfig:
    data_dir: Optional[str] = None  # Directory containing the data files
    data_file: Optional[str] = None  # Name of the data file

    study_name: str = "vae_optimization"
    project_root: Optional[str] = None  # For resolving relative paths
    random_seed = int = 42
    output_dir: Path = Path("../outputs")
    use_timestamp_output_dir: bool = False  # default behavior
    
    optuna_io_dir: Optional[Path] = None  # Default: {output_dir}/optuna_output
    database: Optional[str] = None  # Default: "optuna_study.db"
