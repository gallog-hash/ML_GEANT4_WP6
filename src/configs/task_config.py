# src/configs/task_config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from configs.base_config import BaseVAEConfig


@dataclass
class TrainingConfig(BaseVAEConfig):
    # Preprocessing configuration
    primary_particle: str = "proton"
    let_type: str = "track"
    cut_with_primary: bool = True
    drop_zero_cols: bool = True
    drop_zero_thr: float = 100.0
    # Training configuration
    train_size: float = 0.7
    val_size: float = 0.2
    test_size: float = 0.1
    scaler: Union[str, Dict[str, Any]] = field(default_factory=lambda: "minmax")
    n_trials: int = 400
    val_loss_threshold = 1e2
    identity_features: List[str] = field(default_factory=lambda: ['x'])
    # Output Directories and Filenames
    model_summary_filename: str = "model_summary.txt"
    model_weights_filename: str = "model_weights.pth"
    hparams_config_filename: str = "hyperparameters_config.json"
    history_filename: str = "model_training_history.json"
    
    features_to_plot: List[str] = field(default_factory=lambda: ['LTT'])
    metrics_to_plot: List[Union[str, List[str]]] = field(
        default_factory=lambda: [
            ["train_loss", "val_loss"],
            "train_loss_mse",
            "train_loss_kld",
            "train_neg_penalty",
        ]
    )
    
@dataclass
class TrainerConfig(TrainingConfig):
    net_size: List[int] = field(default_factory=lambda: [256, 128, 64])
    latent_dim: int = 24
    use_dropout: bool = False
    dropout_rate: float = 0.2
    use_exit_activation: bool = False
    exit_activation: Dict[str, Any] = field(
        default_factory=lambda: {"type": "shifted_softplus", 
                                 "params": {"beta_softplus": 1.0}}
    )
    skip_norm_in_final: bool = True
    loss: Dict[str, Any] = field(
        default_factory=lambda: {"type": "standard",
                                 "params": {
                                     "beta": 0.1,
                                     "use_negative_panalty": False,
                                     "neg_penalty_weight": 0.0,
                                     "val_loss_threshold": 1e2,
                                 }}
    )
    optimizer: Dict[str, Any] = field(
        default_factory=lambda: {"optimizer": "adam",
                                 "learning_rate": 1e-3,
                                 "weight_decay":1e-5}
    )
    batch_size: int = 128
    training_epochs: int = 100
    training_log_step: int = 10
    plot_data_splitting: bool = False
    
@dataclass
class OptunaStudyConfig(TrainingConfig):
    study_name: Optional[str] = None  # Default: "vae_optimization"
    study_path: Optional[str] = None  # Default: sqlite:///{output_dir}/optuna_study.db
    # Model default settings
    default_training_epochs: int = 50
    default_training_log_step: int = 5
    default_skip_norm_in_final: bool = True
    default_use_exit_activation: bool = True
    default_use_negative_penalty: bool = False
    default_use_dropout: bool = True
    # Flags to enforce defaults during optimization
    use_default_skip_norm_in_final: bool = False
    use_default_use_exit_activation: bool = False
    use_default_use_negative_penalty: bool = False
    use_default_use_dropout: bool = False
    
@dataclass
class PostTrainingConfig(TrainingConfig):
    # Full paths (to be set after resolving with project root)
    hparams_config_path: Union[str, Path, None] = None
    model_path: Union[str, Path, None] = None
    history_path: Union[str, Path, None] = None
    
@dataclass
class OptimizationAnalysisConfig(BaseVAEConfig):
    study_path: Optional[str] = None  # Default: sqlite:///{output_dir}/optuna_study.db
    features_to_plot: List[str] = field(default_factory=lambda: ["LTT"])
    config_path: Union[str, Path, None] = None
    model_path: Union[str, Path, None] = None

@dataclass
class GenerationConfig(PostTrainingConfig):
    resample_factor: int = 20
    plot_dir: Path = Path("../vae_generate_plots")
