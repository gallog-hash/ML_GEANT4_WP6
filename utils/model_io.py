# src/utils/model_io.py
import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch import device as TorchDevice

from utils.logger import VAELogger

logger = VAELogger("model_io", "info").get_logger()

def load_model_config(config_path: Union[str, Path]) -> dict:
    config_path = Path(config_path)
    
    logger.info(f"Loading model configuration from: {config_path}")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
        logger.info("Model configuration loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model config file not found at: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise 

def load_model(
    weights_path: Union[str, Path], 
    device: Union[str, TorchDevice], 
    params: Optional[dict] = None,
    config_path: Optional[Union[str, Path]] = None, 
):
    # Delayed import to avoid circular dependency with model_builder
    from core.model_builder import build_vae_model_from_params
    
    weights_path = Path(weights_path)

    if params is None:
        if config_path is None:
            logger.error("Both 'params' and 'config_path' are None. "
                         "Unable to load model configuration.")
            raise ValueError("Either 'params' or 'config_path' must be provided.")
        
        try:
            params = load_model_config(config_path)
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            raise

    if params is None:
        raise ValueError(
            "Model parameters must be provided either via `params` or `config_path`."
        )

    for key in ["input_dim", "processed_dim"]:
        if key not in params:
            logger.error(f"Missing required parameter '{key}' in config")
            raise KeyError(f"Missing required parameter '{key}' in config.")
    
    logger.info(f"Loading model weights from: {weights_path}")
    try:
        model = build_vae_model_from_params(
            input_dim=params["input_dim"],
            n_processed_features=params["processed_dim"],
            hyperparams=params,
            device=device
        )
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        logger.info("Model loaded and moved to device successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"Model weights file not found at {weights_path}")
        raise 
    except RuntimeError as e:
        logger.error(f"Error loading model weights: {e}")
        raise
