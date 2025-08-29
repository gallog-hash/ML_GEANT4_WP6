import json
from pathlib import Path
from typing import Type, TypeVar, Union

from dacite import Config as DaciteConfig
from dacite import DaciteError, from_dict

from utils.logger import VAELogger

T = TypeVar("T")

logger = VAELogger("config_loader", "info").get_logger()

def load_config_from_json(config_path: Union[str, Path], config_class: Type[T]) -> T:
    logger.info(f"Loading configuration from: {config_path}")

    config_path = Path(config_path)
    
    try:
        with open(config_path, "r") as f:
            raw_config = json.load(f)
        logger.debug(f"Raw config loaded: {raw_config}")

        config = from_dict(
            data_class=config_class, 
            data=raw_config,
            config=DaciteConfig(type_hooks={Path: Path})
        )
        logger.info(f"Configuration loaded into {config_class.__name__}")
        return config

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except DaciteError as e:
        logger.error(f"Failed to map config to {config_class.__name__}: {e}")
        raise
