import json
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

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
            config=DaciteConfig(type_hooks={Path: Path}),
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


def load_config_with_profile(
    config_path: Union[str, Path],
    config_class: Type[T],
    profile_path: Optional[Union[str, Path]] = None,
    profile_override: Optional[str] = None,
) -> T:
    """
    Load configuration with profile-based overrides.

    This function allows for a base configuration to be extended with
    mode-specific settings defined in a separate profiles file.

    Args:
        config_path: Path to the base configuration JSON file
        config_class: Dataclass type to instantiate
        profile_path: Path to profiles JSON file. If None, defaults to
            '{config_dir}/generation_profiles.json'
        profile_override: Profile name to use instead of the one
            specified in config file

    Returns:
        Instance of config_class with profile settings merged

    Raises:
        FileNotFoundError: If config or profile file not found
        ValueError: If specified profile doesn't exist in profiles file
        json.JSONDecodeError: If JSON parsing fails
        DaciteError: If dataclass instantiation fails

    Example:
        >>> config = load_config_with_profile(
        ...     config_path="configs/generation_config.json",
        ...     config_class=GenerationConfig,
        ...     profile_override="direct"
        ... )
    """
    logger.info(f"Loading configuration with profile from: {config_path}")

    config_path = Path(config_path)

    try:
        # Load base configuration
        with open(config_path, "r") as f:
            base_config = json.load(f)
        logger.debug(f"Base config loaded: {base_config}")

        # Determine profile name (CLI override takes precedence)
        profile_name = profile_override or base_config.pop("profile", None)

        if profile_name:
            # Determine profile file path
            if profile_path is None:
                profile_path = config_path.parent / "generation_profiles.json"
            else:
                profile_path = Path(profile_path)

            logger.info(f"Loading profile '{profile_name}' from: {profile_path}")

            # Load profiles
            try:
                with open(profile_path, "r") as f:
                    profiles = json.load(f)
            except FileNotFoundError:
                logger.error(f"Profile file not found: {profile_path}")
                raise

            # Get selected profile
            if profile_name not in profiles:
                available = ", ".join(profiles.keys())
                raise ValueError(
                    f"Profile '{profile_name}' not found in "
                    f"{profile_path}. Available profiles: {available}"
                )

            profile_overrides = profiles[profile_name]
            logger.debug(f"Profile overrides: {profile_overrides}")

            # Merge: profile overrides base
            merged_config = {**base_config, **profile_overrides}
            logger.info(f"Configuration merged with profile '{profile_name}'")
        else:
            # No profile specified, use base config as-is
            merged_config = base_config
            logger.info("No profile specified, using base configuration")

        # Create dataclass instance
        config = from_dict(
            data_class=config_class,
            data=merged_config,
            config=DaciteConfig(type_hooks={Path: Path}),
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
