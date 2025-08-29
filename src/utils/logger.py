# src/utils/logger.py
import logging
import sys


class VAELogger:
    """
    Centralized logging system for the VAE different pipelines.
    """
    def __init__(self, name: str = 'VAELogger', log_level: str = 'info'):
        self._logger = logging.getLogger(name)
        self._logger.propagate = False # Prevent propagation to root logger
        
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self._get_log_level(log_level))
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] %(message)s", 
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self.set_logger_level(log_level)
        self._prefix = ""
        
    def get_logger(self):
        return self._logger
        
    def set_logger_level(self, level: str = 'info'):
        log_level = self._get_log_level(level)
        self._logger.setLevel(log_level)
        
    def _get_log_level(self, level_str: str = 'info'):
        levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        try:
            return levels[level_str.lower()]
        except KeyError:
            raise ValueError("Invalid log level specified.")
        
    def set_logger_prefix(self, prefix):
        self._prefix = f"[{prefix}]"
        self._wrap_methods_with_prefix()
        
    def _wrap_methods_with_prefix(self):
        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            original = getattr(self._logger, level)
            setattr(self._logger, level, 
                    lambda msg, o=original: o(f"{self._prefix} {msg}".strip())
            )
            
def log_params_dict(category: str, params: dict, logger=None):
    """
    Logs a dictionary of parameters under a category header

    Args:
        category (str): The title or block label (e.g., 'Optimizer', 'Loss
        Settings').
        params (dict): Dictionary to log.
        logger (logging.Logger, optional): Logger instance. If None, uses
        default VAELogger. 
    """
    if logger is None:
        logger = VAELogger("hyperparam_logger").get_logger()

    logger.debug(f"--- {category} Parameters ---")
    for key, value in params.items():
        logger.debug(f"  • {key}: {value}")
