# modules/config.py
import os
import yaml
from src.utils.logging import logger

class ConfigError(Exception):
    pass

_config_cache = None

def load_config(config_file="settings.yml"):
    global _config_cache
    if _config_cache is None:
        if not os.path.exists(config_file):
            logger.error("Configuration file %s not found.", config_file)
            raise ConfigError(f"Configuration file {config_file} not found.")
        with open(config_file, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
        logger.info("Configuration loaded from %s", config_file)
    return _config_cache
