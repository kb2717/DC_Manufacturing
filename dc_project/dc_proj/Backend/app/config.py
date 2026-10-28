from pathlib import Path

import yaml
from dotenv import load_dotenv

# Backend/ — config.yaml and the relative paths inside it resolve from here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)

_config = None


def load_config(config_path=None):
    global _config
    if _config is not None and config_path is None:
        return _config

    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            "Copy config.yaml.example to config.yaml (inside Backend/, "
            "next to main.py) and fill in your values."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if config_path is None:
        _config = cfg
    return cfg


def resolve_path(relative_path):
    return PROJECT_ROOT / relative_path