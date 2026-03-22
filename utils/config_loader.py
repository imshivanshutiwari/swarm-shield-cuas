import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str = "configs") -> Dict[str, Dict[str, Any]]:
    """Load all four YAML configuration files from the configs directory."""
    base = Path(config_dir)
    configs = {
        "swarm": load_yaml(base / "swarm_config.yaml"),
        "marl": load_yaml(base / "marl_config.yaml"),
        "snn": load_yaml(base / "snn_config.yaml"),
        "detection": load_yaml(base / "detection_config.yaml"),
    }
    return configs


def merge_configs(*config_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple config dicts, later dicts override earlier ones."""
    merged: Dict[str, Any] = {}
    for d in config_dicts:
        merged.update(d)
    return merged


def get_env_var(key: str, default: Any = None) -> Any:
    """Retrieve an environment variable with an optional default."""
    return os.environ.get(key, default)


if __name__ == "__main__":
    configs = load_all_configs()
    print("Loaded configs:", list(configs.keys()))
    print("ALL OK")
