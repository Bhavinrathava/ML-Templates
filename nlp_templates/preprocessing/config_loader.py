"""
Configuration loader for classifier training and evaluation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage classifier configurations from JSON or YAML files."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML file.

        Args:
            config_path (str): Path to config file (JSON or YAML)

        Returns:
            dict: Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if path.suffix.lower() in [".json"]:
            with open(path) as f:
                return json.load(f)
        elif path.suffix.lower() in [".yaml", ".yml"]:
            with open(path) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {path.suffix}. "
                "Use .json or .yaml"
            )

    @staticmethod
    def get_classifier_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get classifier-specific configuration."""
        return config.get("classifier", {})

    @staticmethod
    def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model hyperparameters configuration."""
        return config.get("model", {})

    @staticmethod
    def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get training configuration."""
        return config.get("training", {})

    @staticmethod
    def get_mlflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return config.get("mlflow", {})

    @staticmethod
    def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get data configuration."""
        return config.get("data", {})

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to file.

        Args:
            config (dict): Configuration to save
            output_path (str): Path to save config (JSON or YAML)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".json":
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
        elif path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(
                f"Unsupported format: {path.suffix}. Use .json or .yaml"
            )
