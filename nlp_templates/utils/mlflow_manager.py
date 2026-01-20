"""
MLflow Manager for model tracking, logging, and registry operations.

This module provides utilities to interact with MLflow for experiment tracking,
model logging, metrics recording, and model registry management.
"""

import os
import pickle
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.entities import ViewType


class MLflowManager:
    """
    Manages MLflow operations including model logging, metrics tracking, and registry.

    This class simplifies interaction with MLflow for experiment tracking,
    model versioning, and loading models from the registry.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "default",
        registry_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow Manager.

        Args:
            tracking_uri (str): URI for MLflow tracking server
                               (e.g., "http://localhost:5000" or local path)
            experiment_name (str): Name of the experiment to use. Defaults to "default"
            registry_uri (str, optional): URI for MLflow model registry.
                                         Defaults to same as tracking_uri
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        self.experiment_name = experiment_name

        # Set tracking URIs
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri != self.tracking_uri:
            mlflow.set_registry_uri(self.registry_uri)

        # Set or create experiment
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Create or set the experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            raise RuntimeError(f"Failed to setup experiment: {e}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name (str, optional): Name for the run
            tags (dict, optional): Dictionary of tags to log

        Returns:
            str: The run ID
        """
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id

        if tags:
            mlflow.set_tags(tags)

        return run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params (dict): Dictionary of parameters to log
        """
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            raise RuntimeError(f"Failed to log parameters: {e}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number for the metrics
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            raise RuntimeError(f"Failed to log metrics: {e}")

    def log_model(
        self,
        model: Any,
        model_name: str,
        framework: str = "sklearn",
        artifacts_path: Optional[str] = None,
    ) -> str:
        """
        Log a model to MLflow.

        Args:
            model: The model object to log
            model_name (str): Name to register the model with
            framework (str): Framework type ("sklearn", "pickle", etc.). Defaults to "sklearn"
            artifacts_path (str, optional): Path to additional artifacts to log

        Returns:
            str: Model URI
        """
        try:
            if framework == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif framework == "pickle":
                mlflow.log_artifact(
                    self._save_model_pickle(model), artifact_path="model"
                )
            else:
                # For custom models, use pickle
                mlflow.log_artifact(
                    self._save_model_pickle(model), artifact_path="model"
                )

            if artifacts_path and os.path.exists(artifacts_path):
                mlflow.log_artifacts(artifacts_path, artifact_path="artifacts")

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            return model_uri
        except Exception as e:
            raise RuntimeError(f"Failed to log model: {e}")

    def register_model(
        self,
        model_uri: str,
        registered_model_name: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Register a model in the MLflow model registry.

        Args:
            model_uri (str): URI of the model (e.g., "runs:/abc123/model")
            registered_model_name (str): Name to register the model as
            description (str, optional): Model description

        Returns:
            str: Registered model version
        """
        try:
            model_version = mlflow.register_model(
                model_uri, registered_model_name
            )

            if description:
                client = mlflow.tracking.MlflowClient()
                client.update_registered_model(
                    name=registered_model_name, description=description
                )

            return model_version
        except Exception as e:
            raise RuntimeError(f"Failed to register model: {e}")

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLflow.

        Args:
            model_uri (str): URI of the model to load
                           (e.g., "runs:/abc123/model" or "models:/my_model/production")

        Returns:
            Any: The loaded model object
        """
        try:
            model = mlflow.sklearn.load_model(model_uri)
            return model
        except Exception:
            # Try loading as generic artifact
            try:
                model = mlflow.pyfunc.load_model(model_uri)
                return model
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from {model_uri}: {e}"
                )

    def load_model_from_registry(
        self, model_name: str, stage: str = "Production"
    ) -> Any:
        """
        Load a model from the MLflow model registry by name and stage.

        Args:
            model_name (str): Name of the registered model
            stage (str): Stage of the model ("Production", "Staging", "Archived", "None")
                        Defaults to "Production"

        Returns:
            Any: The loaded model object
        """
        try:
            model_uri = f"models:/{model_name}/{stage}"
            return self.load_model(model_uri)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}' from stage '{stage}': {e}"
            )

    def get_best_run(
        self, metric_name: str, order: str = "DESC"
    ) -> Dict[str, Any]:
        """
        Get the best run from the current experiment based on a metric.

        Args:
            metric_name (str): Name of the metric to use for ranking
            order (str): Order to rank ("DESC" or "ASC"). Defaults to "DESC"

        Returns:
            dict: Dictionary containing run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            order_clause = f"metrics.{metric_name} {order}"

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[order_clause],
                max_results=1,
            )

            if len(runs) == 0:
                raise ValueError("No runs found in experiment")

            return runs.iloc[0].to_dict()
        except Exception as e:
            raise RuntimeError(f"Failed to get best run: {e}")

    def list_registered_models(self) -> List[str]:
        """
        List all registered models in the registry.

        Returns:
            list: List of registered model names
        """
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.search_registered_models()
            return [model.name for model in models]
        except Exception as e:
            raise RuntimeError(f"Failed to list registered models: {e}")

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a registered model.

        Args:
            model_name (str): Name of the registered model

        Returns:
            list: List of model version information
        """
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "created_timestamp": v.creation_timestamp,
                }
                for v in versions
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to get model versions: {e}")

    def transition_model_stage(
        self, model_name: str, version: str, stage: str
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            model_name (str): Name of the registered model
            version (str): Version number of the model
            stage (str): Target stage ("Staging", "Production", "Archived")
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to transition model '{model_name}' v{version} to '{stage}': {e}"
            )

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log an artifact (file) to MLflow.

        Args:
            local_path (str): Path to the file to log
            artifact_path (str, optional): Path within MLflow artifacts directory
        """
        try:
            if os.path.isfile(local_path):
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
            elif os.path.isdir(local_path):
                mlflow.log_artifacts(local_path, artifact_path=artifact_path)
            else:
                raise FileNotFoundError(f"Path not found: {local_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to log artifact: {e}")

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """
        Log a dictionary as JSON artifact.

        Args:
            data (dict): Dictionary to log
            filename (str): Name of the JSON file to create
        """
        try:
            temp_file = f"/tmp/{filename}"
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            mlflow.log_artifact(temp_file)
        except Exception as e:
            raise RuntimeError(f"Failed to log dictionary: {e}")

    @staticmethod
    def _save_model_pickle(model: Any, filename: str = "model.pkl") -> str:
        """
        Save a model using pickle.

        Args:
            model: Model object to save
            filename (str): Filename for the pickle file

        Returns:
            str: Path to the saved pickle file
        """
        temp_path = f"/tmp/{filename}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            pickle.dump(model, f)
        return temp_path
