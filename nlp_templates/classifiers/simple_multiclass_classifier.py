"""
Simple multiclass classifier implementation.

Handles flat-label classification with configurable model, training,
evaluation, metrics tracking, and MLflow logging.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from nlp_templates.models.neural_network import NeuralNetworkClassifier

from nlp_templates.base.base_classifier import BaseClassifier
from nlp_templates.utils.mlflow_manager import MLflowManager
from nlp_templates.utils.logging_utils import get_logger
from nlp_templates.preprocessing.config_loader import ConfigLoader
from nlp_templates.evaluation.metrics import (
    calculate_metrics,
    get_confusion_matrix,
    get_classification_report,
)
from nlp_templates.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_metrics_comparison,
)


class SimpleMulticlassClassifier(BaseClassifier):
    """
    Simple multiclass classifier with flat label classification.

    Supports multiple model types, config-driven training,
    metrics calculation, and MLflow integration.
    """

    MODEL_REGISTRY = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "naive_bayes": MultinomialNB,
        "svm": SVC,
        "neural_network": NeuralNetworkClassifier,
    }

    def __init__(
        self,
        name: str = "simple_multiclass_classifier",
        config_path: Optional[str] = None,
        random_state: int = 42,
        test_size: float = 0.3,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
    ):
        """
        Initialize SimpleMulticlassClassifier.

        Args:
            name (str): Classifier name
            config_path (str, optional): Path to config file (JSON/YAML)
            random_state (int): Random state for reproducibility
            test_size (float): Test set fraction
            mlflow_tracking_uri (str, optional): MLflow tracking URI
            mlflow_experiment_name (str, optional): MLflow experiment name
        """
        super().__init__(
            name=name,
            random_state=random_state,
            test_size=test_size,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
        )

        self.logger = get_logger(name)
        self.config = {}
        self.config_path = config_path

        # Load config if provided
        if config_path:
            self.config = ConfigLoader.load_config(config_path)
            self.logger.info(f"Loaded config from {config_path}")

        # Model parameters
        self.model_type = None
        self.model_params = {}

        # Metrics and evaluation
        self.train_metrics = {}
        self.test_metrics = {}
        self.cm = None
        self.class_names = None

        # MLflow manager
        self.mlflow_manager = None

    def load_data(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Load and prepare data for training.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series or np.ndarray): Labels
            label_column (str, optional): Column name for labels (if X contains labels)
        """
        if (
            isinstance(X, pd.DataFrame)
            and label_column
            and label_column in X.columns
        ):
            y = X[label_column]
            X = X.drop(columns=[label_column])

        # Convert to numpy if needed
        if isinstance(y, pd.Series):
            y = y.values

        self.logger.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")

        # Store class names
        self.class_names = np.unique(y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.logger.info(
            f"Data split: "
            f"train={self.X_train.shape}, test={self.X_test.shape}"
        )

    def build_model(self) -> None:
        """
        Build the model with configuration parameters.

        Model type and hyperparameters are read from config or defaults.
        """
        # Get model configuration
        model_config = ConfigLoader.get_model_config(self.config)
        self.model_type = model_config.get("type", "logistic_regression")
        self.model_params = model_config.get("params", {})

        # Add random_state if supported (skip neural_network, it handles it separately)
        if self.model_type in [
            "logistic_regression",
            "random_forest",
            "svm",
        ]:
            self.model_params["random_state"] = self.random_state
        elif self.model_type == "neural_network":
            # Neural network uses random_state separately
            self.model_params["random_state"] = self.random_state

        # Get model class
        if self.model_type not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )

        ModelClass = self.MODEL_REGISTRY[self.model_type]
        self.model = ModelClass(**self.model_params)

        self.logger.info(
            f"Model built: type={self.model_type}, params={self.model_params}"
        )

    def train(self) -> None:
        """Train the model on training data."""
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.logger.info(f"Training {self.model_type} model...")
        self.model.fit(self.X_train, self.y_train)
        self.logger.info("Model training completed.")

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate model on both train and test sets.

        Returns:
            dict: Evaluation results with metrics and confusion matrix
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Metrics
        self.train_metrics = calculate_metrics(self.y_train, y_train_pred)
        self.test_metrics = calculate_metrics(self.y_test, y_test_pred)

        # Confusion matrix
        self.cm = get_confusion_matrix(self.y_test, y_test_pred)

        self.logger.info(f"Train metrics: {self.train_metrics}")
        self.logger.info(f"Test metrics: {self.test_metrics}")

        return {
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "confusion_matrix": self.cm.tolist(),
            "classification_report": get_classification_report(
                self.y_test, y_test_pred, output_dict=True
            ),
        }

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame or np.ndarray): Features

        Returns:
            np.ndarray: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X (pd.DataFrame or np.ndarray): Features

        Returns:
            np.ndarray: Prediction probabilities (if model supports it)
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(
                f"Model {self.model_type} does not support predict_proba"
            )

        return self.model.predict_proba(X)

    def save_visualizations(
        self, output_dir: str = "outputs"
    ) -> Dict[str, str]:
        """
        Save evaluation visualizations.

        Args:
            output_dir (str): Directory to save visualizations

        Returns:
            dict: Paths to saved visualizations
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        viz_paths = {}

        # Confusion matrix
        if self.cm is not None:
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                self.cm, class_names=self.class_names, output_path=cm_path
            )
            viz_paths["confusion_matrix"] = cm_path
            self.logger.info(f"Saved confusion matrix to {cm_path}")

        # Metrics comparison
        if self.train_metrics and self.test_metrics:
            metrics_path = os.path.join(output_dir, "metrics_comparison.png")
            plot_metrics_comparison(
                self.train_metrics,
                self.test_metrics,
                output_path=metrics_path,
            )
            viz_paths["metrics_comparison"] = metrics_path
            self.logger.info(f"Saved metrics comparison to {metrics_path}")

        return viz_paths

    def setup_mlflow(self) -> None:
        """Initialize MLflow manager if credentials provided."""
        if self.mlflow_tracking_uri:
            self.mlflow_manager = MLflowManager(
                tracking_uri=self.mlflow_tracking_uri,
                experiment_name=self.mlflow_experiment_name
                or "simple_multiclass",
            )
            self.logger.info("MLflow manager initialized.")

    def log_to_mlflow(
        self,
        run_name: str = "default_run",
        log_artifacts: bool = True,
        artifacts_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log model and metrics to MLflow.

        Args:
            run_name (str): Name for MLflow run
            log_artifacts (bool): Whether to log visualization artifacts
            artifacts_dir (str, optional): Directory with artifacts to log

        Returns:
            str: MLflow run ID or None if MLflow not configured
        """
        if not self.mlflow_manager:
            self.logger.warning("MLflow not configured. Setup MLflow first.")
            return None

        # Start run
        run_id = self.mlflow_manager.start_run(run_name=run_name)
        self.logger.info(f"MLflow run started: {run_id}")

        try:
            # Log parameters
            self.mlflow_manager.log_params(
                {
                    "model_type": self.model_type,
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    **self.model_params,
                }
            )

            # Log metrics
            metrics_to_log = {}
            for key, value in self.train_metrics.items():
                metrics_to_log[f"train_{key}"] = value
            for key, value in self.test_metrics.items():
                metrics_to_log[f"test_{key}"] = value

            self.mlflow_manager.log_metrics(metrics_to_log)

            # Tag model with key metrics
            metric_tags = {
                "train_accuracy": f"{self.train_metrics.get('accuracy', 0):.4f}",
                "test_accuracy": f"{self.test_metrics.get('accuracy', 0):.4f}",
                "train_f1": f"{self.train_metrics.get('f1', 0):.4f}",
                "test_f1": f"{self.test_metrics.get('f1', 0):.4f}",
                "train_precision": f"{self.train_metrics.get('precision', 0):.4f}",
                "test_precision": f"{self.test_metrics.get('precision', 0):.4f}",
                "train_recall": f"{self.train_metrics.get('recall', 0):.4f}",
                "test_recall": f"{self.test_metrics.get('recall', 0):.4f}",
            }
            self.mlflow_manager.set_tags(metric_tags)

            # Log model
            self.mlflow_manager.log_model(
                self.model, model_name=f"{self.name}_model", framework="sklearn"
            )

            # Log artifacts
            if log_artifacts and artifacts_dir:
                self.mlflow_manager.client.log_artifacts(artifacts_dir)
                self.logger.info(f"Logged artifacts from {artifacts_dir}")

            self.logger.info("Metrics and model logged to MLflow.")

        finally:
            self.mlflow_manager.end_run()

        return run_id

    def full_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        label_column: Optional[str] = None,
        save_visualizations: bool = True,
        output_dir: str = "outputs",
        log_to_mlflow: bool = True,
        mlflow_run_name: str = "default_run",
    ) -> Dict[str, Any]:
        """
        Run complete pipeline: load -> build -> train -> evaluate.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series or np.ndarray): Labels
            label_column (str, optional): Column name for labels in X
            save_visualizations (bool): Whether to save visualizations
            output_dir (str): Output directory for visualizations
            log_to_mlflow (bool): Whether to log to MLflow
            mlflow_run_name (str): MLflow run name

        Returns:
            dict: Results including metrics and paths
        """
        self.logger.info("Starting full pipeline...")

        # Load data
        self.load_data(X, y, label_column)

        # Build and train
        self.build_model()
        self.train()

        # Evaluate
        results = self.evaluate()

        # Save visualizations
        if save_visualizations:
            results["visualizations"] = self.save_visualizations(output_dir)

        # Log to MLflow
        if log_to_mlflow and self.mlflow_tracking_uri:
            self.setup_mlflow()
            results["mlflow_run_id"] = self.log_to_mlflow(
                run_name=mlflow_run_name,
                log_artifacts=save_visualizations,
                artifacts_dir=output_dir if save_visualizations else None,
            )

        self.logger.info("Pipeline completed successfully.")
        return results
