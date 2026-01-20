"""
Hierarchical Neural Network Classifier for multi-level classification.

This classifier implements hierarchical classification where:
- Level 0: Classifies into L0 categories
- Level 1: For each L0 category, a separate classifier predicts L1 categories
- Level N: Extends to arbitrary depth

Example hierarchy:
    Animal -> Mammal -> Dog
    Animal -> Bird -> Eagle
    Plant -> Flower -> Rose
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sklearn.model_selection import train_test_split

from nlp_templates.base.base_classifier import BaseClassifier
from nlp_templates.models.neural_network import NeuralNetworkClassifier
from nlp_templates.preprocessing.config_loader import ConfigLoader
from nlp_templates.evaluation.metrics import (
    calculate_metrics,
    get_confusion_matrix,
)
from nlp_templates.utils.logging_utils import get_logger


class HierarchicalNNClassifier(BaseClassifier):
    """
    Hierarchical Neural Network Classifier for multi-level hierarchical prediction.

    Supports arbitrary hierarchy depth with configurable neural network at each level.
    """

    def __init__(
        self,
        name: str = "hierarchical_nn_classifier",
        config_path: Optional[str] = None,
        random_state: int = 42,
        test_size: float = 0.3,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
    ):
        """
        Initialize Hierarchical NN Classifier.

        Args:
            name (str): Classifier name
            config_path (str, optional): Path to config file
            random_state (int): Random seed
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

        # Hierarchy structure
        self.level_classifiers: Dict[int, Dict[str, Any]] = {}
        self.hierarchy_levels = 2  # Default 2 levels
        self.label_hierarchy = None  # Stores hierarchy structure

        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train_hierarchy = None  # Shape (n_samples, n_levels)
        self.y_test_hierarchy = None
        self.class_names_by_level = {}

        # Metrics
        self.train_metrics = {}
        self.test_metrics = {}

    def load_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.DataFrame,
    ) -> None:
        """
        Load hierarchical data.

        Args:
            X (pd.DataFrame or np.ndarray): Features (n_samples, n_features)
            y (np.ndarray or pd.DataFrame): Hierarchical labels
                                           - np.ndarray: (n_samples, n_levels)
                                           - pd.DataFrame: Multiple columns for each level
        """
        # Convert y to numpy array if DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.values

        # Ensure y is 2D (n_samples, n_levels)
        if y.ndim == 1:
            raise ValueError(
                "Hierarchical labels must be 2D array with shape (n_samples, n_levels). "
                "Use columns for each hierarchy level."
            )

        self.hierarchy_levels = y.shape[1]
        self.y_train_hierarchy = y

        # Store class names for each level
        for level in range(self.hierarchy_levels):
            self.class_names_by_level[level] = np.unique(y[:, level])

        self.logger.info(
            f"Data loaded: X shape={X.shape}, "
            f"y shape={y.shape} (hierarchy levels={self.hierarchy_levels})"
        )
        self.logger.info(
            f"Classes per level: "
            f"{', '.join(str(len(c)) for c in self.class_names_by_level.values())}"
        )

        # Split data - use same indices for all levels
        (
            self.X_train,
            self.X_test,
            self.y_train_hierarchy,
            self.y_test_hierarchy,
        ) = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y[:, 0],  # Stratify by first level
        )

        self.logger.info(
            f"Data split: "
            f"train={self.X_train.shape}, test={self.X_test.shape}"
        )

    def build_model(self) -> None:
        """Build hierarchical classifiers for each level."""
        # Get NN config from config or use defaults
        nn_config = self.config.get("model", {}).get("params", {})
        if not nn_config:
            nn_config = {
                "hidden_dims": [128, 64],
                "activation": "relu",
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32,
            }

        # Build level 0 classifier (root classifier)
        self.logger.info(
            f"Building level 0 classifier for {len(self.class_names_by_level[0])} classes"
        )
        self.level_classifiers[0] = {
            "classifier": NeuralNetworkClassifier(
                random_state=self.random_state, **nn_config
            ),
            "classes": self.class_names_by_level[0],
        }

        # Build level classifiers (one per parent class)
        for level in range(1, self.hierarchy_levels):
            self.level_classifiers[level] = {}

            # For each class in the previous level, create a classifier
            for parent_class in self.class_names_by_level[level - 1]:
                # Find samples with this parent class
                mask = self.y_train_hierarchy[:, level - 1] == parent_class
                child_classes = np.unique(self.y_train_hierarchy[mask, level])

                self.logger.info(
                    f"Building level {level} classifier for parent "
                    f"'{parent_class}' -> {len(child_classes)} child classes"
                )

                self.level_classifiers[level][parent_class] = {
                    "classifier": NeuralNetworkClassifier(
                        random_state=self.random_state, **nn_config
                    ),
                    "classes": child_classes,
                }

        self.logger.info(
            f"Hierarchical model built with {self.hierarchy_levels} levels"
        )

    def train(self) -> None:
        """Train hierarchical classifiers."""
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Train level 0 classifier
        self.logger.info("Training level 0 classifier...")
        level_0_clf = self.level_classifiers[0]["classifier"]
        level_0_clf.fit(self.X_train, self.y_train_hierarchy[:, 0])
        self.logger.info("Level 0 classifier trained.")

        # Train level classifiers
        for level in range(1, self.hierarchy_levels):
            self.logger.info(f"Training level {level} classifiers...")

            for parent_class in self.class_names_by_level[level - 1]:
                # Find training samples with this parent class
                mask = self.y_train_hierarchy[:, level - 1] == parent_class

                if not mask.any():
                    self.logger.warning(
                        f"No training samples for parent class '{parent_class}' "
                        f"at level {level - 1}"
                    )
                    continue

                X_parent = self.X_train[mask]
                y_parent = self.y_train_hierarchy[mask, level]

                # Train classifier
                clf = self.level_classifiers[level][parent_class]["classifier"]
                clf.fit(X_parent, y_parent)

            self.logger.info(f"Level {level} classifiers trained.")

        self.logger.info("All hierarchical classifiers trained.")

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict hierarchical labels.

        Args:
            X (np.ndarray or pd.DataFrame): Features

        Returns:
            np.ndarray: Hierarchical predictions (n_samples, n_levels)
        """
        if not self.level_classifiers:
            raise ValueError("Model not trained. Call train() first.")

        n_samples = X.shape[0]
        # Use the same dtype as training labels to avoid type mismatch in metrics
        label_dtype = self.y_train_hierarchy.dtype
        predictions = np.empty((n_samples, self.hierarchy_levels), dtype=label_dtype)

        # Level 0 predictions
        level_0_clf = self.level_classifiers[0]["classifier"]
        predictions[:, 0] = level_0_clf.predict(X)

        # Level predictions
        for level in range(1, self.hierarchy_levels):
            for i in range(n_samples):
                parent_class = predictions[i, level - 1]

                # Get classifier for this parent
                if parent_class in self.level_classifiers[level]:
                    clf = self.level_classifiers[level][parent_class][
                        "classifier"
                    ]
                    predictions[i, level] = clf.predict(X[i : i + 1])[0]
                else:
                    # Fallback: use first class if parent not in training
                    self.logger.warning(
                        f"Unknown parent class '{parent_class}' at level {level - 1}"
                    )
                    available_classes = self.level_classifiers[level].keys()
                    predictions[i, level] = list(available_classes)[0]

        return predictions

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> List[np.ndarray]:
        """
        Get prediction probabilities for each level.

        Args:
            X (np.ndarray or pd.DataFrame): Features

        Returns:
            list: Probabilities for each level [(n_samples, n_classes_l0), ...]
        """
        if not self.level_classifiers:
            raise ValueError("Model not trained. Call train() first.")

        probabilities = []

        # Level 0 probabilities
        level_0_clf = self.level_classifiers[0]["classifier"]
        probabilities.append(level_0_clf.predict_proba(X))

        # Level probabilities (per parent class)
        for level in range(1, self.hierarchy_levels):
            n_samples = X.shape[0]
            level_preds = self.predict(X)  # Get predictions up to current level

            # For each parent class, get probabilities
            parent_classes = np.unique(level_preds[:, level - 1])

            # Get max number of child classes across all parents at this level
            max_child_classes = max(
                len(self.level_classifiers[level][pc]["classes"])
                for pc in self.level_classifiers[level]
            )

            level_proba = np.zeros((n_samples, max_child_classes))

            for parent_class in parent_classes:
                if parent_class in self.level_classifiers[level]:
                    mask = level_preds[:, level - 1] == parent_class
                    if mask.any():
                        clf = self.level_classifiers[level][parent_class][
                            "classifier"
                        ]
                        proba = clf.predict_proba(X[mask])
                        # Assign probabilities to the first n columns
                        level_proba[mask, : proba.shape[1]] = proba

            probabilities.append(level_proba)

        return probabilities

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate hierarchical classifier.

        Returns:
            dict: Evaluation metrics for each level
        """
        if not self.level_classifiers:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions
        y_train_pred = self.predict(self.X_train)
        y_test_pred = self.predict(self.X_test)

        results = {
            "train_metrics": {},
            "test_metrics": {},
            "confusion_matrices": {},
        }

        # Evaluate each level
        for level in range(self.hierarchy_levels):
            self.logger.info(f"Evaluating level {level}...")

            train_metrics = calculate_metrics(
                self.y_train_hierarchy[:, level], y_train_pred[:, level]
            )
            test_metrics = calculate_metrics(
                self.y_test_hierarchy[:, level], y_test_pred[:, level]
            )

            results["train_metrics"][f"level_{level}"] = train_metrics
            results["test_metrics"][f"level_{level}"] = test_metrics

            cm = get_confusion_matrix(
                self.y_test_hierarchy[:, level], y_test_pred[:, level]
            )
            results["confusion_matrices"][f"level_{level}"] = cm.tolist()

            self.logger.info(f"Level {level} train metrics: {train_metrics}")
            self.logger.info(f"Level {level} test metrics: {test_metrics}")

        self.train_metrics = results["train_metrics"]
        self.test_metrics = results["test_metrics"]

        return results

    def full_pipeline(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.DataFrame,
        save_visualizations: bool = False,
        output_dir: str = "outputs",
        log_to_mlflow: bool = False,
        mlflow_run_name: str = "hierarchical_run",
    ) -> Dict[str, Any]:
        """
        Execute full training and evaluation pipeline.

        Args:
            X: Features
            y: Hierarchical labels
            save_visualizations (bool): Whether to save visualizations
            output_dir (str): Output directory
            log_to_mlflow (bool): Whether to log to MLflow
            mlflow_run_name (str): MLflow run name

        Returns:
            dict: Pipeline results
        """
        # Load data
        self.load_data(X, y)

        # Build model
        self.build_model()

        # Train
        self.train()

        # Evaluate
        results = self.evaluate()

        # Log to MLflow if configured
        if log_to_mlflow and self.mlflow_manager:
            self.setup_mlflow()
            self._log_to_mlflow_hierarchical(mlflow_run_name)

        return results

    def _log_to_mlflow_hierarchical(
        self, run_name: str = "hierarchical_run"
    ) -> None:
        """Log hierarchical model to MLflow."""
        if not self.mlflow_manager:
            return

        run_id = self.mlflow_manager.start_run(run_name=run_name)

        try:
            # Log parameters
            self.mlflow_manager.log_params(
                {
                    "model_type": "hierarchical_nn",
                    "hierarchy_levels": self.hierarchy_levels,
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                }
            )

            # Log metrics for each level
            for level in range(self.hierarchy_levels):
                metrics_to_log = {}
                for key, value in self.train_metrics.get(
                    f"level_{level}", {}
                ).items():
                    metrics_to_log[f"train_level_{level}_{key}"] = value
                for key, value in self.test_metrics.get(
                    f"level_{level}", {}
                ).items():
                    metrics_to_log[f"test_level_{level}_{key}"] = value

                self.mlflow_manager.log_metrics(metrics_to_log)

            self.logger.info("Hierarchical model logged to MLflow.")

        finally:
            self.mlflow_manager.end_run()
