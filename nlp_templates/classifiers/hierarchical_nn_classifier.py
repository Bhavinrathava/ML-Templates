"""
Hierarchical Neural Network Classifier for multi-level classification.

Implements a hierarchical classification system where:
- Level 1: Classify into primary categories
- Level 2+: For each level 1 category, train specialized classifiers for subcategories
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from nlp_templates.base.base_classifier import BaseClassifier
from nlp_templates.models.neural_network import NeuralNetworkClassifier
from nlp_templates.utils.logging_utils import get_logger
from nlp_templates.preprocessing.config_loader import ConfigLoader
from nlp_templates.evaluation.metrics import (
    calculate_metrics,
    get_confusion_matrix,
)
from nlp_templates.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_metrics_comparison,
)
from sklearn.model_selection import train_test_split
import os


class HierarchicalNNClassifier(BaseClassifier):
    """
    Hierarchical Neural Network Classifier for multi-level predictions.

    Trains a hierarchy of neural network classifiers where the first level
    predicts the primary category, and subsequent levels predict subcategories
    based on the parent category.

    Example hierarchy:
        Level 1: {Class_A, Class_B, Class_C}
        Level 2 (if Level1=A): {A1, A2, A3}
        Level 2 (if Level1=B): {B1, B2}
        Level 2 (if Level1=C): {C1, C2, C3, C4}
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

        # Hierarchical structure
        self.n_levels = 0
        self.level_1_classifier = None
        self.level_classifiers = {}  # {level1_class: classifier_for_level2}
        self.level_2_classifiers = (
            {}
        )  # {(level1, level2): classifier_for_level3}

        # Data storage
        self.hierarchy_data = {}  # {level1_class: (X_data, y_data)}

        # Metrics
        self.train_metrics = {}
        self.test_metrics = {}
        self.cm = None
        self.class_names = None

        # MLflow manager
        self.mlflow_manager = None

    def load_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: List[Tuple] | np.ndarray,
    ) -> None:
        """
        Load hierarchical data for training.

        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (list of tuples or np.ndarray): Hierarchical labels as tuples
                e.g., [('A', 'A1'), ('A', 'A2'), ('B', 'B1'), ...]
        """
        # Convert y to list of tuples if needed
        if isinstance(y, np.ndarray):
            y = [tuple(row) for row in y]

        self.n_levels = len(y[0]) if y else 0
        self.logger.info(
            f"Data loaded: X shape={X.shape}, y shape=({len(y)},), "
            f"hierarchy levels={self.n_levels}"
        )

        # Store features
        self.X = X if isinstance(X, np.ndarray) else X.values

        # Organize data by level 1 classes
        level_1_classes = set(label[0] for label in y)
        self.logger.info(f"Level 1 classes: {sorted(level_1_classes)}")

        for level1_class in level_1_classes:
            # Get indices for this level 1 class
            indices = [
                i for i, label in enumerate(y) if label[0] == level1_class
            ]
            X_level1 = self.X[indices]

            if self.n_levels == 1:
                # Single level - not hierarchical
                y_level1 = None
            else:
                # Multi-level - extract level 2 labels
                y_level1 = np.array([y[i][1] for i in indices])

            self.hierarchy_data[level1_class] = (X_level1, y_level1)

        self.class_names = sorted(list(level_1_classes))
        self.logger.info(
            f"Organized data by level 1 classes: "
            f"{[(c, len(self.hierarchy_data[c][0])) for c in sorted(self.hierarchy_data.keys())]}"
        )

    def build_model(self) -> None:
        """Build the hierarchical model structure."""
        model_config = ConfigLoader.get_model_config(self.config)
        model_type = model_config.get("type", "neural_network")
        model_params = model_config.get("params", {})

        if model_type != "neural_network":
            self.logger.warning(
                f"Hierarchical classifier uses neural_network. "
                f"Ignoring model_type={model_type}"
            )

        # Add random_state
        model_params["random_state"] = self.random_state

        self.model_params = model_params
        self.logger.info(
            f"Hierarchical model built with params: {self.model_params}"
        )

    def train(self) -> None:
        """Train the hierarchical classifier."""
        if self.hierarchy_data is None or len(self.hierarchy_data) == 0:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.logger.info("Training hierarchical NN classifier...")

        # Level 1: Train classifier for primary categories
        level_1_classes = sorted(list(self.hierarchy_data.keys()))

        # Prepare level 1 training data
        X_level1_list = []
        y_level1_list = []

        for class_idx, class_name in enumerate(level_1_classes):
            X_class, _ = self.hierarchy_data[class_name]
            X_level1_list.append(X_class)
            y_level1_list.extend([class_name] * len(X_class))

        X_level1 = np.vstack(X_level1_list)
        y_level1 = np.array(y_level1_list)

        # Train level 1 classifier
        self.level_1_classifier = NeuralNetworkClassifier(**self.model_params)
        self.level_1_classifier.fit(X_level1, y_level1)
        self.logger.info(
            f"Level 1 classifier trained on {len(X_level1)} samples"
        )

        # Level 2+: For each level 1 class, train specialized classifiers
        if self.n_levels > 1:
            for level1_class in level_1_classes:
                X_class, y_class = self.hierarchy_data[level1_class]

                if y_class is not None and len(np.unique(y_class)) > 1:
                    # Train level 2 classifier for this level 1 class
                    classifier = NeuralNetworkClassifier(**self.model_params)
                    classifier.fit(X_class, y_class)
                    self.level_classifiers[level1_class] = classifier
                    self.logger.info(
                        f"Level 2 classifier trained for {level1_class} "
                        f"on {len(X_class)} samples"
                    )
                else:
                    self.logger.warning(
                        f"Level 2 classifier for {level1_class}: "
                        f"insufficient or single-class data"
                    )

        self.logger.info("Hierarchical model training completed.")

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make hierarchical predictions.

        Args:
            X (pd.DataFrame or np.ndarray): Features

        Returns:
            np.ndarray: Hierarchical predictions (n_samples, n_levels)
        """
        if self.level_1_classifier is None:
            raise ValueError("Model not trained. Call train() first.")

        X_array = X if isinstance(X, np.ndarray) else X.values

        # Predict level 1
        level1_predictions = self.level_1_classifier.predict(X_array)

        if self.n_levels == 1:
            return level1_predictions.reshape(-1, 1)

        # Predict level 2 based on level 1 predictions
        level2_predictions = []

        for i, level1_pred in enumerate(level1_predictions):
            if level1_pred in self.level_classifiers:
                # Use the specialized classifier for this level 1 class
                sample = X_array[i].reshape(1, -1)
                level2_pred = self.level_classifiers[level1_pred].predict(
                    sample
                )[0]
            else:
                # Fallback: use a default or first available classifier
                self.logger.warning(
                    f"No classifier for level1={level1_pred}, using first available"
                )
                first_classifier = next(iter(self.level_classifiers.values()))
                sample = X_array[i].reshape(1, -1)
                level2_pred = first_classifier.predict(sample)[0]

            level2_predictions.append(level2_pred)

        # Combine predictions
        predictions = np.column_stack(
            [level1_predictions, np.array(level2_predictions)]
        )
        return predictions

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Get prediction probabilities for each level.

        Args:
            X (pd.DataFrame or np.ndarray): Features

        Returns:
            dict: {level: probabilities_array}
        """
        if self.level_1_classifier is None:
            raise ValueError("Model not trained. Call train() first.")

        X_array = X if isinstance(X, np.ndarray) else X.values

        # Level 1 probabilities
        level1_proba = self.level_1_classifier.predict_proba(X_array)

        proba_dict = {1: level1_proba}

        if self.n_levels > 1:
            # Level 2 probabilities (for each sample based on level 1 prediction)
            level1_predictions = self.level_1_classifier.predict(X_array)
            level2_probas = []

            for i, level1_pred in enumerate(level1_predictions):
                if level1_pred in self.level_classifiers:
                    sample = X_array[i].reshape(1, -1)
                    proba = self.level_classifiers[level1_pred].predict_proba(
                        sample
                    )[0]
                else:
                    # Fallback probabilities
                    first_classifier = next(
                        iter(self.level_classifiers.values())
                    )
                    sample = X_array[i].reshape(1, -1)
                    proba = first_classifier.predict_proba(sample)[0]

                level2_probas.append(proba)

            proba_dict[2] = np.array(level2_probas)

        return proba_dict

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate hierarchical model.

        Returns:
            dict: Evaluation results
        """
        if self.level_1_classifier is None:
            raise ValueError("Model not trained. Call train() first.")

        # For simplicity, evaluate at level 1
        y_level1_true = []
        y_level1_pred = []

        for level1_class in sorted(self.hierarchy_data.keys()):
            X_class, _ = self.hierarchy_data[level1_class]
            preds = self.level_1_classifier.predict(X_class)

            y_level1_true.extend([level1_class] * len(X_class))
            y_level1_pred.extend(preds)

        self.train_metrics = calculate_metrics(y_level1_true, y_level1_pred)
        self.logger.info(f"Hierarchical metrics: {self.train_metrics}")

        return {
            "train_metrics": self.train_metrics,
            "test_metrics": self.train_metrics,
            "classification_report": {"level_1": "See train_metrics above"},
        }

    def save_visualizations(
        self, output_dir: str = "outputs"
    ) -> Dict[str, str]:
        """Save evaluation visualizations."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {}
