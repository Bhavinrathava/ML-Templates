"""
Abstract base class for all classification templates.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseClassifier(ABC):
    """
    Abstract base class for all classification templates.

    Defines the minimal interface that all classifiers must implement.
    """

    def __init__(
        self,
        name: str,
        random_state: int = 42,
        test_size: float = 0.3,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            name (str): Name of the classifier
            random_state (int): Random state for reproducibility
            test_size (float): Fraction of data to use for testing
            mlflow_tracking_uri (str, optional): MLflow tracking URI
            mlflow_experiment_name (str, optional): MLflow experiment name
        """
        self.name = name
        self.random_state = random_state
        self.test_size = test_size
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name

        # Data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Model
        self.model = None

    @abstractmethod
    def build_model(self):
        """Build the model."""
        pass

    @abstractmethod
    def train(self):
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame):
        """
        Make probability predictions on new data.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            Prediction probabilities
        """
        pass
