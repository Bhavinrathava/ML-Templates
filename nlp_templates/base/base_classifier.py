from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """
    Abstract base class for all classification templates.

    Responsibilities:
    1. Define the interface (abstract methods) that subclasses MUST implement
    2. Provide common functionality shared across all classifiers
    3. Handle cross-cutting concerns (MLflow, saving/loading, config)
    """

    def __init__(
        self,
        name: str,
        random_state: int = 42,
        test_size: float = 0.3,
        mlflow_tracking_uri: str = None,
        mlflow_experiment_name: str = None,
        config_path: str = None,
    ):
        """
        Initialize the classifier.

        Args:
            name (str): Name of the classifier
            random_state (int): Random state for reproducibility
            test_size (float): Fraction of data to use for testing
            mlflow_tracking_uri (str): MLflow tracking URI
            mlflow_experiment_name (str): MLflow experiment name
            config_path (str): Path to config file
        """
        self.name = name
        self.random_state = random_state
        self.test_size = test_size
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.config_path = config_path
        self.config = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.model_type = None
        self.levels = None
        self.level_encoders = None
        self.hierarchy_map = None
        self.feature_names = None
        self.metadata = None

    @abstractmethod
    def build_model(self):
        """
        Build the model.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame): Data to make predictions on

        Returns:
            pd.DataFrame: Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame): Data to make predictions on

        Returns:
            pd.DataFrame: Predictions
        """
        pass
