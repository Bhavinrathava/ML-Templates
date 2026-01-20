"""
Text Classification module.

Provides end-to-end text classification: Text -> Embeddings -> Predictions.
Supports both flat and hierarchical classification with automatic label encoding.
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from nlp_templates.classifiers.simple_multiclass_classifier import (
    SimpleMulticlassClassifier,
)
from nlp_templates.classifiers.hierarchical_multiclass_classifier import (
    HierarchicalNNClassifier,
)
from nlp_templates.utils.logging_utils import get_logger


class TextClassifier:
    """
    End-to-end text classifier that handles:
    1. Text -> Embeddings conversion
    2. Automatic label encoding (supports text labels)
    3. Embeddings -> Label prediction
    4. Model persistence with MLflow integration

    Supports both flat multi-class and hierarchical classification.

    Example usage with text labels:
        >>> clf = TextClassifier(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        >>> clf.fit(
        ...     texts=["great product", "terrible service", "okay item"],
        ...     labels=["positive", "negative", "neutral"]
        ... )
        >>> predictions = clf.predict(["amazing quality"])
        >>> print(predictions)  # ["positive"]
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        classifier_type: str = "simple",
        name: str = "text_classifier",
        config_path: Optional[str] = None,
        random_state: int = 42,
        test_size: float = 0.2,
        batch_size: int = 32,
        device: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
    ):
        """
        Initialize TextClassifier.

        Args:
            embedding_model: HuggingFace model name for embeddings.
                           Default: "sentence-transformers/all-MiniLM-L6-v2"
            classifier_type: Type of classifier - "simple" or "hierarchical"
            name: Classifier name for logging
            config_path: Path to YAML/JSON config for classifier
            random_state: Random seed for reproducibility
            test_size: Test set fraction (0.0 to 1.0)
            batch_size: Batch size for embedding generation
            device: Device for embedding model ("cuda", "cpu", or None for auto)
            mlflow_tracking_uri: MLflow tracking URI for logging
            mlflow_experiment_name: MLflow experiment name
        """
        self.logger = get_logger(name)
        self.name = name
        self.embedding_model_name = embedding_model
        self.classifier_type = classifier_type
        self.config_path = config_path
        self.random_state = random_state
        self.test_size = test_size
        self.batch_size = batch_size
        self.device = device
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name

        # Will be initialized lazily
        self._embedder = None
        self._classifier = None

        # Label encoders for text labels
        self._label_encoders: Dict[int, LabelEncoder] = {}
        self._label_columns: Optional[List[str]] = None
        self._uses_text_labels = False

        # Store training data info
        self.embedding_dim = None
        self.is_fitted = False
        self.classes_: Optional[Dict[int, np.ndarray]] = None

        self.logger.info(
            f"TextClassifier initialized with embedding_model='{embedding_model}', "
            f"classifier_type='{classifier_type}'"
        )

    @property
    def embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            self._load_embedder()
        return self._embedder

    def _load_embedder(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
            self.embedding_dim = self._embedder.get_sentence_embedding_dimension()
            self.logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def _init_classifier(self, is_hierarchical: bool = False) -> None:
        """Initialize the underlying classifier."""
        if is_hierarchical or self.classifier_type == "hierarchical":
            self._classifier = HierarchicalNNClassifier(
                name=f"{self.name}_hierarchical",
                config_path=self.config_path,
                random_state=self.random_state,
                test_size=self.test_size,
                mlflow_tracking_uri=self.mlflow_tracking_uri,
                mlflow_experiment_name=self.mlflow_experiment_name,
            )
        else:
            self._classifier = SimpleMulticlassClassifier(
                name=f"{self.name}_simple",
                config_path=self.config_path,
                random_state=self.random_state,
                test_size=self.test_size,
                mlflow_tracking_uri=self.mlflow_tracking_uri,
                mlflow_experiment_name=self.mlflow_experiment_name,
            )

    def _encode_labels(
        self,
        labels: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    ) -> np.ndarray:
        """
        Encode text labels to integers.

        Args:
            labels: Labels to encode (can be text or numeric)

        Returns:
            np.ndarray: Encoded integer labels
        """
        # Convert to appropriate format
        if isinstance(labels, pd.DataFrame):
            self._label_columns = labels.columns.tolist()
            labels_array = labels.values
        elif isinstance(labels, pd.Series):
            labels_array = labels.values.reshape(-1, 1)
        elif isinstance(labels, list):
            labels_array = np.array(labels)
            if labels_array.ndim == 1:
                labels_array = labels_array.reshape(-1, 1)
        else:
            labels_array = labels
            if labels_array.ndim == 1:
                labels_array = labels_array.reshape(-1, 1)

        # Check if labels are text (need encoding)
        self._uses_text_labels = labels_array.dtype.kind in ('U', 'S', 'O')

        if not self._uses_text_labels:
            # Already numeric, just ensure proper shape
            if labels_array.shape[1] == 1:
                return labels_array.ravel()
            return labels_array

        # Encode each column
        encoded = np.zeros(labels_array.shape, dtype=int)
        self.classes_ = {}

        for col_idx in range(labels_array.shape[1]):
            encoder = LabelEncoder()
            encoded[:, col_idx] = encoder.fit_transform(labels_array[:, col_idx])
            self._label_encoders[col_idx] = encoder
            self.classes_[col_idx] = encoder.classes_

            self.logger.info(
                f"Label column {col_idx}: {len(encoder.classes_)} classes - "
                f"{list(encoder.classes_)}"
            )

        # Return 1D if single column
        if encoded.shape[1] == 1:
            return encoded.ravel()
        return encoded

    def _decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Decode integer labels back to original text labels.

        Args:
            encoded_labels: Integer encoded labels

        Returns:
            np.ndarray: Original text labels
        """
        if not self._uses_text_labels:
            return encoded_labels

        # Ensure 2D for processing
        if encoded_labels.ndim == 1:
            encoded_labels = encoded_labels.reshape(-1, 1)

        decoded = np.empty(encoded_labels.shape, dtype=object)

        for col_idx in range(encoded_labels.shape[1]):
            if col_idx in self._label_encoders:
                decoded[:, col_idx] = self._label_encoders[col_idx].inverse_transform(
                    encoded_labels[:, col_idx].astype(int)
                )
            else:
                decoded[:, col_idx] = encoded_labels[:, col_idx]

        # Return 1D if single column
        if decoded.shape[1] == 1:
            return decoded.ravel()
        return decoded

    def embed_texts(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Convert texts to embeddings.

        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar during embedding

        Returns:
            np.ndarray: Embeddings of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()

        # Handle empty strings
        texts = [str(t) if t else "" for t in texts]

        embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def fit(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
        labels: Union[np.ndarray, pd.Series, pd.DataFrame, List],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> "TextClassifier":
        """
        Fit the text classifier.

        Args:
            texts: Training texts
            labels: Training labels (can be text or numeric)
                   - For simple classification: 1D array/list of labels
                   - For hierarchical: 2D array or DataFrame with multiple columns
            model_config: Optional model configuration dict

        Returns:
            self: Fitted classifier

        Example:
            >>> clf.fit(
            ...     texts=["great!", "terrible", "okay"],
            ...     labels=["positive", "negative", "neutral"]
            ... )

            >>> # Hierarchical with DataFrame
            >>> clf.fit(
            ...     texts=products,
            ...     labels=pd.DataFrame({
            ...         "category": ["Electronics", "Clothing", ...],
            ...         "subcategory": ["Phones", "Shirts", ...]
            ...     })
            ... )
        """
        self.logger.info(f"Fitting TextClassifier on {len(texts)} samples...")

        # Generate embeddings
        self.logger.info("Generating embeddings...")
        X = self.embed_texts(texts, show_progress=True)
        self.logger.info(f"Embeddings generated. Shape: {X.shape}")

        # Encode labels (handles text labels automatically)
        self.logger.info("Encoding labels...")
        y = self._encode_labels(labels)
        self.logger.info(f"Labels encoded. Shape: {y.shape if hasattr(y, 'shape') else len(y)}")

        # Determine if hierarchical based on label shape
        is_hierarchical = isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1

        if is_hierarchical and self.classifier_type != "hierarchical":
            self.logger.info(
                "Detected hierarchical labels, switching to hierarchical classifier"
            )
            self.classifier_type = "hierarchical"

        # Initialize classifier
        self._init_classifier(is_hierarchical)

        # Apply model config if provided
        if model_config:
            self._classifier.config = {"model": model_config}
        elif not self._classifier.config:
            # Default neural network config
            self._classifier.config = {
                "model": {
                    "type": "neural_network",
                    "params": {
                        "hidden_dims": [256, 128],
                        "activation": "relu",
                        "dropout_rate": 0.3,
                        "learning_rate": 0.001,
                        "epochs": 50,
                        "batch_size": 32,
                    },
                }
            }

        # Train classifier
        self._classifier.load_data(X, y)
        self._classifier.build_model()
        self._classifier.train()

        self.is_fitted = True
        self.logger.info("TextClassifier fitted successfully.")

        return self

    def predict(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
        return_encoded: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Predict labels for texts.

        Args:
            texts: Texts to classify
            return_encoded: If True, return integer encoded labels instead of original
            show_progress: Show progress bar during embedding

        Returns:
            np.ndarray: Predicted labels (text labels if trained with text labels)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        # Generate embeddings
        X = self.embed_texts(texts, show_progress=show_progress)

        # Predict (returns encoded integers)
        encoded_predictions = self._classifier.predict(X)

        if return_encoded or not self._uses_text_labels:
            return encoded_predictions

        # Decode to original text labels
        return self._decode_labels(encoded_predictions)

    def predict_proba(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
        show_progress: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get prediction probabilities for texts.

        Args:
            texts: Texts to classify
            show_progress: Show progress bar during embedding

        Returns:
            np.ndarray or List[np.ndarray]: Prediction probabilities
                - Simple classifier: (n_samples, n_classes)
                - Hierarchical: List of arrays per level
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        # Generate embeddings
        X = self.embed_texts(texts, show_progress=show_progress)

        # Predict probabilities
        return self._classifier.predict_proba(X)

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the classifier on test set.

        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        return self._classifier.evaluate()

    def get_classes(self, level: int = 0) -> np.ndarray:
        """
        Get class labels for a given level.

        Args:
            level: Hierarchy level (0 for simple classification)

        Returns:
            np.ndarray: Class labels
        """
        if self.classes_ is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        if level not in self.classes_:
            raise ValueError(f"Level {level} not found. Available: {list(self.classes_.keys())}")

        return self.classes_[level]

    def save(self, path: str) -> str:
        """
        Save the classifier to disk.

        Saves:
        - Label encoders
        - Model configuration
        - Classifier state

        Note: The embedding model is not saved (it's loaded by name).

        Args:
            path: Directory path to save the model

        Returns:
            str: Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "name": self.name,
            "embedding_model_name": self.embedding_model_name,
            "classifier_type": self.classifier_type,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "batch_size": self.batch_size,
            "uses_text_labels": self._uses_text_labels,
            "label_columns": self._label_columns,
            "embedding_dim": self.embedding_dim,
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save label encoders
        if self._label_encoders:
            with open(save_dir / "label_encoders.pkl", "wb") as f:
                pickle.dump(self._label_encoders, f)

        # Save classes
        if self.classes_:
            classes_serializable = {
                str(k): v.tolist() for k, v in self.classes_.items()
            }
            with open(save_dir / "classes.json", "w") as f:
                json.dump(classes_serializable, f, indent=2)

        # Save classifier (without embedding model)
        with open(save_dir / "classifier.pkl", "wb") as f:
            pickle.dump(self._classifier, f)

        self.logger.info(f"Model saved to {save_dir}")
        return str(save_dir)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TextClassifier":
        """
        Load a classifier from disk.

        Args:
            path: Directory path where model was saved
            device: Device for embedding model

        Returns:
            TextClassifier: Loaded classifier
        """
        load_dir = Path(path)

        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            embedding_model=metadata["embedding_model_name"],
            classifier_type=metadata["classifier_type"],
            name=metadata["name"],
            random_state=metadata["random_state"],
            test_size=metadata["test_size"],
            batch_size=metadata["batch_size"],
            device=device,
        )

        # Load label encoders
        encoders_path = load_dir / "label_encoders.pkl"
        if encoders_path.exists():
            with open(encoders_path, "rb") as f:
                instance._label_encoders = pickle.load(f)

        # Load classes
        classes_path = load_dir / "classes.json"
        if classes_path.exists():
            with open(classes_path, "r") as f:
                classes_dict = json.load(f)
                instance.classes_ = {
                    int(k): np.array(v) for k, v in classes_dict.items()
                }

        # Load classifier
        with open(load_dir / "classifier.pkl", "rb") as f:
            instance._classifier = pickle.load(f)

        instance._uses_text_labels = metadata["uses_text_labels"]
        instance._label_columns = metadata.get("label_columns")
        instance.embedding_dim = metadata.get("embedding_dim")
        instance.is_fitted = True

        instance.logger.info(f"Model loaded from {load_dir}")
        return instance

    def log_to_mlflow(
        self,
        run_name: str = "text_classifier_run",
        log_model: bool = True,
        artifacts_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log the classifier to MLflow.

        Args:
            run_name: Name for the MLflow run
            log_model: Whether to log the model artifacts
            artifacts_dir: Optional directory with additional artifacts

        Returns:
            str: MLflow run ID or None if MLflow not configured
        """
        if not self.mlflow_tracking_uri:
            self.logger.warning("MLflow not configured. Set mlflow_tracking_uri.")
            return None

        try:
            from nlp_templates.utils.mlflow_manager import MLflowManager
        except ImportError:
            self.logger.warning("MLflow not installed.")
            return None

        mlflow_manager = MLflowManager(
            tracking_uri=self.mlflow_tracking_uri,
            experiment_name=self.mlflow_experiment_name or "text_classifier",
        )

        run_id = mlflow_manager.start_run(run_name=run_name)

        try:
            # Log parameters
            params = {
                "embedding_model": self.embedding_model_name,
                "classifier_type": self.classifier_type,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "uses_text_labels": self._uses_text_labels,
            }

            if self.classes_:
                for level, classes in self.classes_.items():
                    params[f"n_classes_level_{level}"] = len(classes)

            mlflow_manager.log_params(params)

            # Log metrics from evaluation
            if self.is_fitted:
                results = self._classifier.evaluate()
                metrics = {}

                if "test_metrics" in results:
                    if isinstance(results["test_metrics"], dict):
                        # Check if it's hierarchical (nested dict)
                        first_val = next(iter(results["test_metrics"].values()), None)
                        if isinstance(first_val, dict):
                            # Hierarchical metrics
                            for level_key, level_metrics in results["test_metrics"].items():
                                for metric_name, value in level_metrics.items():
                                    metrics[f"test_{level_key}_{metric_name}"] = value
                        else:
                            # Simple metrics
                            for k, v in results["test_metrics"].items():
                                metrics[f"test_{k}"] = v

                mlflow_manager.log_metrics(metrics)

            # Log label encoders as artifact
            if log_model and self._label_encoders:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".pkl", delete=False
                ) as f:
                    pickle.dump(self._label_encoders, f)
                    mlflow_manager.client.log_artifact(f.name, "label_encoders")

            # Log classes mapping
            if log_model and self.classes_:
                import tempfile
                classes_serializable = {
                    str(k): list(v) for k, v in self.classes_.items()
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(classes_serializable, f, indent=2)
                    mlflow_manager.client.log_artifact(f.name, "classes")

            self.logger.info(f"Logged to MLflow. Run ID: {run_id}")

        finally:
            mlflow_manager.end_run()

        return run_id

    def fit_predict(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
        labels: Union[np.ndarray, pd.Series, pd.DataFrame, List],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Fit classifier and return predictions on training data.

        Args:
            texts: Training texts
            labels: Training labels
            model_config: Optional model configuration

        Returns:
            np.ndarray: Predictions on training data
        """
        self.fit(texts, labels, model_config)
        return self.predict(texts)

    def get_embeddings(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Get embeddings for texts without classification.

        Args:
            texts: Texts to embed
            show_progress: Show progress bar

        Returns:
            np.ndarray: Text embeddings
        """
        return self.embed_texts(texts, show_progress)

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        label_type = "text" if self._uses_text_labels else "numeric"
        return (
            f"TextClassifier("
            f"embedding_model='{self.embedding_model_name}', "
            f"classifier_type='{self.classifier_type}', "
            f"label_type='{label_type}', "
            f"status='{status}')"
        )


class HierarchicalTextClassifier(TextClassifier):
    """
    Specialized text classifier for hierarchical labels with multiple fields.

    Supports text labels that are automatically encoded and decoded.

    Example:
        >>> clf = HierarchicalTextClassifier()
        >>> clf.fit(
        ...     texts=["iPhone 15 Pro", "Nike Running Shoes"],
        ...     labels=pd.DataFrame({
        ...         "category": ["Electronics", "Clothing"],
        ...         "subcategory": ["Phones", "Shoes"]
        ...     })
        ... )
        >>> predictions = clf.predict(["Samsung Galaxy"])
        >>> print(predictions)
        # [["Electronics", "Phones"]]
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        name: str = "hierarchical_text_classifier",
        config_path: Optional[str] = None,
        random_state: int = 42,
        test_size: float = 0.2,
        batch_size: int = 32,
        device: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
    ):
        """
        Initialize HierarchicalTextClassifier.

        Args:
            embedding_model: HuggingFace model name for embeddings
            name: Classifier name
            config_path: Path to config file
            random_state: Random seed
            test_size: Test set fraction
            batch_size: Batch size for embeddings
            device: Device for embedding model
            mlflow_tracking_uri: MLflow tracking URI
            mlflow_experiment_name: MLflow experiment name
        """
        super().__init__(
            embedding_model=embedding_model,
            classifier_type="hierarchical",
            name=name,
            config_path=config_path,
            random_state=random_state,
            test_size=test_size,
            batch_size=batch_size,
            device=device,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
        )

    def get_level_predictions(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
        level: int,
    ) -> np.ndarray:
        """
        Get predictions for a specific hierarchy level.

        Args:
            texts: Texts to classify
            level: Hierarchy level (0-indexed)

        Returns:
            np.ndarray: Predictions for the specified level (original labels)
        """
        predictions = self.predict(texts)
        if predictions.ndim == 1:
            if level != 0:
                raise ValueError("Single-level predictions, use level=0")
            return predictions

        if level >= predictions.shape[1]:
            raise ValueError(
                f"Level {level} out of range. Max level: {predictions.shape[1] - 1}"
            )
        return predictions[:, level]

    def get_label_mapping(self, level: int = 0) -> Dict[int, str]:
        """
        Get mapping from encoded integers to original labels for a level.

        Args:
            level: Hierarchy level

        Returns:
            dict: Mapping from int to label string
        """
        if level not in self._label_encoders:
            raise ValueError(f"Level {level} not found")

        encoder = self._label_encoders[level]
        return {i: label for i, label in enumerate(encoder.classes_)}

    def predict_as_dataframe(
        self,
        texts: Union[List[str], pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Predict and return results as a DataFrame with column names.

        Args:
            texts: Texts to classify

        Returns:
            pd.DataFrame: Predictions with original column names
        """
        predictions = self.predict(texts)

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        columns = self._label_columns or [f"level_{i}" for i in range(predictions.shape[1])]

        return pd.DataFrame(predictions, columns=columns)
