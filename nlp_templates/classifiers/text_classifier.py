"""
Text Classification module.

Provides end-to-end text classification: Text -> Embeddings -> Predictions.
Supports both flat and hierarchical classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

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
    2. Embeddings -> Label prediction

    Supports both flat multi-class and hierarchical classification.

    Example usage:
        >>> clf = TextClassifier(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        >>> clf.fit(texts=["great product", "terrible service"], labels=[1, 0])
        >>> predictions = clf.predict(["amazing quality"])
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

        # Will be initialized lazily
        self._embedder = None
        self._classifier = None

        # Store training data info
        self.embedding_dim = None
        self.is_fitted = False

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
            )
        else:
            self._classifier = SimpleMulticlassClassifier(
                name=f"{self.name}_simple",
                config_path=self.config_path,
                random_state=self.random_state,
                test_size=self.test_size,
            )

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
            labels: Training labels
                   - For simple classification: 1D array of labels
                   - For hierarchical: 2D array (n_samples, n_levels)
            model_config: Optional model configuration dict

        Returns:
            self: Fitted classifier
        """
        self.logger.info(f"Fitting TextClassifier on {len(texts)} samples...")

        # Generate embeddings
        self.logger.info("Generating embeddings...")
        X = self.embed_texts(texts, show_progress=True)
        self.logger.info(f"Embeddings generated. Shape: {X.shape}")

        # Convert labels
        if isinstance(labels, (pd.Series, list)):
            y = np.array(labels)
        elif isinstance(labels, pd.DataFrame):
            y = labels.values
        else:
            y = labels

        # Determine if hierarchical based on label shape
        is_hierarchical = y.ndim == 2 and y.shape[1] > 1

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
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Predict labels for texts.

        Args:
            texts: Texts to classify
            show_progress: Show progress bar during embedding

        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        # Generate embeddings
        X = self.embed_texts(texts, show_progress=show_progress)

        # Predict
        return self._classifier.predict(X)

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

        Useful for exploring embeddings or using them elsewhere.

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
        return (
            f"TextClassifier("
            f"embedding_model='{self.embedding_model_name}', "
            f"classifier_type='{self.classifier_type}', "
            f"status='{status}')"
        )


class HierarchicalTextClassifier(TextClassifier):
    """
    Specialized text classifier for hierarchical labels.

    Convenience class that defaults to hierarchical classification.

    Example:
        >>> clf = HierarchicalTextClassifier()
        >>> # labels shape: (n_samples, n_hierarchy_levels)
        >>> clf.fit(texts, labels)
        >>> predictions = clf.predict(new_texts)  # shape: (n, n_levels)
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
            np.ndarray: Predictions for the specified level
        """
        predictions = self.predict(texts)
        if level >= predictions.shape[1]:
            raise ValueError(
                f"Level {level} out of range. Max level: {predictions.shape[1] - 1}"
            )
        return predictions[:, level]
