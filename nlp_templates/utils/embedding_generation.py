"""
Embedding generation utilities using HuggingFace models.
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np


class EmbeddingGenerator:
    """
    Generate embeddings for text data using HuggingFace models.

    This class wraps HuggingFace transformers to easily generate embeddings
    for dataframe columns.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the embedding generator with a HuggingFace model.

        Args:
            model_name (str): HuggingFace model name.
                            Defaults to "sentence-transformers/all-MiniLM-L6-v2"

        Example:
            >>> generator = EmbeddingGenerator("sentence-transformers/all-mpnet-base-v2")
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")

    def generate_embeddings(
        self,
        texts: Union[List[str], pd.Series],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (Union[List[str], pd.Series]): Texts to embed
            batch_size (int): Batch size for processing. Defaults to 32
            show_progress_bar (bool): Show progress bar. Defaults to True

        Returns:
            np.ndarray: Embeddings array of shape (n_texts, embedding_dim)

        Example:
            >>> texts = ["Hello world", "How are you?"]
            >>> embeddings = generator.generate_embeddings(texts)
            >>> print(embeddings.shape)
            (2, 384)
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        if not texts:
            raise ValueError("Texts list cannot be empty")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        return embeddings

    def add_embeddings_to_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: Union[str, List[str]],
        embedding_column: str = "embedding",
        separator: str = " ",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Generate embeddings for dataframe and add as new column.

        Args:
            df (pd.DataFrame): Input dataframe
            text_columns (Union[str, List[str]]): Column name(s) to use for embeddings.
                                                 If list, texts are concatenated.
            embedding_column (str): Name of the new embedding column. Defaults to "embedding"
            separator (str): Separator to use when combining text columns. Defaults to " "
            batch_size (int): Batch size for processing. Defaults to 32

        Returns:
            pd.DataFrame: DataFrame with new embedding column added

        Example:
            >>> df = pd.DataFrame({
            ...     'title': ['Hello', 'World'],
            ...     'description': ['A greeting', 'The planet']
            ... })
            >>> df_with_embeddings = generator.add_embeddings_to_dataframe(
            ...     df,
            ...     text_columns=['title', 'description'],
            ...     embedding_column='embeddings'
            ... )
        """
        # Normalize text_columns to list
        if isinstance(text_columns, str):
            text_columns = [text_columns]

        # Validate columns exist
        missing_cols = set(text_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Combine text from multiple columns
        if len(text_columns) == 1:
            texts = df[text_columns[0]].fillna("").astype(str)
        else:
            texts = (
                df[text_columns]
                .fillna("")
                .astype(str)
                .agg(separator.join, axis=1)
            )

        # Generate embeddings
        embeddings = self.generate_embeddings(
            texts.tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
        )

        # Add to dataframe
        df_copy = df.copy()
        df_copy[embedding_column] = embeddings.tolist()

        return df_copy

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from the model.

        Returns:
            int: Embedding dimension

        Example:
            >>> dim = generator.get_embedding_dimension()
            >>> print(f"Embedding dimension: {dim}")
            Embedding dimension: 384
        """
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        """String representation."""
        dim = self.get_embedding_dimension()
        return (
            f"EmbeddingGenerator(model='{self.model_name}', "
            f"embedding_dim={dim})"
        )
