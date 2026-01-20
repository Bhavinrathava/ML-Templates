"""
Unit tests for embedding_generation module.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Mock sentence_transformers before importing EmbeddingGenerator
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers.SentenceTransformer"] = Mock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp_templates.utils.embedding_generation import EmbeddingGenerator


class TestEmbeddingGeneratorInit(unittest.TestCase):
    """Test EmbeddingGenerator initialization."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_initialization_with_default_model(self, mock_transformer):
        """Test initialization with default model."""
        mock_transformer.return_value = Mock()

        generator = EmbeddingGenerator()

        self.assertEqual(
            generator.model_name, "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.assertIsNotNone(generator.model)
        mock_transformer.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_initialization_with_custom_model(self, mock_transformer):
        """Test initialization with custom model name."""
        mock_transformer.return_value = Mock()
        custom_model = "sentence-transformers/all-mpnet-base-v2"

        generator = EmbeddingGenerator(model_name=custom_model)

        self.assertEqual(generator.model_name, custom_model)
        mock_transformer.assert_called_once_with(custom_model)

    @patch("sentence_transformers.SentenceTransformer")
    def test_initialization_import_error(self, mock_transformer):
        """Test initialization raises ImportError if sentence-transformers missing."""
        mock_transformer.side_effect = ImportError()

        with self.assertRaises(ImportError) as context:
            EmbeddingGenerator()

        self.assertIn("sentence-transformers", str(context.exception))

    @patch("sentence_transformers.SentenceTransformer")
    def test_initialization_runtime_error(self, mock_transformer):
        """Test initialization raises RuntimeError for invalid model."""
        mock_transformer.side_effect = Exception("Model not found")

        with self.assertRaises(RuntimeError) as context:
            EmbeddingGenerator(model_name="invalid/model")

        self.assertIn("Failed to load model", str(context.exception))


class TestEmbeddingGeneratorEmbeddings(unittest.TestCase):
    """Test embedding generation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the model
        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        self.mock_model.get_sentence_embedding_dimension.return_value = 3

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=self.mock_model,
        ):
            self.generator = EmbeddingGenerator()

    def test_generate_embeddings_from_list(self):
        """Test generating embeddings from a list of texts."""
        texts = ["Hello world", "How are you?", "Great day"]

        embeddings = self.generator.generate_embeddings(
            texts, show_progress_bar=False
        )

        self.assertEqual(embeddings.shape, (3, 3))
        self.mock_model.encode.assert_called_once()
        args, kwargs = self.mock_model.encode.call_args
        self.assertEqual(args[0], texts)
        self.assertFalse(kwargs["show_progress_bar"])

    def test_generate_embeddings_from_series(self):
        """Test generating embeddings from a pandas Series."""
        series = pd.Series(["Text 1", "Text 2", "Text 3"])

        embeddings = self.generator.generate_embeddings(
            series, show_progress_bar=False
        )

        self.assertEqual(embeddings.shape, (3, 3))
        args, _ = self.mock_model.encode.call_args
        self.assertEqual(args[0], series.tolist())

    def test_generate_embeddings_batch_size(self):
        """Test batch size parameter."""
        texts = ["Text 1", "Text 2"]
        batch_size = 16

        self.generator.generate_embeddings(
            texts, batch_size=batch_size, show_progress_bar=False
        )

        _, kwargs = self.mock_model.encode.call_args
        self.assertEqual(kwargs["batch_size"], batch_size)

    def test_generate_embeddings_empty_list(self):
        """Test error when generating embeddings for empty list."""
        with self.assertRaises(ValueError) as context:
            self.generator.generate_embeddings([])

        self.assertIn("cannot be empty", str(context.exception))

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        dim = self.generator.get_embedding_dimension()

        self.assertEqual(dim, 3)
        self.mock_model.get_sentence_embedding_dimension.assert_called_once()


class TestEmbeddingGeneratorDataFrame(unittest.TestCase):
    """Test dataframe embedding operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        self.mock_model.get_sentence_embedding_dimension.return_value = 2

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=self.mock_model,
        ):
            self.generator = EmbeddingGenerator()

        self.df = pd.DataFrame(
            {
                "title": ["Product A", "Product B", "Product C"],
                "description": ["Good quality", "Best seller", "Great value"],
                "price": [10, 20, 30],
            }
        )

    def test_add_embeddings_single_column(self):
        """Test adding embeddings from a single column."""
        result_df = self.generator.add_embeddings_to_dataframe(
            self.df, text_columns="title", embedding_column="embeddings"
        )

        self.assertIn("embeddings", result_df.columns)
        self.assertEqual(len(result_df), 3)
        self.assertEqual(len(result_df["embeddings"].iloc[0]), 2)
        # Original df should not be modified
        self.assertNotIn("embeddings", self.df.columns)

    def test_add_embeddings_multiple_columns(self):
        """Test adding embeddings from multiple columns."""
        result_df = self.generator.add_embeddings_to_dataframe(
            self.df,
            text_columns=["title", "description"],
            embedding_column="text_embeddings",
        )

        self.assertIn("text_embeddings", result_df.columns)
        self.assertEqual(len(result_df), 3)

        # Verify combined text was used
        args, _ = self.mock_model.encode.call_args
        texts = args[0]
        self.assertIn("Product A", texts[0])
        self.assertIn("Good quality", texts[0])

    def test_add_embeddings_custom_separator(self):
        """Test custom separator for combining text columns."""
        result_df = self.generator.add_embeddings_to_dataframe(
            self.df,
            text_columns=["title", "description"],
            separator=" | ",
            embedding_column="embeddings",
        )

        args, _ = self.mock_model.encode.call_args
        texts = args[0]
        self.assertIn(" | ", texts[0])

    def test_add_embeddings_missing_column(self):
        """Test error when text column doesn't exist."""
        with self.assertRaises(ValueError) as context:
            self.generator.add_embeddings_to_dataframe(
                self.df, text_columns="nonexistent_column"
            )

        self.assertIn("Missing columns", str(context.exception))

    def test_add_embeddings_missing_multiple_columns(self):
        """Test error when some text columns don't exist."""
        with self.assertRaises(ValueError) as context:
            self.generator.add_embeddings_to_dataframe(
                self.df, text_columns=["title", "nonexistent"]
            )

        self.assertIn("Missing columns", str(context.exception))
        self.assertIn("nonexistent", str(context.exception))

    def test_add_embeddings_handles_missing_values(self):
        """Test handling of missing values in text columns."""
        df_with_nan = self.df.copy()
        df_with_nan.loc[1, "title"] = None
        df_with_nan.loc[2, "description"] = None

        result_df = self.generator.add_embeddings_to_dataframe(
            df_with_nan,
            text_columns=["title", "description"],
            embedding_column="embeddings",
        )

        self.assertIn("embeddings", result_df.columns)
        # Should still have embeddings for all rows
        self.assertEqual(len(result_df), 3)

    def test_add_embeddings_custom_column_name(self):
        """Test custom embedding column name."""
        custom_name = "my_embeddings"
        result_df = self.generator.add_embeddings_to_dataframe(
            self.df, text_columns="title", embedding_column=custom_name
        )

        self.assertIn(custom_name, result_df.columns)
        self.assertNotIn("embedding", result_df.columns)

    def test_add_embeddings_batch_size(self):
        """Test batch size parameter for dataframe embeddings."""
        batch_size = 8
        self.generator.add_embeddings_to_dataframe(
            self.df, text_columns="title", batch_size=batch_size
        )

        _, kwargs = self.mock_model.encode.call_args
        self.assertEqual(kwargs["batch_size"], batch_size)


class TestEmbeddingGeneratorRepr(unittest.TestCase):
    """Test string representation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=self.mock_model,
        ):
            self.generator = EmbeddingGenerator(
                "sentence-transformers/all-mpnet-base-v2"
            )

    def test_repr(self):
        """Test __repr__ method."""
        repr_str = repr(self.generator)

        self.assertIn("EmbeddingGenerator", repr_str)
        self.assertIn("all-mpnet-base-v2", repr_str)
        self.assertIn("384", repr_str)


class TestEmbeddingGeneratorIntegration(unittest.TestCase):
    """Integration tests for EmbeddingGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.random.rand(3, 128)
        self.mock_model.get_sentence_embedding_dimension.return_value = 128

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=self.mock_model,
        ):
            self.generator = EmbeddingGenerator()

    def test_workflow_single_column(self):
        """Test complete workflow with single text column."""
        df = pd.DataFrame(
            {
                "text": [
                    "This is great",
                    "Not so good",
                    "Excellent product",
                ]
            }
        )

        result_df = self.generator.add_embeddings_to_dataframe(
            df, text_columns="text", embedding_column="embeddings"
        )

        # Verify all data is preserved
        self.assertEqual(len(result_df), len(df))
        self.assertTrue(all(result_df["text"] == df["text"]))
        self.assertIn("embeddings", result_df.columns)
        self.assertEqual(len(result_df["embeddings"].iloc[0]), 128)

    def test_workflow_multiple_columns(self):
        """Test complete workflow with multiple text columns."""
        df = pd.DataFrame(
            {
                "title": ["Item A", "Item B", "Item C"],
                "review": ["Very good", "Not bad", "Excellent"],
                "category": ["Electronics", "Books", "Clothing"],
            }
        )

        result_df = self.generator.add_embeddings_to_dataframe(
            df,
            text_columns=["title", "review"],
            embedding_column="combined_embeddings",
            separator=" - ",
        )

        # Verify structure
        self.assertEqual(len(result_df), 3)
        self.assertIn("combined_embeddings", result_df.columns)
        # Original columns preserved
        self.assertIn("category", result_df.columns)
        self.assertTrue(all(result_df["category"] == df["category"]))


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingGeneratorInit))
    suite.addTests(
        loader.loadTestsFromTestCase(TestEmbeddingGeneratorEmbeddings)
    )
    suite.addTests(
        loader.loadTestsFromTestCase(TestEmbeddingGeneratorDataFrame)
    )
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingGeneratorRepr))
    suite.addTests(
        loader.loadTestsFromTestCase(TestEmbeddingGeneratorIntegration)
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
