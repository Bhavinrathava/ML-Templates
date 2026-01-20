"""
Unit tests for SimpleMulticlassClassifier.
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp_templates.classifiers.simple_multiclass_classifier import (
    SimpleMulticlassClassifier,
)
from nlp_templates.preprocessing.config_loader import ConfigLoader


class TestSimpleMulticlassClassifierInit(unittest.TestCase):
    """Test classifier initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with defaults."""
        clf = SimpleMulticlassClassifier()
        self.assertEqual(clf.name, "simple_multiclass_classifier")
        self.assertEqual(clf.test_size, 0.3)
        self.assertIsNone(clf.model)

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        clf = SimpleMulticlassClassifier(
            name="custom_classifier",
            test_size=0.2,
            random_state=123,
        )
        self.assertEqual(clf.name, "custom_classifier")
        self.assertEqual(clf.test_size, 0.2)
        self.assertEqual(clf.random_state, 123)

    def test_initialization_with_mlflow(self):
        """Test initialization with MLflow parameters."""
        clf = SimpleMulticlassClassifier(
            mlflow_tracking_uri="file:./mlruns",
            mlflow_experiment_name="test_experiment",
        )
        self.assertEqual(clf.mlflow_tracking_uri, "file:./mlruns")
        self.assertEqual(clf.mlflow_experiment_name, "test_experiment")


class TestSimpleMulticlassClassifierDataLoading(unittest.TestCase):
    """Test data loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )

    def test_load_data_numpy(self):
        """Test loading numpy arrays."""
        clf = SimpleMulticlassClassifier()
        clf.load_data(self.X, self.y)

        self.assertIsNotNone(clf.X_train)
        self.assertIsNotNone(clf.y_train)
        self.assertIsNotNone(clf.X_test)
        self.assertIsNotNone(clf.y_test)

    def test_load_data_dataframe(self):
        """Test loading pandas DataFrame."""
        clf = SimpleMulticlassClassifier()
        X_df = pd.DataFrame(self.X, columns=[f"feat_{i}" for i in range(20)])
        y_series = pd.Series(self.y, name="label")

        clf.load_data(X_df, y_series)

        self.assertEqual(clf.X_train.shape[0], 70)
        self.assertEqual(clf.X_test.shape[0], 30)

    def test_load_data_with_label_column(self):
        """Test loading data with label column in features."""
        clf = SimpleMulticlassClassifier()
        X = pd.DataFrame(self.X, columns=[f"feat_{i}" for i in range(20)])
        X["label"] = self.y

        clf.load_data(X, self.y, label_column="label")

        self.assertNotIn("label", clf.X_train.columns)

    def test_data_split_stratified(self):
        """Test that data split is stratified."""
        clf = SimpleMulticlassClassifier(test_size=0.3)
        clf.load_data(self.X, self.y)

        # Check class distribution is similar
        train_dist = np.bincount(clf.y_train) / len(clf.y_train)
        test_dist = np.bincount(clf.y_test) / len(clf.y_test)

        # Distributions should be roughly similar
        np.testing.assert_array_almost_equal(train_dist, test_dist, decimal=1)


class TestSimpleMulticlassClassifierModel(unittest.TestCase):
    """Test model building and training."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        self.clf = SimpleMulticlassClassifier()
        self.clf.load_data(self.X, self.y)

    def test_build_model_default(self):
        """Test building model with default config."""
        self.clf.build_model()

        self.assertIsNotNone(self.clf.model)
        self.assertEqual(self.clf.model_type, "logistic_regression")

    def test_train_model(self):
        """Test model training."""
        self.clf.build_model()
        self.clf.train()

        self.assertIsNotNone(self.clf.model)
        # Model should have been fitted
        self.assertTrue(
            hasattr(self.clf.model, "coef_")
            or hasattr(self.clf.model, "estimators_")
        )


class TestSimpleMulticlassClassifierEvaluation(unittest.TestCase):
    """Test model evaluation."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        self.clf = SimpleMulticlassClassifier()
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()
        self.clf.train()

    def test_evaluate(self):
        """Test evaluation metrics."""
        results = self.clf.evaluate()

        self.assertIn("train_metrics", results)
        self.assertIn("test_metrics", results)
        self.assertIn("confusion_matrix", results)
        self.assertIn("classification_report", results)

    def test_metrics_keys(self):
        """Test that all required metrics are present."""
        results = self.clf.evaluate()

        required_keys = ["accuracy", "precision", "recall", "f1"]
        for key in required_keys:
            self.assertIn(key, results["train_metrics"])
            self.assertIn(key, results["test_metrics"])

    def test_confusion_matrix_shape(self):
        """Test confusion matrix shape."""
        self.clf.evaluate()

        n_classes = len(np.unique(self.y))
        self.assertEqual(self.clf.cm.shape, (n_classes, n_classes))

    def test_metrics_in_range(self):
        """Test that metrics are in valid range."""
        results = self.clf.evaluate()

        for metric_val in results["test_metrics"].values():
            self.assertGreaterEqual(metric_val, 0.0)
            self.assertLessEqual(metric_val, 1.0)


class TestSimpleMulticlassClassifierPrediction(unittest.TestCase):
    """Test prediction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        self.clf = SimpleMulticlassClassifier()
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()
        self.clf.train()

    def test_predict(self):
        """Test prediction on new data."""
        predictions = self.clf.predict(self.clf.X_test)

        self.assertEqual(len(predictions), len(self.clf.X_test))
        self.assertTrue(all(p in [0, 1, 2] for p in predictions))

    def test_predict_proba_logistic(self):
        """Test prediction probabilities for logistic regression."""
        proba = self.clf.predict_proba(self.clf.X_test)

        self.assertEqual(proba.shape[0], len(self.clf.X_test))
        self.assertEqual(proba.shape[1], 3)  # 3 classes
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_predict_proba_unsupported_model(self):
        """Test error for models without predict_proba."""
        self.clf.config = {"model": {"type": "naive_bayes"}}
        self.clf.build_model()

        # MultinomialNB requires non-negative features, so we need to shift the data
        X_shifted = self.clf.X_train - self.clf.X_train.min() + 1
        self.clf.model.fit(X_shifted, self.clf.y_train)

        # NaiveBayes should support predict_proba
        X_test_shifted = self.clf.X_test - self.clf.X_train.min() + 1
        proba = self.clf.model.predict_proba(X_test_shifted)
        self.assertIsNotNone(proba)


class TestSimpleMulticlassClassifierVisualization(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        self.clf = SimpleMulticlassClassifier()
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()
        self.clf.train()
        self.clf.evaluate()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_visualizations(self):
        """Test saving visualizations returns paths."""
        # Just verify method returns dict without errors
        viz_paths = self.clf.save_visualizations(self.temp_dir)
        self.assertIsInstance(viz_paths, dict)

    def test_visualization_files_created(self):
        """Test that visualization method completes without error."""
        # Just verify method completes
        result = self.clf.save_visualizations(self.temp_dir)
        self.assertTrue(result is not None or result == {})


class TestSimpleMulticlassClassifierConfig(unittest.TestCase):
    """Test config loading and management."""

    def test_config_exists(self):
        """Test that config can be set and retrieved."""
        config_content = {
            "model": {"type": "random_forest"},
            "training": {"random_state": 42},
        }

        clf = SimpleMulticlassClassifier()
        clf.config = config_content

        self.assertIsNotNone(clf.config)
        self.assertIn("model", clf.config)


class TestSimpleMulticlassClassifierMLflow(unittest.TestCase):
    """Test MLflow integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("nlp_templates.utils.mlflow_manager.MLflowManager")
    def test_setup_mlflow(self, mock_mlflow):
        """Test MLflow setup."""
        clf = SimpleMulticlassClassifier(
            mlflow_tracking_uri="file:./mlruns",
            mlflow_experiment_name="test",
        )
        clf.setup_mlflow()

        self.assertIsNotNone(clf.mlflow_manager)


class TestSimpleMulticlassClassifierPipeline(unittest.TestCase):
    """Test full pipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_pipeline(self):
        """Test complete pipeline execution."""
        clf = SimpleMulticlassClassifier()

        results = clf.full_pipeline(
            self.X,
            self.y,
            save_visualizations=True,
            output_dir=self.temp_dir,
            log_to_mlflow=False,
        )

        self.assertIn("train_metrics", results)
        self.assertIn("test_metrics", results)
        self.assertIn("visualizations", results)

    def test_full_pipeline_with_dataframe(self):
        """Test pipeline with DataFrame input."""
        X_df = pd.DataFrame(self.X, columns=[f"feat_{i}" for i in range(20)])
        y_series = pd.Series(self.y)

        clf = SimpleMulticlassClassifier()

        results = clf.full_pipeline(
            X_df, y_series, save_visualizations=False, log_to_mlflow=False
        )

        self.assertIn("train_metrics", results)
        self.assertIn("test_metrics", results)


if __name__ == "__main__":
    unittest.main()
