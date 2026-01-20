"""
Unit tests for MLflowManager class.
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp_templates.utils.mlflow_manager import MLflowManager


class TestMLflowManager(unittest.TestCase):
    """Test cases for MLflowManager."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create temporary directory for MLflow tracking
        cls.temp_dir = tempfile.mkdtemp()
        cls.tracking_uri = f"file:{cls.temp_dir}/mlruns"

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up test case."""
        self.manager = MLflowManager(
            tracking_uri=self.tracking_uri, experiment_name="test_experiment"
        )

    def tearDown(self):
        """Clean up after test case."""
        # End any active runs
        try:
            self.manager.end_run()
        except:
            pass

    def test_initialization(self):
        """Test MLflowManager initialization."""
        self.assertEqual(self.manager.tracking_uri, self.tracking_uri)
        self.assertEqual(self.manager.experiment_name, "test_experiment")
        self.assertEqual(self.manager.registry_uri, self.tracking_uri)

    def test_initialization_with_custom_registry_uri(self):
        """Test initialization with custom registry URI."""
        custom_registry = "http://registry:5000"
        manager = MLflowManager(
            tracking_uri=self.tracking_uri,
            experiment_name="test_experiment",
            registry_uri=custom_registry,
        )
        self.assertEqual(manager.registry_uri, custom_registry)

    def test_start_and_end_run(self):
        """Test starting and ending an MLflow run."""
        run_id = self.manager.start_run(run_name="test_run")
        self.assertIsNotNone(run_id)
        self.assertIsInstance(run_id, str)
        self.manager.end_run()

    def test_log_params(self):
        """Test logging parameters."""
        self.manager.start_run()
        params = {"learning_rate": 0.01, "batch_size": 32, "epochs": 10}

        # Should not raise an exception
        self.manager.log_params(params)
        self.manager.end_run()

    def test_log_metrics(self):
        """Test logging metrics."""
        self.manager.start_run()
        metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.89}

        # Should not raise an exception
        self.manager.log_metrics(metrics)
        self.manager.log_metrics({"loss": 0.05}, step=1)
        self.manager.end_run()

    def test_log_model_sklearn(self):
        """Test logging a scikit-learn model."""
        # Create a simple model
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=42
        )
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Log the model
        self.manager.start_run()
        model_uri = self.manager.log_model(
            model, "test_model", framework="sklearn"
        )
        self.manager.end_run()

        self.assertIsNotNone(model_uri)
        self.assertIn("runs:", model_uri)

    def test_log_params_and_metrics(self):
        """Test logging both parameters and metrics together."""
        self.manager.start_run(
            run_name="param_metric_test", tags={"type": "test"}
        )

        params = {"param1": 1.5, "param2": 100}
        metrics = {"metric1": 0.85, "metric2": 0.92}

        self.manager.log_params(params)
        self.manager.log_metrics(metrics)
        self.manager.end_run()

    @patch("mlflow.tracking.MlflowClient.search_registered_models")
    def test_list_registered_models(self, mock_search_models):
        """Test listing registered models."""
        # Mock the registry call to avoid hanging on unavailable registry
        mock_search_models.return_value = []

        models = self.manager.list_registered_models()
        self.assertIsInstance(models, list)
        mock_search_models.assert_called_once()

    def test_save_model_pickle(self):
        """Test pickle model saving utility."""
        model = {"test": "model"}
        path = MLflowManager._save_model_pickle(model, "test_model.pkl")

        self.assertTrue(os.path.exists(path))

        # Clean up
        if os.path.exists(path):
            os.remove(path)

    def test_multiple_runs(self):
        """Test managing multiple runs sequentially."""
        # First run
        run_id_1 = self.manager.start_run(run_name="run_1")
        self.manager.log_metrics({"score": 0.8})
        self.manager.end_run()

        # Second run
        run_id_2 = self.manager.start_run(run_name="run_2")
        self.manager.log_metrics({"score": 0.9})
        self.manager.end_run()

        # Run IDs should be different
        self.assertNotEqual(run_id_1, run_id_2)

    def test_log_model_with_artifacts(self):
        """Test logging model with additional artifacts."""
        # Create a simple model
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Create temporary artifacts directory
        with tempfile.TemporaryDirectory() as temp_artifacts:
            # Create a dummy artifact file
            artifact_file = os.path.join(temp_artifacts, "config.txt")
            with open(artifact_file, "w") as f:
                f.write("test config")

            self.manager.start_run()
            model_uri = self.manager.log_model(
                model,
                "test_model_with_artifacts",
                framework="sklearn",
                artifacts_path=temp_artifacts,
            )
            self.manager.end_run()

            self.assertIsNotNone(model_uri)

    def test_error_handling_invalid_model_load(self):
        """Test error handling when loading invalid model."""
        with self.assertRaises(RuntimeError):
            self.manager.load_model("invalid://model/uri")


class TestMLflowManagerIntegration(unittest.TestCase):
    """Integration tests for MLflowManager."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.tracking_uri = f"file:{cls.temp_dir}/mlruns"

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up test case."""
        self.manager = MLflowManager(
            tracking_uri=self.tracking_uri, experiment_name="integration_test"
        )

    def tearDown(self):
        """Clean up after test case."""
        try:
            self.manager.end_run()
        except:
            pass

    def test_full_workflow(self):
        """Test complete workflow: params -> metrics -> model."""
        # Create and train model
        X, y = make_classification(
            n_samples=100, n_features=10, random_state=42
        )
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        # Log everything
        self.manager.start_run(run_name="full_workflow_test")

        # Log hyperparameters
        self.manager.log_params(
            {
                "algorithm": "logistic_regression",
                "max_iter": 200,
                "random_state": 42,
            }
        )

        # Log metrics
        train_score = model.score(X, y)
        self.manager.log_metrics(
            {"train_accuracy": train_score, "cross_val_score": 0.92}
        )

        # Log model
        model_uri = self.manager.log_model(model, "integrated_model")

        self.manager.end_run()

        # Verify
        self.assertIsNotNone(model_uri)
        self.assertIn("runs:", model_uri)


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestMLflowManager))
    suite.addTests(loader.loadTestsFromTestCase(TestMLflowManagerIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
