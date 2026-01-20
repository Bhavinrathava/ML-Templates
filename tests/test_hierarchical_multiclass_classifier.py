"""
Tests for HierarchicalNNClassifier.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from nlp_templates.classifiers.hierarchical_multiclass_classifier import (
    HierarchicalNNClassifier,
)


class TestHierarchicalNNClassifier(unittest.TestCase):
    """Test cases for HierarchicalNNClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate hierarchical data
        # Level 0: [0, 1] (2 classes)
        # Level 1: [0, 1, 2] when L0=0, [0, 1] when L0=1
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10

        X = np.random.randn(self.n_samples, self.n_features)

        # Create hierarchical labels
        y_level0 = np.random.choice([0, 1], size=self.n_samples)
        y_level1 = np.where(
            y_level0 == 0,
            np.random.choice([0, 1, 2], size=self.n_samples),
            np.random.choice([0, 1], size=self.n_samples),
        )

        self.X = X
        self.y = np.column_stack([y_level0, y_level1])

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test classifier initialization."""
        clf = HierarchicalNNClassifier(name="test_hierarchical")
        self.assertEqual(clf.name, "test_hierarchical")
        self.assertEqual(clf.random_state, 42)
        self.assertEqual(clf.test_size, 0.3)

    def test_load_data(self):
        """Test data loading."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)

        self.assertEqual(clf.X_train.shape[1], self.n_features)
        self.assertEqual(clf.X_test.shape[1], self.n_features)
        self.assertEqual(clf.y_train_hierarchy.shape[1], 2)
        self.assertEqual(clf.y_test_hierarchy.shape[1], 2)

        # Check hierarchy levels detected
        self.assertEqual(clf.hierarchy_levels, 2)

    def test_load_data_invalid_dimensions(self):
        """Test error when y is 1D."""
        clf = HierarchicalNNClassifier()
        y_1d = np.array([0, 1, 0, 1])

        with self.assertRaises(ValueError):
            clf.load_data(self.X[:4], y_1d)

    def test_build_model(self):
        """Test model building."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)
        clf.build_model()

        # Check level 0 classifier exists
        self.assertIn(0, clf.level_classifiers)
        self.assertIn("classifier", clf.level_classifiers[0])

        # Check level 1 classifiers exist for each level 0 class
        self.assertIn(1, clf.level_classifiers)
        self.assertGreater(len(clf.level_classifiers[1]), 0)

    def test_train(self):
        """Test model training."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)
        clf.build_model()

        # Should not raise
        clf.train()

    def test_predict(self):
        """Test prediction."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)
        clf.build_model()
        clf.train()

        predictions = clf.predict(clf.X_test)

        self.assertEqual(predictions.shape, clf.y_test_hierarchy.shape)
        self.assertEqual(predictions.shape[1], 2)  # 2 levels

    def test_predict_proba(self):
        """Test probability predictions."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)
        clf.build_model()
        clf.train()

        probabilities = clf.predict_proba(clf.X_test)

        self.assertEqual(len(probabilities), 2)  # 2 levels
        self.assertEqual(probabilities[0].shape[0], clf.X_test.shape[0])

    def test_evaluate(self):
        """Test evaluation."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)
        clf.build_model()
        clf.train()

        results = clf.evaluate()

        self.assertIn("train_metrics", results)
        self.assertIn("test_metrics", results)
        self.assertIn("confusion_matrices", results)

        # Check metrics for both levels
        self.assertIn("level_0", results["test_metrics"])
        self.assertIn("level_1", results["test_metrics"])

        # Check metric keys
        for level_key in results["test_metrics"]:
            metrics = results["test_metrics"][level_key]
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1", metrics)

    def test_full_pipeline(self):
        """Test full pipeline."""
        clf = HierarchicalNNClassifier()

        results = clf.full_pipeline(
            self.X,
            self.y,
            output_dir=self.temp_dir,
            log_to_mlflow=False,
        )

        self.assertIn("train_metrics", results)
        self.assertIn("test_metrics", results)
        self.assertGreater(len(results["test_metrics"]), 0)

    def test_config_loading(self):
        """Test loading from config."""
        # Create a config file
        import tempfile
        import yaml

        config = {
            "model": {
                "params": {
                    "hidden_dims": [64, 32],
                    "activation": "relu",
                    "epochs": 20,
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            clf = HierarchicalNNClassifier(config_path=config_path)
            self.assertEqual(clf.config_path, config_path)
            self.assertIn("model", clf.config)
        finally:
            Path(config_path).unlink()

    def test_hierarchy_with_3_levels(self):
        """Test with 3-level hierarchy."""
        # Create 3-level hierarchical data
        y_level0 = np.array([0, 0, 1, 1] * 50)
        y_level1 = np.array([0, 1, 0, 1] * 50)
        y_level2 = np.array([0, 0, 1, 1] * 50)
        y_3level = np.column_stack([y_level0, y_level1, y_level2])

        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, y_3level)

        self.assertEqual(clf.hierarchy_levels, 3)

        clf.build_model()
        self.assertIn(0, clf.level_classifiers)
        self.assertIn(1, clf.level_classifiers)
        self.assertIn(2, clf.level_classifiers)

        clf.train()
        predictions = clf.predict(clf.X_test)
        self.assertEqual(predictions.shape[1], 3)

    def test_predict_before_train_error(self):
        """Test error when predicting before training."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)

        with self.assertRaises(ValueError):
            clf.predict(self.X)

    def test_evaluate_before_train_error(self):
        """Test error when evaluating before training."""
        clf = HierarchicalNNClassifier()
        clf.load_data(self.X, self.y)

        with self.assertRaises(ValueError):
            clf.evaluate()


if __name__ == "__main__":
    unittest.main()
