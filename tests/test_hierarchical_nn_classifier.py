"""
Tests for HierarchicalNNClassifier.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path

from nlp_templates.classifiers.hierarchical_nn_classifier import (
    HierarchicalNNClassifier,
)


class TestHierarchicalNNClassifier(unittest.TestCase):
    """Tests for HierarchicalNNClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic hierarchical data
        np.random.seed(42)

        self.n_samples = 150
        self.n_features = 10

        # Create features
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Create hierarchical labels (level1, level2)
        # Level 1: A, B, C
        # Level 2 (if A): A1, A2
        # Level 2 (if B): B1, B2, B3
        # Level 2 (if C): C1, C2
        self.y = []
        for i in range(self.n_samples):
            if i < 50:
                level1 = "A"
                level2 = "A1" if i < 25 else "A2"
            elif i < 100:
                level1 = "B"
                level2 = ["B1", "B2", "B3"][(i - 50) % 3]
            else:
                level1 = "C"
                level2 = "C1" if i < 125 else "C2"

            self.y.append((level1, level2))

        # Create classifier
        self.clf = HierarchicalNNClassifier(
            name="test_hierarchical_classifier",
            random_state=42,
        )

    def test_initialization(self):
        """Test classifier initialization."""
        self.assertEqual(self.clf.name, "test_hierarchical_classifier")
        self.assertEqual(self.clf.random_state, 42)
        self.assertIsNone(self.clf.level_1_classifier)
        self.assertEqual(len(self.clf.level_classifiers), 0)

    def test_load_data(self):
        """Test data loading."""
        self.clf.load_data(self.X, self.y)

        self.assertEqual(self.clf.n_levels, 2)
        self.assertEqual(self.clf.X.shape, (self.n_samples, self.n_features))
        self.assertEqual(len(self.clf.hierarchy_data), 3)  # A, B, C
        self.assertIn("A", self.clf.hierarchy_data)
        self.assertIn("B", self.clf.hierarchy_data)
        self.assertIn("C", self.clf.hierarchy_data)

    def test_build_model(self):
        """Test model building."""
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()

        self.assertIsNotNone(self.clf.model_params)
        self.assertIn("random_state", self.clf.model_params)

    def test_train(self):
        """Test model training."""
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()
        self.clf.train()

        # Check that classifiers are trained
        self.assertIsNotNone(self.clf.level_1_classifier)
        self.assertGreater(len(self.clf.level_classifiers), 0)

    def test_predict(self):
        """Test prediction."""
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()
        self.clf.train()

        # Test predictions
        predictions = self.clf.predict(self.X[:5])

        self.assertEqual(predictions.shape[0], 5)
        self.assertEqual(predictions.shape[1], 2)  # Two levels
        # Check that level 1 predictions are valid
        for pred in predictions[:, 0]:
            self.assertIn(pred, ["A", "B", "C"])

    def test_predict_proba(self):
        """Test probability predictions."""
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()
        self.clf.train()

        # Test predict_proba
        proba = self.clf.predict_proba(self.X[:5])

        self.assertIn(1, proba)  # Level 1 probabilities
        self.assertEqual(proba[1].shape[0], 5)
        self.assertEqual(proba[1].shape[1], 3)  # A, B, C

        if 2 in proba:
            self.assertEqual(proba[2].shape[0], 5)

    def test_evaluate(self):
        """Test evaluation."""
        self.clf.load_data(self.X, self.y)
        self.clf.build_model()
        self.clf.train()

        results = self.clf.evaluate()

        self.assertIn("train_metrics", results)
        self.assertIn("accuracy", results["train_metrics"])
        self.assertGreaterEqual(results["train_metrics"]["accuracy"], 0)
        self.assertLessEqual(results["train_metrics"]["accuracy"], 1)

    def test_three_level_hierarchy(self):
        """Test three-level hierarchy."""
        # Create three-level labels
        y_three = [
            ("A", "A1", "A1a"),
            ("A", "A1", "A1b"),
            ("A", "A2", "A2a"),
            ("B", "B1", "B1a"),
            ("B", "B1", "B1b"),
        ] * 30

        X_test = np.random.randn(len(y_three), self.n_features)

        clf = HierarchicalNNClassifier(random_state=42)
        clf.load_data(X_test, y_three)

        self.assertEqual(clf.n_levels, 3)
        self.assertIn("A", clf.hierarchy_data)
        self.assertIn("B", clf.hierarchy_data)


class TestHierarchicalNNClassifierIntegration(unittest.TestCase):
    """Integration tests for HierarchicalNNClassifier."""

    def setUp(self):
        """Set up integration tests."""
        np.random.seed(42)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_full_pipeline(self):
        """Test full training and prediction pipeline."""
        # Generate data
        X = np.random.randn(200, 15)
        y = [
            ("cat", "tabby") if i % 2 == 0 else ("dog", "golden")
            for i in range(100)
        ]
        y.extend(
            [
                ("cat", "persian") if i % 2 == 0 else ("dog", "labrador")
                for i in range(100)
            ]
        )

        # Train
        clf = HierarchicalNNClassifier(random_state=42)
        clf.load_data(X, y)
        clf.build_model()
        clf.train()

        # Evaluate
        results = clf.evaluate()
        self.assertIn("train_metrics", results)

        # Predict
        predictions = clf.predict(X[:10])
        self.assertEqual(predictions.shape[0], 10)
        self.assertEqual(predictions.shape[1], 2)

        # Probabilities
        proba = clf.predict_proba(X[:10])
        self.assertIn(1, proba)


if __name__ == "__main__":
    unittest.main()
