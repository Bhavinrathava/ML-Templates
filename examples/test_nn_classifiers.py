"""
Test Script for Neural Network Classifiers
============================================

This script tests both the SimpleMulticlassClassifier (with neural_network backend)
and the HierarchicalNNClassifier using synthetic data.

Usage:
    python examples/test_nn_classifiers.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from nlp_templates.classifiers.simple_multiclass_classifier import (
    SimpleMulticlassClassifier,
)
from nlp_templates.classifiers.hierarchical_multiclass_classifier import (
    HierarchicalNNClassifier,
)


def create_simple_classification_data(
    n_samples: int = 500,
    n_features: int = 20,
    n_classes: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data for simple multiclass classification.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    return X, y


def create_hierarchical_classification_data(
    n_samples: int = 500,
    n_features: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data for hierarchical classification.

    Creates a 2-level hierarchy:
        Level 0: 3 classes (0, 1, 2)
        Level 1: Varying child classes per parent
            - Parent 0 -> children [0, 1, 2]
            - Parent 1 -> children [0, 1]
            - Parent 2 -> children [0, 1, 2, 3]

    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Hierarchical labels (n_samples, 2)
    """
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate level 0 labels (3 classes)
    y_level0 = np.random.choice([0, 1, 2], size=n_samples)

    # Generate level 1 labels based on level 0 (hierarchical structure)
    y_level1 = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if y_level0[i] == 0:
            y_level1[i] = np.random.choice([0, 1, 2])  # 3 child classes
        elif y_level0[i] == 1:
            y_level1[i] = np.random.choice([0, 1])  # 2 child classes
        else:
            y_level1[i] = np.random.choice([0, 1, 2, 3])  # 4 child classes

    y = np.column_stack([y_level0, y_level1])
    return X, y


def test_simple_multiclass_classifier():
    """Test SimpleMulticlassClassifier with neural network backend."""
    print("=" * 70)
    print("Testing SimpleMulticlassClassifier with Neural Network")
    print("=" * 70)

    # Create data
    print("\n[1] Creating synthetic classification data...")
    X, y = create_simple_classification_data(
        n_samples=500, n_features=20, n_classes=4
    )
    print(f"    Data shape: X={X.shape}, y={y.shape}")
    print(f"    Classes: {np.unique(y)}")

    # Initialize classifier with neural network config
    print("\n[2] Initializing SimpleMulticlassClassifier...")
    classifier = SimpleMulticlassClassifier(
        name="simple_nn_test",
        random_state=42,
        test_size=0.3,
    )

    # Set neural network configuration manually
    classifier.config = {
        "model": {
            "type": "neural_network",
            "params": {
                "hidden_dims": [64, 32],
                "activation": "relu",
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": 30,
                "batch_size": 32,
            },
        }
    }

    # Load data
    print("\n[3] Loading data...")
    classifier.load_data(X, y)
    print(f"    Train size: {classifier.X_train.shape[0]}")
    print(f"    Test size: {classifier.X_test.shape[0]}")

    # Build and train model
    print("\n[4] Building neural network model...")
    classifier.build_model()
    print(f"    Model type: {classifier.model_type}")

    print("\n[5] Training model...")
    classifier.train()

    # Evaluate
    print("\n[6] Evaluating model...")
    results = classifier.evaluate()

    print("\n    Results:")
    print("    " + "-" * 50)
    print(f"    Train Accuracy: {results['train_metrics']['accuracy']:.4f}")
    print(f"    Train F1:       {results['train_metrics']['f1']:.4f}")
    print(f"    Test Accuracy:  {results['test_metrics']['accuracy']:.4f}")
    print(f"    Test F1:        {results['test_metrics']['f1']:.4f}")

    # Test prediction
    print("\n[7] Testing predictions...")
    sample_X = X[:5]
    predictions = classifier.predict(sample_X)
    print(f"    Sample predictions: {predictions}")

    # Test predict_proba
    print("\n[8] Testing probability predictions...")
    proba = classifier.predict_proba(sample_X)
    print(f"    Probability shape: {proba.shape}")
    print(f"    Sample probabilities (first row): {proba[0].round(3)}")

    print("\n" + "=" * 70)
    print("SimpleMulticlassClassifier test PASSED")
    print("=" * 70)

    return results


def test_hierarchical_nn_classifier():
    """Test HierarchicalNNClassifier."""
    print("\n" + "=" * 70)
    print("Testing HierarchicalNNClassifier")
    print("=" * 70)

    # Create data
    print("\n[1] Creating synthetic hierarchical data...")
    X, y = create_hierarchical_classification_data(
        n_samples=500, n_features=20
    )
    print(f"    Data shape: X={X.shape}, y={y.shape}")
    print(f"    Hierarchy levels: {y.shape[1]}")
    print(f"    Level 0 classes: {np.unique(y[:, 0])}")
    print(f"    Level 1 classes: {np.unique(y[:, 1])}")

    # Initialize classifier
    print("\n[2] Initializing HierarchicalNNClassifier...")
    classifier = HierarchicalNNClassifier(
        name="hierarchical_nn_test",
        random_state=42,
        test_size=0.3,
    )

    # Set config for faster training
    classifier.config = {
        "model": {
            "params": {
                "hidden_dims": [64, 32],
                "activation": "relu",
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": 30,
                "batch_size": 32,
            }
        }
    }

    # Load data
    print("\n[3] Loading hierarchical data...")
    classifier.load_data(X, y)
    print(f"    Train size: {classifier.X_train.shape[0]}")
    print(f"    Test size: {classifier.X_test.shape[0]}")
    print(f"    Hierarchy levels detected: {classifier.hierarchy_levels}")

    # Build model
    print("\n[4] Building hierarchical model...")
    classifier.build_model()
    print(f"    Level classifiers built: {list(classifier.level_classifiers.keys())}")

    # Train
    print("\n[5] Training hierarchical model...")
    classifier.train()

    # Evaluate
    print("\n[6] Evaluating model...")
    results = classifier.evaluate()

    print("\n    Results by Level:")
    print("    " + "-" * 50)
    for level in range(classifier.hierarchy_levels):
        train_acc = results["train_metrics"][f"level_{level}"]["accuracy"]
        test_acc = results["test_metrics"][f"level_{level}"]["accuracy"]
        train_f1 = results["train_metrics"][f"level_{level}"]["f1"]
        test_f1 = results["test_metrics"][f"level_{level}"]["f1"]
        print(f"    Level {level}:")
        print(f"      Train - Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"      Test  - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

    # Test prediction
    print("\n[7] Testing predictions...")
    sample_X = X[:5]
    predictions = classifier.predict(sample_X)
    print(f"    Prediction shape: {predictions.shape}")
    print(f"    Sample predictions (first 3):")
    for i in range(3):
        print(f"      Sample {i}: Level0={predictions[i, 0]}, Level1={predictions[i, 1]}")

    # Test predict_proba
    print("\n[8] Testing probability predictions...")
    probabilities = classifier.predict_proba(sample_X)
    print(f"    Number of probability arrays: {len(probabilities)}")
    for i, proba in enumerate(probabilities):
        print(f"    Level {i} probabilities shape: {proba.shape}")

    print("\n" + "=" * 70)
    print("HierarchicalNNClassifier test PASSED")
    print("=" * 70)

    return results


def test_hierarchical_3_levels():
    """Test HierarchicalNNClassifier with 3 hierarchy levels."""
    print("\n" + "=" * 70)
    print("Testing HierarchicalNNClassifier with 3 Levels")
    print("=" * 70)

    # Create 3-level hierarchical data
    print("\n[1] Creating 3-level hierarchical data...")
    np.random.seed(42)
    n_samples = 400
    n_features = 15

    X = np.random.randn(n_samples, n_features)

    # Level 0: 2 classes
    y_level0 = np.random.choice([0, 1], size=n_samples)

    # Level 1: depends on level 0
    y_level1 = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if y_level0[i] == 0:
            y_level1[i] = np.random.choice([0, 1, 2])
        else:
            y_level1[i] = np.random.choice([0, 1])

    # Level 2: depends on level 1
    y_level2 = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if y_level1[i] == 0:
            y_level2[i] = np.random.choice([0, 1])
        elif y_level1[i] == 1:
            y_level2[i] = np.random.choice([0, 1, 2])
        else:
            y_level2[i] = np.random.choice([0, 1])

    y = np.column_stack([y_level0, y_level1, y_level2])
    print(f"    Data shape: X={X.shape}, y={y.shape}")
    print(f"    Hierarchy levels: {y.shape[1]}")

    # Initialize and train
    print("\n[2] Training 3-level hierarchical classifier...")
    classifier = HierarchicalNNClassifier(
        name="hierarchical_3level_test",
        random_state=42,
        test_size=0.3,
    )

    classifier.config = {
        "model": {
            "params": {
                "hidden_dims": [32, 16],
                "epochs": 20,
                "batch_size": 32,
            }
        }
    }

    classifier.load_data(X, y)
    classifier.build_model()
    classifier.train()

    # Evaluate
    print("\n[3] Evaluating 3-level model...")
    results = classifier.evaluate()

    print("\n    Results by Level:")
    print("    " + "-" * 50)
    for level in range(3):
        test_acc = results["test_metrics"][f"level_{level}"]["accuracy"]
        test_f1 = results["test_metrics"][f"level_{level}"]["f1"]
        print(f"    Level {level}: Accuracy={test_acc:.4f}, F1={test_f1:.4f}")

    # Test predictions
    predictions = classifier.predict(X[:5])
    print(f"\n    Sample predictions shape: {predictions.shape}")
    print(f"    Sample prediction: {predictions[0]}")

    print("\n" + "=" * 70)
    print("3-Level HierarchicalNNClassifier test PASSED")
    print("=" * 70)

    return results


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "NN CLASSIFIER TEST SUITE" + " " * 24 + "#")
    print("#" * 70)

    # Test simple classifier
    simple_results = test_simple_multiclass_classifier()

    # Test hierarchical classifier (2 levels)
    hierarchical_results = test_hierarchical_nn_classifier()

    # Test hierarchical classifier (3 levels)
    hierarchical_3level_results = test_hierarchical_3_levels()

    # Summary
    print("\n" + "#" * 70)
    print("#" + " " * 25 + "TEST SUMMARY" + " " * 31 + "#")
    print("#" * 70)

    print("\n  All tests completed successfully!")
    print("\n  SimpleMulticlassClassifier (Neural Network):")
    print(f"    - Test Accuracy: {simple_results['test_metrics']['accuracy']:.4f}")
    print(f"    - Test F1: {simple_results['test_metrics']['f1']:.4f}")

    print("\n  HierarchicalNNClassifier (2 levels):")
    for level in range(2):
        acc = hierarchical_results["test_metrics"][f"level_{level}"]["accuracy"]
        print(f"    - Level {level} Test Accuracy: {acc:.4f}")

    print("\n  HierarchicalNNClassifier (3 levels):")
    for level in range(3):
        acc = hierarchical_3level_results["test_metrics"][f"level_{level}"]["accuracy"]
        print(f"    - Level {level} Test Accuracy: {acc:.4f}")

    print("\n" + "#" * 70)


if __name__ == "__main__":
    main()
