"""
Example usage of SimpleMulticlassClassifier.

Demonstrates:
1. Loading config from file
2. Creating and training a classifier
3. Evaluating with metrics and visualizations
4. Logging to MLflow
5. Making predictions
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from nlp_templates.classifiers.simple_multiclass_classifier import (
    SimpleMulticlassClassifier,
)


def example_basic_usage():
    """Basic example: train and evaluate without config file."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage (No Config)")
    print("=" * 80)

    # Generate sample data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Initialize classifier
    clf = SimpleMulticlassClassifier(name="basic_classifier")

    # Load data
    clf.load_data(X, y)

    # Build and train with default config
    clf.build_model()
    clf.train()

    # Evaluate
    results = clf.evaluate()
    print("\nTrain Metrics:", results["train_metrics"])
    print("Test Metrics:", results["test_metrics"])

    # Make predictions
    predictions = clf.predict(clf.X_test[:5])
    print(f"\nPredictions for first 5 samples: {predictions}")

    # Save visualizations
    viz_paths = clf.save_visualizations("outputs/example1")
    print(f"Visualizations saved to: {viz_paths}")


def example_with_config_file():
    """Example using config file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Using Config File")
    print("=" * 80)

    # Generate sample data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Initialize classifier with config file
    clf = SimpleMulticlassClassifier(
        name="config_driven_classifier",
        config_path="examples/config_example.yaml",
    )

    # Load data
    clf.load_data(X, y)

    # Build and train
    clf.build_model()
    clf.train()

    # Evaluate
    results = clf.evaluate()
    print("\nTrain Metrics:", results["train_metrics"])
    print("Test Metrics:", results["test_metrics"])


def example_with_mlflow():
    """Example with MLflow integration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: With MLflow Integration")
    print("=" * 80)

    # Generate sample data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Initialize classifier with MLflow
    clf = SimpleMulticlassClassifier(
        name="mlflow_classifier",
        config_path="examples/config_example.yaml",
        mlflow_tracking_uri="file:./mlruns",
        mlflow_experiment_name="simple_multiclass_experiment",
    )

    # Setup MLflow
    clf.setup_mlflow()

    # Load data
    clf.load_data(X, y)

    # Build and train
    clf.build_model()
    clf.train()

    # Evaluate
    results = clf.evaluate()

    # Save visualizations
    viz_paths = clf.save_visualizations("outputs/example3")

    # Log to MLflow
    run_id = clf.log_to_mlflow(
        run_name="example_run_3",
        log_artifacts=True,
        artifacts_dir="outputs/example3",
    )

    print(f"\nMLflow run ID: {run_id}")
    print(f"Visualizations: {viz_paths}")


def example_full_pipeline():
    """Example using full_pipeline method."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Full Pipeline")
    print("=" * 80)

    # Generate sample data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Initialize classifier
    clf = SimpleMulticlassClassifier(
        name="full_pipeline_classifier",
        config_path="examples/config_example.yaml",
        mlflow_tracking_uri="file:./mlruns",
        mlflow_experiment_name="full_pipeline_experiment",
    )

    # Run full pipeline
    results = clf.full_pipeline(
        X,
        y,
        save_visualizations=True,
        output_dir="outputs/example4",
        log_to_mlflow=True,
        mlflow_run_name="full_pipeline_run",
    )

    print("\n" + "=" * 40)
    print("Pipeline Results:")
    print("=" * 40)
    print(f"Train Metrics: {results['train_metrics']}")
    print(f"Test Metrics: {results['test_metrics']}")
    if "mlflow_run_id" in results:
        print(f"MLflow Run ID: {results['mlflow_run_id']}")
    if "visualizations" in results:
        print(f"Visualizations: {results['visualizations']}")


def example_with_dataframe():
    """Example using pandas DataFrame with label column."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Using DataFrame with Label Column")
    print("=" * 80)

    # Generate sample data and create DataFrame
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Create DataFrame with label column
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["label"] = y

    # Initialize classifier
    clf = SimpleMulticlassClassifier(name="dataframe_classifier")

    # Load data (extract labels from DataFrame)
    X_df = df.drop(columns=["label"])
    y_df = df["label"]
    clf.load_data(X_df, y_df)

    # Build and train
    clf.build_model()
    clf.train()

    # Evaluate
    results = clf.evaluate()
    print("\nTrain Metrics:", results["train_metrics"])
    print("Test Metrics:", results["test_metrics"])


def example_different_models():
    """Example with different model types."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Different Model Types (Sklearn + Neural Network)")
    print("=" * 80)

    # Generate sample data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    model_types = [
        "logistic_regression",
        "random_forest",
        "svm",
        "neural_network",
    ]

    for model_type in model_types:
        print(f"\n--- Training {model_type} ---")

        # Create config dict
        if model_type == "neural_network":
            params = {
                "hidden_dims": [128, 64],
                "activation": "relu",
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32,
            }
        else:
            params = (
                {"max_iter": 1000}
                if model_type == "logistic_regression"
                else (
                    {"n_estimators": 50}
                    if model_type == "random_forest"
                    else {"kernel": "rbf"}
                )
            )

        config = {
            "model": {
                "type": model_type,
                "params": params,
            }
        }

        # Initialize classifier with inline config
        clf = SimpleMulticlassClassifier(name=f"{model_type}_classifier")
        clf.config = config

        # Load data
        clf.load_data(X, y)

        # Build and train
        clf.build_model()
        clf.train()

        # Evaluate
        results = clf.evaluate()
        print(f"  Train Accuracy: {results['train_metrics']['accuracy']:.4f}")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}")


def example_neural_network():
    """Example with Neural Network classifier."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Neural Network Classifier")
    print("=" * 80)

    # Generate sample data
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Neural Network config with custom hidden layers
    config = {
        "model": {
            "type": "neural_network",
            "params": {
                "hidden_dims": [128, 64, 32],  # 3 hidden layers
                "activation": "relu",
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 16,
            },
        }
    }

    # Initialize classifier
    clf = SimpleMulticlassClassifier(
        name="neural_network_classifier",
    )
    clf.config = config

    # Load data
    clf.load_data(X, y)

    # Build and train
    print("\nBuilding Neural Network with hidden dims: [128, 64, 32]")
    clf.build_model()
    print("Training Neural Network...")
    clf.train()

    # Evaluate
    results = clf.evaluate()
    print("\n" + "=" * 40)
    print("Neural Network Results:")
    print("=" * 40)
    print(f"Train Accuracy: {results['train_metrics']['accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Test Recall: {results['test_metrics']['recall']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")

    # Save visualizations
    viz_paths = clf.save_visualizations("outputs/example_nn")
    print(f"\nVisualizations saved to: {viz_paths}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_with_config_file()
    # example_with_mlflow()  # Uncomment to test MLflow
    # example_full_pipeline()  # Uncomment to test full pipeline
    example_with_dataframe()
    example_different_models()
    example_neural_network()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
