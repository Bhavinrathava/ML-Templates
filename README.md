# NLP Templates

[![PyPI version](https://badge.fury.io/py/nlp-templates.svg)](https://badge.fury.io/py/nlp-templates)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready ML classification templates with text classification, hierarchical classification, and neural network support.

## Features

- **TextClassifier**: End-to-end text classification (Text -> Embeddings -> Labels)
- **HierarchicalTextClassifier**: Multi-level hierarchical text classification
- **SimpleMulticlassClassifier**: Feature-based multi-class classification with multiple backends (logistic regression, random forest, SVM, neural network)
- **HierarchicalNNClassifier**: Feature-based hierarchical classification with neural networks
- **MLflow Integration**: Built-in experiment tracking and model logging
- **Configurable**: YAML/JSON configuration support for hyperparameters
- **Evaluation Tools**: Metrics calculation, confusion matrices, and visualizations

## Installation

### Basic Installation

```bash
pip install nlp-templates
```

### With Text Classification Support (embeddings)

```bash
pip install nlp-templates[embeddings]
```

### With PyTorch Support (for neural network classifiers)

```bash
pip install nlp-templates[torch]
```

### With MLflow Support

```bash
pip install nlp-templates[mlflow]
```

### Full Installation (all optional dependencies)

```bash
pip install nlp-templates[all]
```

## Quick Start

### Text Classification (Recommended for NLP)

The easiest way to classify text - handles embedding generation automatically:

```python
from nlp_templates import TextClassifier

# Training data
texts = [
    "This product is amazing!", "Excellent quality!",
    "Terrible, waste of money.", "Very disappointed."
]
labels = [1, 1, 0, 0]  # 1=positive, 0=negative

# Initialize and train
clf = TextClassifier(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
clf.fit(texts, labels)

# Predict
new_texts = ["Best purchase ever!", "Broke after one day."]
predictions = clf.predict(new_texts)
print(predictions)  # [1, 0]
```

### Hierarchical Text Classification

For multi-level category prediction:

```python
from nlp_templates import HierarchicalTextClassifier
import numpy as np

texts = [
    "iPhone with great camera", "Android phone 5G",
    "Gaming laptop RTX", "Business ultrabook",
    "Cotton t-shirt summer", "Formal dress shirt",
]

# Hierarchical labels: [Category, Subcategory]
# Category: 0=Electronics, 1=Clothing
# Subcategory: 0=Phones, 1=Laptops (for Electronics), 0=Shirts (for Clothing)
labels = np.array([
    [0, 0], [0, 0],  # Electronics - Phones
    [0, 1], [0, 1],  # Electronics - Laptops
    [1, 0], [1, 0],  # Clothing - Shirts
])

clf = HierarchicalTextClassifier()
clf.fit(texts, labels)

predictions = clf.predict(["New smartphone release"])
print(predictions)  # [[0, 0]] -> Electronics, Phones
```

### Feature-based Classification

```python
import numpy as np
from nlp_templates import SimpleMulticlassClassifier

# Create sample data
X = np.random.randn(500, 20)
y = np.random.choice([0, 1, 2, 3], size=500)

# Initialize classifier
clf = SimpleMulticlassClassifier(
    name="my_classifier",
    random_state=42,
    test_size=0.3,
)

# Configure for neural network
clf.config = {
    "model": {
        "type": "neural_network",
        "params": {
            "hidden_dims": [64, 32],
            "epochs": 50,
            "batch_size": 32,
        }
    }
}

# Train and evaluate
clf.load_data(X, y)
clf.build_model()
clf.train()
results = clf.evaluate()

print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
```

### Hierarchical Classification

```python
import numpy as np
from nlp_templates import HierarchicalNNClassifier

# Create hierarchical data (2-level hierarchy)
X = np.random.randn(500, 20)
y_level0 = np.random.choice([0, 1, 2], size=500)
y_level1 = np.random.choice([0, 1, 2, 3], size=500)
y = np.column_stack([y_level0, y_level1])

# Initialize hierarchical classifier
clf = HierarchicalNNClassifier(
    name="hierarchical_clf",
    random_state=42,
    test_size=0.3,
)

# Train and evaluate
clf.load_data(X, y)
clf.build_model()
clf.train()
results = clf.evaluate()

# Get predictions
predictions = clf.predict(X[:10])
print(f"Predictions shape: {predictions.shape}")  # (10, 2) for 2 levels
```

### Using Configuration Files

```python
from nlp_templates import SimpleMulticlassClassifier

# Load from YAML config
clf = SimpleMulticlassClassifier(
    name="configured_classifier",
    config_path="config/model_config.yaml",
)

# Run full pipeline
results = clf.full_pipeline(
    X, y,
    save_visualizations=True,
    output_dir="outputs",
)
```

Example config file (`config/model_config.yaml`):

```yaml
model:
  type: neural_network
  params:
    hidden_dims: [128, 64, 32]
    activation: relu
    dropout_rate: 0.3
    learning_rate: 0.001
    epochs: 100
    batch_size: 64
```

## Available Model Types

For `SimpleMulticlassClassifier`:

| Model Type | Description |
|------------|-------------|
| `logistic_regression` | Sklearn LogisticRegression |
| `random_forest` | Sklearn RandomForestClassifier |
| `naive_bayes` | Sklearn MultinomialNB |
| `svm` | Sklearn SVC |
| `neural_network` | PyTorch-based neural network |

## API Reference

### TextClassifier (Text -> Embeddings -> Labels)

```python
TextClassifier(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    classifier_type="simple",   # "simple" or "hierarchical"
    name="text_classifier",
    random_state=42,
    test_size=0.2,
    batch_size=32,              # Batch size for embedding generation
    device=None,                # "cuda", "cpu", or None for auto
)
```

**Methods:**
- `fit(texts, labels)`: Train classifier on text data
- `predict(texts)`: Predict labels for texts
- `predict_proba(texts)`: Get prediction probabilities
- `evaluate()`: Evaluate on test set
- `get_embeddings(texts)`: Get embeddings without classification

### HierarchicalTextClassifier

Same as `TextClassifier` but defaults to hierarchical classification. Accepts labels of shape `(n_samples, n_hierarchy_levels)`.

### SimpleMulticlassClassifier (Feature-based)

```python
SimpleMulticlassClassifier(
    name="classifier_name",
    config_path=None,           # Path to YAML/JSON config
    random_state=42,
    test_size=0.3,
    mlflow_tracking_uri=None,   # MLflow tracking URI
    mlflow_experiment_name=None,
)
```

**Methods:**
- `load_data(X, y)`: Load and split data
- `build_model()`: Build model from config
- `train()`: Train the model
- `evaluate()`: Evaluate on train/test sets
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get prediction probabilities
- `full_pipeline(X, y, ...)`: Run complete pipeline

### HierarchicalNNClassifier (Feature-based)

```python
HierarchicalNNClassifier(
    name="hierarchical_classifier",
    config_path=None,
    random_state=42,
    test_size=0.3,
    mlflow_tracking_uri=None,
    mlflow_experiment_name=None,
)
```

**Methods:**
- `load_data(X, y)`: Load hierarchical data (y shape: n_samples x n_levels)
- `build_model()`: Build classifier for each hierarchy level
- `train()`: Train all level classifiers
- `evaluate()`: Evaluate each level separately
- `predict(X)`: Predict all hierarchy levels
- `predict_proba(X)`: Get probabilities for each level
- `full_pipeline(X, y, ...)`: Run complete pipeline

## Requirements

**Core dependencies:**
- Python >= 3.9
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- pyyaml >= 6.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

**Optional dependencies:**
- sentence-transformers >= 2.2.0 (for text classification with embeddings)
- torch >= 2.0.0 (for neural network classifiers)
- mlflow >= 2.0.0 (for experiment tracking)
- optuna >= 3.0.0 (for hyperparameter tuning)

## Development

```bash
# Clone the repository
git clone https://github.com/bhavinr9/nlp-templates.git
cd nlp-templates

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=nlp_templates --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request