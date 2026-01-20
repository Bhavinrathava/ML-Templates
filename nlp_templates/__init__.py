"""
NLP Templates - Production-ready ML classification templates.

This package provides ready-to-use machine learning classification templates
with support for hierarchical classification, neural networks, text classification,
and MLflow integration.

Example usage (feature-based classification):
    >>> from nlp_templates import SimpleMulticlassClassifier
    >>> clf = SimpleMulticlassClassifier(name="my_classifier")
    >>> clf.load_data(X, y)
    >>> clf.build_model()
    >>> clf.train()
    >>> results = clf.evaluate()

For text classification (text -> embeddings -> labels):
    >>> from nlp_templates import TextClassifier
    >>> clf = TextClassifier(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    >>> clf.fit(texts=["great product", "bad service"], labels=[1, 0])
    >>> predictions = clf.predict(["excellent quality"])

For hierarchical classification:
    >>> from nlp_templates import HierarchicalTextClassifier
    >>> clf = HierarchicalTextClassifier()
    >>> clf.fit(texts, labels)  # labels: (n_samples, n_hierarchy_levels)
    >>> predictions = clf.predict(new_texts)
"""

__version__ = "0.1.0"
__author__ = "Bhavinkumar Rathava"
__email__ = "bhavinr9@gmail.com"

from nlp_templates.classifiers import (
    SimpleMulticlassClassifier,
    HierarchicalNNClassifier,
    TextClassifier,
    HierarchicalTextClassifier,
)
from nlp_templates.evaluation import (
    calculate_metrics,
    get_confusion_matrix,
    get_classification_report,
    plot_confusion_matrix,
    plot_metrics_comparison,
)
from nlp_templates.preprocessing import ConfigLoader

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Classifiers
    "SimpleMulticlassClassifier",
    "HierarchicalNNClassifier",
    "TextClassifier",
    "HierarchicalTextClassifier",
    # Evaluation
    "calculate_metrics",
    "get_confusion_matrix",
    "get_classification_report",
    "plot_confusion_matrix",
    "plot_metrics_comparison",
    # Preprocessing
    "ConfigLoader",
]
