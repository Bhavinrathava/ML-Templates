"""Classifier implementations."""

from nlp_templates.classifiers.simple_multiclass_classifier import (
    SimpleMulticlassClassifier,
)
from nlp_templates.classifiers.hierarchical_multiclass_classifier import (
    HierarchicalNNClassifier,
)

__all__ = ["SimpleMulticlassClassifier", "HierarchicalNNClassifier"]
