"""Evaluation utilities."""

from nlp_templates.evaluation.metrics import (
    calculate_metrics,
    get_confusion_matrix,
    get_classification_report,
)
from nlp_templates.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_metrics_comparison,
)

__all__ = [
    "calculate_metrics",
    "get_confusion_matrix",
    "get_classification_report",
    "plot_confusion_matrix",
    "plot_metrics_comparison",
]
