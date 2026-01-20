"""
Evaluation metrics and visualization utilities.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        average (str): Averaging method ('weighted', 'macro', 'micro', 'binary')

    Returns:
        dict: Dictionary with metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        np.ndarray: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, output_dict: bool = False
) -> str | Dict:
    """
    Get detailed classification report.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        output_dict (bool): If True, return as dictionary

    Returns:
        str or dict: Classification report
    """
    return classification_report(y_true, y_pred, output_dict=output_dict)
