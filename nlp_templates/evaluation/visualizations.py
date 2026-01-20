"""
Visualization utilities for model evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """
    Create and save confusion matrix visualization.

    Args:
        cm (np.ndarray): Confusion matrix from sklearn
        class_names (list, optional): List of class names
        output_path (str, optional): Path to save figure
        figsize (tuple): Figure size

    Returns:
        tuple: (figure, axes) or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(
            "matplotlib and seaborn required for visualization. "
            "Install with: pip install matplotlib seaborn"
        )
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")

    return fig, ax


def plot_metrics_comparison(
    train_metrics: dict,
    test_metrics: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """
    Create and save metrics comparison visualization.

    Args:
        train_metrics (dict): Training metrics
        test_metrics (dict): Test metrics
        output_path (str, optional): Path to save figure
        figsize (tuple): Figure size

    Returns:
        tuple: (figure, axes) or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib required for visualization. Install with: pip install matplotlib"
        )
        return None

    metrics_names = list(train_metrics.keys())
    train_values = [train_metrics[m] for m in metrics_names]
    test_values = [test_metrics[m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, train_values, width, label="Train")
    ax.bar(x + width / 2, test_values, width, label="Test")

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Training vs Test Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")

    return fig, ax
