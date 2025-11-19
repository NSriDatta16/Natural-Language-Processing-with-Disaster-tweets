from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Compute basic classification metrics + confusion matrix.

    Returns:
        accuracy, precision, recall, f1, conf_matrix
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return acc, precision, recall, f1, cm


def print_metrics(acc, precision, recall, f1, cm) -> None:
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("\nConfusion Matrix:")
    print(cm)
