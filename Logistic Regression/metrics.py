import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    return sum(y_true == y_pred) / len(y_true)
