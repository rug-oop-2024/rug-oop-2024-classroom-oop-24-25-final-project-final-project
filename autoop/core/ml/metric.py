from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# List of supported metric names
METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "mean_absolute_error",
    "r2"
]

def get_metric(name: str):
    """Factory function to get a metric by name."""
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "f1_score":
        return F1Score()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r2":
        return R2()
    else:
        raise ValueError(f"Metric {name} is not supported. Available metrics: {METRICS}")

class Metric(ABC):

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

# Concrete Metric Implementations

class MeanSquaredError(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

class Accuracy(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        correct_predictions = np.sum(y_true == y_pred)
        accuracy = correct_predictions / len(y_true)
        return accuracy

class Precision(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        if predicted_positives == 0:
            return 0.0
        precision = true_positives / predicted_positives
        return precision

class Recall(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        if actual_positives == 0:
            return 0.0
        recall = true_positives / actual_positives
        return recall

class F1Score(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = Precision()(y_true, y_pred)
        recall = Recall()(y_true, y_pred)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

class MeanAbsoluteError(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

class R2(Metric):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        explained_variance = np.sum((y_true - y_pred) ** 2)
        if total_variance == 0:
            return 0.0
        r2 = 1 - (explained_variance / total_variance)
        return r2


y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

mse_metric = get_metric("mean_squared_error")
mae_metric = get_metric("mean_absolute_error")
r2_metric = get_metric("r2")

print(f"MSE: {mse_metric(y_true_reg, y_pred_reg)}")
print(f"MAE: {mae_metric(y_true_reg, y_pred_reg)}")
print(f"R2: {r2_metric(y_true_reg, y_pred_reg)}")


y_true_class = np.array([1, 0, 1, 1])
y_pred_class = np.array([1, 0, 0, 1])

accuracy_metric = get_metric("accuracy")
precision_metric = get_metric("precision")
recall_metric = get_metric("recall")
f1_metric = get_metric("f1_score")

print(f"Accuracy: {accuracy_metric(y_true_class, y_pred_class)}")
print(f"Precision: {precision_metric(y_true_class, y_pred_class)}")
print(f"Recall: {recall_metric(y_true_class, y_pred_class)}")
print(f"F1 Score: {f1_metric(y_true_class, y_pred_class)}")
