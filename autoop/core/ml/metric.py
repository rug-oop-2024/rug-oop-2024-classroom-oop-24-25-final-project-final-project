from abc import ABC, abstractmethod
# TODO: Unsure what to use typing for
from typing import Any
import numpy as np

METRICS = [
    # Regression
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared",
    # Classification
    "accuracy",
    "f1",
    "log_loss"
]


def get_metric(name: str) -> "Metric":
    """
    Return metric instance given by name.

    Parameters
    ----------
    name : str
        Metric name.

    Returns
    -------
    any
        Metric instance matching name.

    Raises
    ------
    ValueError
        If the name matches no instance or is incorrectly spelt.
    NotImplementedError
        If the name matches the instance but is not implemented.
    """
    if name in METRICS:
        # Parse string to match the class name.
        class_name = ''.join(word.capitalize() for word in name.split('_'))
        try:
            # Return class
            return globals()[class_name]()
        except TypeError:
            raise NotImplementedError(f"{class_name} is not implemented.")
    else:
        raise ValueError(f"{class_name} is possibly mispelt.")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        """
        Calculate metric.

        Parameters
        ----------
        truth : ndarray
            Grouth truth values.
        pred : ndarray
            Predicted values.

        Returns
        -------
        float
            Real number representing the metric value.

        Raises
        ------
        To be implemented.
        """
        ...


# region Regression
class MeanSquaredError(Metric):
    """Mean Squared Error metric implementation for regression."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        return np.mean((truth - pred) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric implementation for regression."""
    ...


class RSquared(Metric):
    """R-Squared metric implementation for regression."""
    ...
# endregion


# region Classification
class Accuracy(Metric):
    """Accuracy metric implementation for classification."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        return np.mean(truth == pred)


class F1(Metric):
    """F1 metric implementation for classification."""
    ...


class LogLoss(Metric):
    """Logarithmic Loss implemenation for classication."""
    ...
# endregion
