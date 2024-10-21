from abc import ABC, abstractmethod
# TODO: Unsure what to use typing for
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
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
        except KeyError:
            raise NotImplementedError(f"{name} is not implemented.")
    else:
        raise ValueError(f"{name} is mispelt.")


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


# TODO: Move to extensions
class MeanSquaredError(Metric):
    """Mean Squared Error metric implementation."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        return np.mean((truth - pred) ** 2)


class Accuracy(Metric):
    """Accuracy metric implementation."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        return np.mean(truth == pred)
