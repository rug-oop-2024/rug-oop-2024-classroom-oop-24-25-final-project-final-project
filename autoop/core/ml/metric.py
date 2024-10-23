from abc import ABC, abstractmethod
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np

#  get_metrics implemented at the end of the file (!)


class Metric(ABC):
    """Base class for all metrics. All metrics must implement calculate()"""
    _name = None
    _description = None
    _task_type = None

    def __str__(self) -> str:
        """ String representation of any Metric """
        return f"{self._name}: {self._description}" + \
               f", specified for {self._task_type}"

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the metric """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Allows for calling the metric.
        Calls ValueErrors if y_true and y_pred are not numpy arrays
        of the same size.
        """
        if not isinstance(y_true, np.ndarray) or not isinstance(
                                                    y_pred, np.ndarray):
            raise ValueError("both y_true and y_pred must be numpy arrays")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same size")
        return self.calculate(y_true, y_pred)

    @property
    def task_type(self):
        """ Getter for the task type """
        return self._task_type


class MSE(Metric):
    """ Implements the Mean Squared Error metric"""
    _name = "Mean Squared Error"
    _description = "Calculates the average of all" + \
        " the squared differences between the ground truth and the prediction"
    _task_type = "regression"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the mean squared error """
        return np.mean((y_true - y_pred) ** 2)


class RMSE(Metric):
    """ Implements the Root Mean Squared Error metric"""
    _name = "Root Mean Squared Error"
    _description = "Calculates the root of the average of all" + \
        " the squared differences between the ground truth and the prediction"
    _task_type = "regression"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the root mean squared error """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class R2(Metric):
    """ Implements the R^2 metric """
    _name = "R^2"
    _description = "Calculates the R^2 score, representing the" + \
        " goodness of fit of a linear model to the data"
    _task_type = "regression"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the R2 score """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot


class Accuracy(Metric):
    """ Implements the Accuracy metric """
    _name = "Accuracy"
    _description = "Calculates the percentage of correct predictions"
    _task_type = "classification"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculates the mean accuracy """
        return np.mean(y_true == y_pred)


class Recall(Metric):
    """ Implements the Recall metric """
    _name = "Recall"
    _description = "Calculates the percentage of correct positive" + \
        " predictions when compared to all true positive labels"
    _task_type = "classification"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ SKlearn wrapper for the recall score """
        return recall_score(y_true, y_pred)


class Precision(Metric):
    """ Implements the Precision metric """
    _name = "Precision"
    _description = "Calculates the percentage of correct positive" + \
        " predictions when compared to all predicted positive labels"
    _task_type = "classification"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ SKlearn wrapper for the precision score """
        return precision_score(y_true, y_pred)


class F1(Metric):
    """ Implements the F1-score metric """
    _name = "F1"
    _description = "Calculates the F1 score, representing the" + \
        " harmonic mean of precision and recall"
    _task_type = "classification"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ SKlearn wrapper for the f1 score """
        return f1_score(y_true, y_pred)


METRICS = {
    "MSE": MSE,
    "RMSE": RMSE,
    "R2": R2,
    "Accuracy": Accuracy,
    "Recall": Recall,
    "Precision": Precision,
    "F1": F1,
}  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> Metric:
    # Factory function to get a metric by name.
    if name not in METRICS.keys():
        print(f"Metric {name} not implemented.")
        return None
    # create a class instance of the same name and return it
    Metric = METRICS[name]()
    print("Initializing metric- ", Metric)
    print("!!!Note that this metric should only be used for",
          f"{Metric.task_type} tasks!!!")
    return Metric


if __name__ == "__main__":
    metric = get_metric("F1")
    fake_metric = get_metric("fake")
