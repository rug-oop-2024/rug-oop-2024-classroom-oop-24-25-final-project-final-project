import numpy as np
from pydantic import PrivateAttr
from sklearn.linear_model import Lasso as WrappedLasso
from copy import deepcopy

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class Lasso(Model):

    _model: WrappedLasso = PrivateAttr(default_factory=WrappedLasso)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the Lasso method .fit
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "_coef": self._model.coef_,
            "_intercept": self._model.intercept_
        }  # Splitting the vector into weights and bias

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the Lasso method .predict
        """
        return self._model.predict(observations)

    @property
    def model(self) -> 'Lasso':
        """ Returns a copy of model to prevent leakage. """
        return deepcopy(self._model)
