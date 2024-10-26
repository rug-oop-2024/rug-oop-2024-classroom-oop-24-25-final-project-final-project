from xgboost import XGBRegressor as WrappedXGBRegressor
from typing import Literal
from copy import deepcopy

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class XGBRegressor(Model):
    """ XGBoost for regression wrapper """
    def __init__(self,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 gamma: float = 0.0,
                 ) -> None:
        """
        Initialize the XGBoost model with various hyperparameters,
        as defined in the scikit-learn library.
        :param max_depth: Maximum depth
        :param learning_rate: Learning rate
        :param n_estimators: Number of estimators
        :param gamma: Minimum loss reduction
        We did not like how XGboost handles error messages, so we
        decided to reimplement checking for parameter values.
        """
        max_depth, learning_rate, n_estimators, gamma = \
            self.validate_parameters(max_depth, learning_rate, n_estimators, 
                                     gamma)
        self._model = WrappedXGBRegressor(max_depth=max_depth,
                                          learning_rate=learning_rate,
                                          n_estimators=n_estimators,
                                          gamma=gamma)
        super().__init__(type="regression")

    def validate_parameters(max_depth: int,
                            learning_rate: float,
                            n_estimators: int,
                            gamma: float) -> list:
        """
        Replaces every wrong parameter value with its default
        value while informing the user.
        """
        pass  # start with sth like if learning rate < 0, print("Bad"), learning rate = 0.1

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the xgboost method .fit
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "booster": self._model.get_booster(),
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the xgboost method .predict
        """
        return self._model.predict(observations)

    @property
    def model(self) -> 'XGBRegressor':
        """ Returns a copy of model to prevent leakage. """
        return deepcopy(self._model)

