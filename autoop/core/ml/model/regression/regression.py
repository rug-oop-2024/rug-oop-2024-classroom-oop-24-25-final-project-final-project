import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso as SklearnLasso

class MultipleLinearRegression(Model):

    def __init__(self, **kwargs):
        super().__init__(name="Multiple Linear Regression", model_type="regression", **kwargs)
        self.parameters = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):

        x_matrix = np.c_[np.ones((observations.shape[0], 1)), observations]
        self.parameters = (
            np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ ground_truth
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        x_matrix = np.c_[np.ones((observations.shape[0], 1)), observations]
        return x_matrix @ self.parameters



class Lasso(Model):
    
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(name="Lasso Regression", model_type="regression", **kwargs)
        self.model = SklearnLasso(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = {
            "coefficients": self.model.coef_,
            "intercept": self.model.intercept_
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class PolynomialRegression(Model):

    def __init__(self, degree=2, **kwargs):
        super().__init__(name="Polynomial Regression", model_type="regression", **kwargs)
        self.degree = degree
        self.parameters = None

    def _polynomial_features(self, X):
        X_poly = X
        for d in range(2, self.degree + 1):
            X_poly = np.c_[X_poly, X ** d]
        return X_poly

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_poly = self._polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        self.parameters = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = self._polynomial_features(X)
        X_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        return X_b.dot(self.parameters)


if __name__ == "__main__":

    X_train_reg = np.array([[1], [2], [3], [4]])
    y_train_reg = np.array([1.5, 3.0, 4.5, 6.0])

    X_test_reg = np.array([[5], [6]])

    mlr_model = MultipleLinearRegression(asset_path="some/path", version="1.0.0")
    mlr_model.fit(X_train_reg, y_train_reg)
    mlr_predictions = mlr_model.predict(X_test_reg)
    print(f"MLR Predictions: {mlr_predictions}")

    lasso_model = Lasso(alpha=0.1, asset_path="models/lasso_model", version="1.0.0")
    lasso_model.fit(X_train_reg, y_train_reg)
    lasso_predictions = lasso_model.predict(X_test_reg)
    print(f"Lasso Predictions: {lasso_predictions}")

    pol_model = PolynomialRegression(asset_path="models/lasso_model", version="1.0.0")
    pol_model.fit(X_train_reg, y_train_reg)
    polynomial_predictions = pol_model.predict(X_test_reg)
    print(f"Polynomial Predictions: {polynomial_predictions}")

