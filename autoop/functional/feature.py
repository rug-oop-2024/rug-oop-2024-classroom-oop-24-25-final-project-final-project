from typing import List, Literal
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


class Feature:
    """A class used to represent features of a dataset."""

    def __init__(self, name: str, type: Literal["categorical", "numerical"]):
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """Getter for the name of the feature."""
        return self._name

    @property
    def type(self) -> Literal["categorical", "numerical"]:
        """Getter for the type of the feature."""
        return self._type

    def __str__(self) -> str:
        """ String representation of the object """
        return f"Feature name={self.name}, type={self.type}"

    @staticmethod
    def detect_feature_types(dataset: Dataset) -> List[Feature]:
        """
        Detects feature type of a dataset.
        Bases on the assumption that all features are either
        categorical or numerical, and no empty features are present.
        :param dataset: Dataset object, storing the data
        :return: List of features
        """
        ds = dataset.read()
        features = []
        for name in ds.columns:
            if ds[name].dtype == "object":
                features.append(Feature(name=name, type="categorical"))
            else:
                features.append(Feature(name=name, type="numerical"))
        return features
