from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []

    # Use the read() method to get the pandas DataFrame from the Dataset
    df: pd.DataFrame = dataset.read()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_type = 'numerical'
        else:
            feature_type = 'categorical'

        # Create a Feature object for this column and add it to the list
        feature = Feature(name=column, type=feature_type)
        features.append(feature)

    return features
