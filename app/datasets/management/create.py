from autoop.core.ml.dataset import Dataset
import pandas as pd
from typing import IO


def create(file: IO) -> Dataset:
    dataset = Dataset(data=pd.read_csv(file).to_csv(index=False).encode(),
                      name=file.name)
    return dataset
