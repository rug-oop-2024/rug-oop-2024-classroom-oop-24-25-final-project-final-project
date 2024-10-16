
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """Feature.
        name (str): name of feature.
        type (str): type of feature.
    """
    # attributes here
    name: str = Field()
    type: str = Field()

    def __str__(self):
        raise NotImplementedError("To be implemented.")
