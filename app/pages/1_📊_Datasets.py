import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

datasets_name = pd.DataFrame({"Datasets": [art.name for art in datasets]}
                             )

st.set_page_config(page_title="Datasets"
                   )

st.title("Dataset")

st.dataframe(datasets_name,
             column_config={"Datasets": "Name datasets"},
             hide_index=True
             )
