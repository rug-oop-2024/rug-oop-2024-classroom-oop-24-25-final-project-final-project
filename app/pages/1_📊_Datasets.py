import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from app.datasets.management.create import create
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

datasets_display_dat = pd.DataFrame({"Datasets": [art.name for
                                                  art in datasets],
                                     "versions": [dataset.version for
                                                  dataset in datasets]}
                                    )

st.set_page_config(page_title="Datasets"
                   )

st.title("Dataset")

new_dataset = st.file_uploader(label="Upload dataset(csv)",
                               accept_multiple_files=False,
                               type=["csv"])

if new_dataset is not None:
    st.write("shit💩")

st.dataframe(datasets_display_dat,
             column_config={"Datasets": "Name datasets", "versions": "version"},
             hide_index=True
             )
