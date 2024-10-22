import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from app.datasets.management.create import create
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
dataset_names = [_.name for _ in datasets]

# your code here

st.set_page_config(page_title="Datasets")

st.title("Dataset")

new_dataset = st.file_uploader(label="Upload dataset(csv)",
                               accept_multiple_files=False,
                               type=["csv"])

if new_dataset is not None:
    st.write("shit💩")
    create()
else:
    view_dataset = st.selectbox("select dataset", dataset_names)

    if view_dataset is not None:
        st.dataframe(
            pd.read_csv(datasets[dataset_names.index(view_dataset)].asset_path)
            )
