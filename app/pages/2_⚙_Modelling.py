import streamlit as st
import pandas as pd

from pathlib import Path

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here
path = st.selectbox("Select a dataset", datasets)

if path:
    data_path = Path(path)

    dataset = Dataset(data_path)