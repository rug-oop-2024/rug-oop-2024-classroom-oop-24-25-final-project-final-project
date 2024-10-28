from autoop.core.ml.artifact import Artifact
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.sidebar.success("Select a page above.")
st.markdown(open("README.md", encoding="utf-8").read())
