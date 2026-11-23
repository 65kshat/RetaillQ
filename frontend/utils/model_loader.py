import joblib
import streamlit as st
from pathlib import Path


def find_models_dir():
    """Locate the models directory regardless of launch location."""
    cwd = Path.cwd()
    for candidate in [cwd / "models", cwd.parent / "models"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find models directory.")


@st.cache_resource
def load_sales_model():
    path = find_models_dir() / "model_1_sales_forecasting" / "sales_forecasting_model.pkl"
    return joblib.load(path)


@st.cache_resource
def load_footfall_model():
    path = find_models_dir() / "model_2_customer_footfall" / "customer_footfall_model.pkl"
    return joblib.load(path)


@st.cache_resource
def load_classification_model():
    path = find_models_dir() / "model_3_sales_classification" / "sales_classification_model.pkl"
    return joblib.load(path)


@st.cache_resource
def load_segmentation_model():
    mdir = find_models_dir() / "model_4_store_segmentation"
    kmeans = joblib.load(mdir / "store_segmentation_model.pkl")
    preprocessor = joblib.load(mdir / "store_segmentation_preprocessor.pkl")
    return kmeans, preprocessor


@st.cache_resource
def load_closure_model():
    path = find_models_dir() / "model_5_store_closure" / "store_closure_model.pkl"
    return joblib.load(path)


@st.cache_data
def load_segmentation_results():
    path = find_models_dir() / "model_4_store_segmentation" / "store_segmentation_results.csv"
    return pd.read_csv(path)


import pandas as pd  # noqa: E402 — needed for load_segmentation_results
