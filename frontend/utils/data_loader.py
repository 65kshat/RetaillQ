import re
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path

TRAIN_START_DATE = date(2013, 1, 1)
TRAIN_END_DATE = date(2015, 7, 31)


def parse_store_id(store_input: str, store_df=None):
    """
    Validates and converts a Store ID string (e.g. 'ST-0042', 'st-0042')
    into an integer store ID (e.g. 42).
    Returns (store_id, error_message).
    If valid: (int, None)
    If invalid: (None, error_str)
    """
    if not store_input or not str(store_input).strip():
        return None, "Please enter a Store ID (e.g. ST-0001 to ST-1115)."

    cleaned = str(store_input).strip().upper()
    match = re.match(r"^ST-(\d{4})$", cleaned)
    if not match:
        return None, f"Invalid format '{store_input.strip()}'. Store ID must follow the ST-0001 to ST-1115 format."

    num = int(match.group(1))
    if num < 1 or num > 1115:
        return None, f"Store ID '{cleaned}' is out of range. Must be between ST-0001 and ST-1115."

    if store_df is not None and num not in store_df["Store"].values:
        return None, f"Store ID '{cleaned}' does not exist in dataset."

    return num, None


def show_training_date_notice(target_date):
    """Displays a warning if selected date is outside historical training range (Jan 2013 – Jul 2015)."""
    if target_date is not None:
        if target_date < TRAIN_START_DATE or target_date > TRAIN_END_DATE:
            st.warning(
                f"⚠️ **Out of Training Range**: {target_date.strftime('%B %d, %Y')} is outside the "
                f"historical training dataset (Jan 2013 – Jul 2015). Predictions reflect model "
                f"generalization and may have lower reliability."
            )


def find_dataset_dir():
    """Locate the dataset directory regardless of launch location."""
    cwd = Path.cwd()
    for candidate in [cwd / "dataset", cwd.parent / "dataset"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find dataset directory.")


@st.cache_data
def load_train():
    path = find_dataset_dir() / "train.csv"
    return pd.read_csv(path, parse_dates=["Date"], low_memory=False)


@st.cache_data
def load_store():
    path = find_dataset_dir() / "store.csv"
    return pd.read_csv(path)


@st.cache_data
def load_test():
    path = find_dataset_dir() / "test.csv"
    return pd.read_csv(path, parse_dates=["Date"], low_memory=False)


@st.cache_data
def load_merged_train():
    """Train data merged with store info."""
    train = load_train()
    store = load_store()
    return train.merge(store, on="Store", how="left")
