import streamlit as st


def metric_card(label, value, delta=None, prefix="", suffix=""):
    """Display a styled metric."""
    display_val = f"{prefix}{value}{suffix}"
    if delta is not None:
        st.metric(label, display_val, delta=delta)
    else:
        st.metric(label, display_val)


def kpi_row(metrics: list[dict]):
    """
    Render a row of KPI metrics.
    Each dict: {label, value, delta (optional), prefix (optional), suffix (optional)}
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            metric_card(
                m["label"],
                m["value"],
                delta=m.get("delta"),
                prefix=m.get("prefix", ""),
                suffix=m.get("suffix", ""),
            )
