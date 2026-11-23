import sys
from pathlib import Path

# Add frontend directory to path so pages can import components/utils
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from components.sidebar import apply_theme, render_sidebar

st.set_page_config(
    page_title="RetailIQ",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_sidebar()

# Hero
st.markdown("# 🏪 RetailIQ")
st.markdown("### Retail Business Intelligence & Machine Learning Platform")
st.markdown("*Powered by the Rossmann Retail Dataset — 1,115 stores across Germany*")
st.divider()

# Problem statement
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("## About This Project")
    st.markdown("""
RetailIQ is an end-to-end analytics platform designed to help retail managers make
data-driven decisions. Using over **one million historical sales records** from Rossmann
stores, the platform applies machine learning to forecast sales, predict customer traffic,
classify store performance, segment stores, and anticipate store closures.

The core challenge: retail performance is shaped by a complex mix of promotions, holidays,
competition proximity, and store characteristics. RetailIQ brings all these signals together
in a single, interactive dashboard.
""")

with col2:
    st.markdown("## Dataset")
    st.markdown("""
| Attribute | Value |
|-----------|-------|
| Stores | 1,115 |
| Training Records | ~1,017,209 |
| Date Range | Jan 2013 – Jul 2015 |
| Source | Rossmann (Kaggle) |
""")

st.divider()

# ML Components cards
st.markdown("## Machine Learning Components")
st.markdown("Five models power the RetailIQ predictions:")

cols = st.columns(5)
cards = [
    ("📈", "Sales Forecasting", "HistGradientBoosting Regressor", "Predict daily revenue for any store and date."),
    ("👥", "Customer Footfall", "Random Forest Regressor", "Estimate customer traffic before the day begins."),
    ("🏆", "Sales Performance", "HistGradientBoosting Classifier", "Classify a day as Low, Medium, or High sales."),
    ("🔵", "Store Segmentation", "K-Means Clustering", "Group stores into 3 strategic business clusters."),
    ("🔒", "Store Closure", "Decision Tree Classifier", "Predict whether a store will be open or closed."),
]

for col, (icon, title, model, desc) in zip(cols, cards):
    with col:
        st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:18px;height:200px;">
  <div style="font-size:28px">{icon}</div>
  <div style="font-weight:700;color:#e2e8f0;margin-top:8px">{title}</div>
  <div style="font-size:11px;color:#3b82f6;margin:4px 0">{model}</div>
  <div style="font-size:13px;color:#94a3b8;margin-top:6px">{desc}</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Objectives
st.markdown("## Project Objectives")
objectives = [
    "Forecast daily sales for any Rossmann store up to any future date.",
    "Predict customer footfall without requiring actual sales data.",
    "Classify store-day combinations into performance tiers for rapid prioritisation.",
    "Segment all 1,115 stores into strategic clusters for targeted management.",
    "Predict scheduled store closures based on calendar and store features.",
    "Provide an interactive BI dashboard for data exploration and decision support.",
]
for i, obj in enumerate(objectives, 1):
    st.markdown(f"**{i}.** {obj}")

st.divider()

st.markdown("## How to Navigate")
st.info("Use the **sidebar** on the left to switch between pages. Each page is self-contained and independently demonstrates one aspect of the RetailIQ platform.")

st.markdown("""
| Page | Purpose |
|------|---------|
| 📊 Dataset Explorer | Explore and visualise the raw Rossmann data |
| 📈 Sales Forecasting | Generate a sales prediction for a chosen store & date |
| 👥 Customer Footfall | Estimate daily customer count for a store |
| 🏆 Sales Performance | Classify a day's expected performance tier |
| 🔵 Store Segmentation | Browse cluster assignments and cluster characteristics |
| 🔒 Store Closure | Predict whether a store will be open or closed |
""")
