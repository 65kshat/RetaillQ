import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from components.sidebar import apply_theme, render_sidebar
from components.charts import scatter_chart, bar_chart, DARK_TEMPLATE
from utils.model_loader import load_segmentation_results

st.set_page_config(page_title="Store Segmentation · RetailIQ", page_icon="🔵", layout="wide")
apply_theme()
render_sidebar()

st.markdown("# 🔵 Store Segmentation")
st.markdown("K-Means clustering groups all 1,115 stores into **3 strategic clusters** based on sales behaviour, customer traffic, and promotion performance.")
st.divider()

try:
    seg = load_segmentation_results()
except Exception as e:
    st.error(f"Failed to load segmentation results: {e}")
    st.stop()

seg["Cluster"] = seg["Cluster"].astype(str)

CLUSTER_LABELS = {
    "0": "Cluster 0 — Steady Performers",
    "1": "Cluster 1 — Moderate Performers",
    "2": "Cluster 2 — High Performers",
}
CLUSTER_DESC = {
    "0": "Large group of stores with consistent, moderate sales. Reliable revenue, average promotion response.",
    "1": "Smaller group with moderate sales and customers. May benefit from targeted promotions.",
    "2": "High-volume stores with significantly above-average sales and customer traffic. The top tier.",
}

# --- Cluster Overview ---
st.markdown("## Cluster Overview")
cluster_counts = seg["Cluster"].value_counts().sort_index()
c1, c2, c3 = st.columns(3)
for col, (cluster, count) in zip([c1, c2, c3], cluster_counts.items()):
    desc = CLUSTER_DESC.get(str(cluster), "")
    avg_sales = seg[seg["Cluster"] == str(cluster)]["AverageSales"].mean()
    color = {"0": "#3b82f6", "1": "#14b8a6", "2": "#22c55e"}.get(str(cluster), "#fff")
    with col:
        st.markdown(f"""
<div style="background:#1e293b;border:1px solid {color};border-radius:10px;padding:18px">
  <div style="color:{color};font-size:22px;font-weight:700">{CLUSTER_LABELS.get(str(cluster), f'Cluster {cluster}')}</div>
  <div style="color:#e2e8f0;font-size:28px;font-weight:700;margin:8px 0">{count} stores</div>
  <div style="color:#94a3b8;font-size:13px">Avg Sales: €{avg_sales:,.0f}/day</div>
  <div style="color:#94a3b8;font-size:12px;margin-top:8px">{desc}</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Scatter Plot ---
st.markdown("## Store Distribution by Cluster")
fig_scatter = scatter_chart(
    seg, x="AverageCustomers", y="AverageSales",
    color_col="Cluster", title="Average Customers vs Average Sales by Cluster",
    x_title="Average Daily Customers", y_title="Average Daily Sales (€)"
)
st.plotly_chart(fig_scatter)

# --- Comparison Charts ---
st.markdown("## Cluster Characteristics")
col1, col2 = st.columns(2)

with col1:
    cluster_summary = seg.groupby("Cluster")[["AverageSales", "AverageCustomers", "PromoSalesLift"]].mean().reset_index()
    fig_sales = px.bar(
        cluster_summary, x="Cluster", y="AverageSales",
        title="Average Daily Sales by Cluster",
        template=DARK_TEMPLATE,
        color="Cluster",
        color_discrete_sequence=["#3b82f6", "#14b8a6", "#22c55e"],
    )
    fig_sales.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", showlegend=False)
    st.plotly_chart(fig_sales)

with col2:
    fig_promo = px.bar(
        cluster_summary, x="Cluster", y="PromoSalesLift",
        title="Average Promotion Sales Lift by Cluster",
        template=DARK_TEMPLATE,
        color="Cluster",
        color_discrete_sequence=["#3b82f6", "#14b8a6", "#22c55e"],
    )
    fig_promo.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", showlegend=False)
    st.plotly_chart(fig_promo)

# --- Competition Distance ---
col3, col4 = st.columns(2)
with col3:
    fig_comp = px.box(
        seg, x="Cluster", y="CompetitionDistance",
        title="Competition Distance Distribution by Cluster",
        template=DARK_TEMPLATE,
        color="Cluster",
        color_discrete_sequence=["#3b82f6", "#14b8a6", "#22c55e"],
    )
    fig_comp.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", showlegend=False)
    st.plotly_chart(fig_comp)

with col4:
    fig_cust = px.bar(
        cluster_summary, x="Cluster", y="AverageCustomers",
        title="Average Daily Customers by Cluster",
        template=DARK_TEMPLATE,
        color="Cluster",
        color_discrete_sequence=["#3b82f6", "#14b8a6", "#22c55e"],
    )
    fig_cust.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", showlegend=False)
    st.plotly_chart(fig_cust)

st.divider()

# --- Store Type Mix per Cluster ---
st.markdown("## Store Type Mix by Cluster")
if "StoreType" in seg.columns:
    type_mix = seg.groupby(["Cluster", "StoreType"]).size().reset_index(name="Count")
    fig_mix = px.bar(
        type_mix, x="Cluster", y="Count", color="StoreType",
        title="Store Type Distribution per Cluster",
        template=DARK_TEMPLATE, barmode="stack"
    )
    fig_mix.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b")
    st.plotly_chart(fig_mix)

# --- Store-level table ---
st.divider()
st.markdown("## Store-Level Cluster Assignments")
selected_cluster = st.selectbox("Filter by Cluster", ["All", "0", "1", "2"])
display_df = seg if selected_cluster == "All" else seg[seg["Cluster"] == selected_cluster]

cols_to_show = ["Store", "Cluster", "AverageSales", "AverageCustomers", "PromoSalesLift", "CompetitionDistance", "StoreType", "Assortment"]
cols_to_show = [c for c in cols_to_show if c in display_df.columns]
st.dataframe(
    display_df[cols_to_show].sort_values("AverageSales", ascending=False).reset_index(drop=True)
)
