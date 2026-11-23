import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from components.sidebar import apply_theme, render_sidebar
from components.charts import line_chart, bar_chart, scatter_chart
from utils.data_loader import load_train, load_store, load_test

st.set_page_config(page_title="Dataset Explorer · RetailIQ", page_icon="📊", layout="wide")
apply_theme()
render_sidebar()

st.markdown("# 📊 Dataset Explorer")
st.markdown("Explore the Rossmann retail dataset — raw data, trends and store characteristics.")
st.divider()

try:
    train = load_train()
    store = load_store()
    test = load_test()
    merged = train.merge(store, on="Store", how="left")
except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}")
    st.stop()

# --- Overview ---
st.markdown("## Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training Records", f"{len(train):,}")
c2.metric("Test Records", f"{len(test):,}")
c3.metric("Stores", f"{store['Store'].nunique():,}")
c4.metric("Date Range", f"{train['Date'].min().date()} → {train['Date'].max().date()}")

tab1, tab2, tab3 = st.tabs(["📋 Column Info", "⚠️ Missing Values", "🏪 Store Info"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Train / Test Columns**")
        info = pd.DataFrame({"Column": train.columns, "Type": [str(t) for t in train.dtypes]})
        st.dataframe(info, width="stretch", hide_index=True)
    with col_b:
        st.markdown("**Store Columns**")
        sinfo = pd.DataFrame({"Column": store.columns, "Type": [str(t) for t in store.dtypes]})
        st.dataframe(sinfo, width="stretch", hide_index=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Train Missing Values**")
        mv = train.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing"]
        mv = mv[mv["Missing"] > 0]
        if mv.empty:
            st.success("No missing values in training data.")
        else:
            st.dataframe(mv, width="stretch", hide_index=True)
    with col_b:
        st.markdown("**Store Missing Values**")
        smv = store.isnull().sum().reset_index()
        smv.columns = ["Column", "Missing"]
        smv = smv[smv["Missing"] > 0]
        if smv.empty:
            st.success("No missing values in store data.")
        else:
            st.dataframe(smv, width="stretch", hide_index=True)

with tab3:
    st.markdown("**Store Type Distribution**")
    st_dist = store["StoreType"].value_counts().reset_index()
    st_dist.columns = ["StoreType", "Count"]
    st.dataframe(st_dist, width="stretch", hide_index=True)
    st.markdown("**Assortment Distribution**")
    as_dist = store["Assortment"].value_counts().reset_index()
    as_dist.columns = ["Assortment", "Count"]
    st.dataframe(as_dist, width="stretch", hide_index=True)

st.divider()

# --- Sales Trends ---
st.markdown("## Historical Sales Trends")
open_df = merged[merged["Open"] == 1].copy()
monthly = open_df.groupby(open_df["Date"].dt.to_period("M"))["Sales"].sum().reset_index()
monthly["Date"] = monthly["Date"].dt.to_timestamp()
st.plotly_chart(line_chart(monthly, "Date", "Sales", "Monthly Total Sales (All Stores)", y_title="Total Sales (€)"))

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Average Sales by Day of Week")
    dow_sales = open_df.groupby("DayOfWeek")["Sales"].mean().reset_index()
    dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    dow_sales["Day"] = dow_sales["DayOfWeek"].map(dow_names)
    st.plotly_chart(bar_chart(dow_sales, "Day", "Sales", "Avg Sales by Day of Week", y_title="Avg Sales (€)"))

with col2:
    st.markdown("### Average Sales by Store Type")
    stype_sales = open_df.groupby("StoreType")["Sales"].mean().reset_index()
    st.plotly_chart(bar_chart(stype_sales, "StoreType", "Sales", "Avg Sales by Store Type", color="#14b8a6", y_title="Avg Sales (€)"))

# --- Customer Trends ---
st.divider()
st.markdown("## Customer Trends")
monthly_cust = open_df.groupby(open_df["Date"].dt.to_period("M"))["Customers"].sum().reset_index()
monthly_cust["Date"] = monthly_cust["Date"].dt.to_timestamp()
st.plotly_chart(line_chart(monthly_cust, "Date", "Customers", "Monthly Total Customers (All Stores)", color="#14b8a6", y_title="Total Customers"))

# --- Promotions ---
st.divider()
st.markdown("## Promotion Insights")
promo_df = open_df.groupby("Promo")["Sales"].mean().reset_index()
promo_df["Promo"] = promo_df["Promo"].map({0: "No Promo", 1: "Promo Active"})
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(bar_chart(promo_df, "Promo", "Sales", "Avg Sales: Promo vs No Promo", y_title="Avg Sales (€)"))
with col2:
    promo_pct = open_df["Promo"].mean() * 100
    st.metric("Days with Promotion", f"{promo_pct:.1f}%")
    lift = (open_df[open_df["Promo"] == 1]["Sales"].mean() / open_df[open_df["Promo"] == 0]["Sales"].mean() - 1) * 100
    st.metric("Promotion Sales Lift", f"+{lift:.1f}%")
    st.info("Promotional days drive significantly higher sales across all store types.")

# --- Raw Data Sample ---
st.divider()
st.markdown("## Sample Data")
with st.expander("View raw training data (first 100 rows)"):
    st.dataframe(train.head(100), width="stretch")
