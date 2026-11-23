
import streamlit as st


CSS = """
<style>
[data-testid="stSidebar"] {
    background-color: #0d1b2a;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e6ed;
}

[data-testid="stAppViewContainer"] {
    background-color: #111827;
}

[data-testid="metric-container"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px;
}

.model-item {
    background-color: #172638;
    border: 1px solid #263b50;
    border-radius: 7px;
    padding: 9px 10px;
    margin-bottom: 7px;
}

.model-name {
    color: #e0e6ed;
    font-size: 13px;
    font-weight: 600;
}

.model-metric {
    color: #60a5fa;
    font-size: 12px;
    margin-top: 3px;
}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def render_sidebar(current_page=""):
    with st.sidebar:
        st.markdown("## 🏪 RetailIQ")
        st.caption("Retail Business Intelligence")
        st.divider()

        # Navigation
        st.markdown("**Navigation**")
        st.page_link("app.py", label="🏠 Home")
        st.page_link("pages/1_Dataset_Explorer.py", label="📊 Dataset Explorer")
        st.page_link("pages/2_Sales_Forecasting.py", label="📈 Sales Forecasting")
        st.page_link("pages/3_Customer_Footfall.py", label="👥 Customer Footfall")
        st.page_link("pages/4_Sales_Performance.py", label="🏆 Sales Performance")
        st.page_link("pages/5_Store_Segmentation.py", label="🔵 Store Segmentation")
        st.page_link("pages/6_Store_Closure.py", label="🔒 Store Closure")

        st.divider()

        # Model performance
        st.markdown("**🤖 Model Performance**")

        models = [
            ("Sales Forecasting", "R² Score: 61.66%"),
            ("Customer Footfall", "R² Score: 73.42%"),
            ("Sales Performance", "Accuracy: 72.28%"),
            ("Store Segmentation", "3 Clusters · Silhouette: 0.3144"),
            ("Store Closure", "Accuracy: 99.80%"),
        ]

        for name, metric in models:
            st.markdown(
                f"""
                <div class="model-item">
                    <div class="model-name">{name}</div>
                    <div class="model-metric">{metric}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()
        st.caption("Rossmann Retail Dataset · 1,115 Stores")
