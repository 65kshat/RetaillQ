import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import calendar
import streamlit as st
import pandas as pd
from datetime import date
from components.sidebar import apply_theme, render_sidebar
from components.charts import probability_bar
from utils.data_loader import load_store, parse_store_id, show_training_date_notice
from utils.model_loader import load_closure_model


st.set_page_config(page_title="Store Closure · RetailIQ", page_icon="🔒", layout="wide")
apply_theme()
render_sidebar()

st.markdown("# 🔒 Store Closure Prediction")
st.markdown("Predict whether a store will be **Open** or **Closed** on a given date using **DecisionTreeClassifier**.")
st.warning(
    "⚠️ This model learns from historical operating patterns and scheduled closures. "
    "It is reliable for predictable closures (e.g. Sundays, public holidays) but cannot "
    "predict unexpected or emergency closures."
)
st.divider()


def build_features(store_id, target_date, promo, state_holiday, school_holiday, store_row):
    """Construct the feature DataFrame expected by the closure model."""
    dow = target_date.isoweekday()
    year, month, day = target_date.year, target_date.month, target_date.day
    week_of_year = int(target_date.isocalendar()[1])
    day_of_year = target_date.timetuple().tm_yday
    is_weekend = 1 if dow in [6, 7] else 0
    is_month_start = 1 if day == 1 else 0
    _, last_day = calendar.monthrange(year, month)
    is_month_end = 1 if day == last_day else 0
    is_sh = 1 if str(state_holiday).lower() != "0" else 0

    promo2 = int(store_row.get("Promo2", 0))
    promo2_year = store_row.get("Promo2SinceYear", 0)
    promo_interval = store_row.get("PromoInterval", "None")
    if pd.isna(promo_interval): promo_interval = "None"

    month_name = target_date.strftime("%b")
    is_promo2 = 0
    if promo2 == 1 and not pd.isna(promo2_year) and year >= int(promo2_year):
        if month_name in str(promo_interval).split(","):
            is_promo2 = 1

    comp_dist = store_row.get("CompetitionDistance", 2320.0)
    if pd.isna(comp_dist): comp_dist = 2320.0
    comp_m = store_row.get("CompetitionOpenSinceMonth", 0)
    comp_y = store_row.get("CompetitionOpenSinceYear", 0)
    if pd.isna(comp_m): comp_m = 0
    if pd.isna(comp_y): comp_y = 0

    if int(comp_y) > 0 and int(comp_m) > 0:
        comp_months = max(0, (year - int(comp_y)) * 12 + (month - int(comp_m)))
        has_comp = 1
    else:
        comp_months = 0
        has_comp = 0

    return pd.DataFrame([{
        "DayOfWeek": dow, "Year": year, "Month": month, "Day": day,
        "WeekOfYear": week_of_year, "DayOfYear": day_of_year,
        "IsWeekend": is_weekend, "IsMonthStart": is_month_start, "IsMonthEnd": is_month_end,
        "Promo": int(promo), "Promo2": promo2, "IsPromo2Active": is_promo2,
        "CompetitionDistance": float(comp_dist), "CompetitionOpenMonths": comp_months,
        "HasCompetitionDate": has_comp, "SchoolHoliday": int(school_holiday),
        "IsStateHoliday": is_sh,
        "StoreType": str(store_row.get("StoreType", "a")),
        "Assortment": str(store_row.get("Assortment", "a")),
        "StateHoliday": str(state_holiday).lower(),
        "PromoInterval": str(promo_interval),
        "Store": int(store_id),
    }])


try:
    store_df = load_store()
    model = load_closure_model()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

col_inp, col_out = st.columns([1, 1])

with col_inp:
    st.markdown("### Input Parameters")
    store_input = st.text_input(
        "Store ID",
        value="ST-0001",
        placeholder="ST-0001",
        help="Enter Store ID (e.g. ST-0042)"
    )
    store_id, store_err = parse_store_id(store_input, store_df)
    if store_err:
        st.error(store_err)

    target_date = st.date_input(
        "Date",
        value=date.today(),
        min_value=date(1970, 1, 1),
        max_value=date(2099, 12, 31),
        help="Select any calendar date"
    )
    show_training_date_notice(target_date)

    promo = st.selectbox("Promotion Planned?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    state_holiday = st.selectbox(
        "State Holiday",
        ["0", "a", "b", "c"],
        format_func=lambda x: {"0": "None", "a": "Public Holiday", "b": "Easter", "c": "Christmas"}.get(x, x)
    )
    school_holiday = st.selectbox("School Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    predict_btn = st.button("🔮 Predict Status", type="primary", width="stretch")

with col_out:
    st.markdown("### Prediction Result")
    if predict_btn:
        if store_err or store_id is None:
            st.error(store_err or "Please enter a valid Store ID before predicting.")
        elif target_date is None:
            st.error("Please select a valid date.")
        else:
            try:
                store_row = store_df[store_df["Store"] == store_id].iloc[0].to_dict()
                features = build_features(store_id, target_date, promo, state_holiday, school_holiday, store_row)
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                classes = model.classes_

                is_open = int(prediction) == 1
                status_label = "🟢 OPEN" if is_open else "🔴 CLOSED"
                color = "#22c55e" if is_open else "#ef4444"

                st.markdown(f"""
<div style="background:#1e293b;border:2px solid {color};border-radius:10px;padding:28px;text-align:center">
  <div style="color:#94a3b8;font-size:14px">Predicted Store Status</div>
  <div style="color:{color};font-size:48px;font-weight:700;margin:12px 0">{status_label}</div>
  <div style="color:#94a3b8;font-size:13px">Store ST-{store_id:04d} (ID: {store_id}) · {target_date} · {target_date.strftime('%A')}</div>
</div>
""", unsafe_allow_html=True)

                # Map numeric classes to readable labels
                class_labels = ["Closed" if int(c) == 0 else "Open" for c in classes]
                st.plotly_chart(
                    probability_bar(class_labels, list(proba), "Prediction Confidence")
                )

                store_row_data = store_df[store_df["Store"] == store_id].iloc[0]
                st.markdown("#### Input Summary")
                st.markdown(f"""
| Feature | Value |
|---------|-------|
| Store | ST-{store_id:04d} (ID: {store_id}) |
| Date | {target_date} |
| Day of Week | {target_date.strftime('%A')} |
| Promotion | {'Yes' if promo else 'No'} |
| State Holiday | {state_holiday if state_holiday != '0' else 'None'} |
| School Holiday | {'Yes' if school_holiday else 'No'} |
| Store Type | {store_row_data['StoreType']} |
""")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Select parameters and click **Predict Status**.")

st.divider()
st.markdown("""
> **Model behaviour:** The Decision Tree primarily captures consistent historical patterns —
> stores that reliably close on Sundays, public holidays, or during specific periods.
> It **cannot** predict unexpected closures due to operational decisions or events outside
> the training data.
""")
