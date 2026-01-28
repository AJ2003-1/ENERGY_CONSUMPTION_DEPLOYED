import streamlit as st
import numpy as np
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="centered"
)

# ==================================================
# LOAD MODEL (PIPELINE)
# ==================================================
try:
    model = joblib.load(r"C:\Users\aaron\OneDrive\Desktop\VS_Code\ENERGY_CONSUMPTION\best_energy_model (1).pkl")
except FileNotFoundError:
    st.error("❌ 'best_energy_model.pkl' not found. Please place it in the app folder.")
    st.stop()

# ==================================================
# TITLE
# ==================================================
st.title("⚡ Energy Consumption Prediction (kW)")
st.markdown(
    "This application predicts **household energy consumption (kW)** "
    "using a trained **machine learning regression model**."
)
st.divider()

# ==================================================
# INPUT SECTION
# ==================================================
st.subheader("🔢 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    global_reactive_power = st.number_input(
        "Global Reactive Power (kW)",
        min_value=0.000,
        max_value=2.000,
        value=0.440,
        step=0.001,
        format="%.3f"
    )

    voltage = st.number_input(
        "Voltage (V)",
        min_value=200.000,
        max_value=260.000,
        value=240.000,
        step=0.001,
        format="%.3f"
    )

    global_intensity = st.number_input(
        "Global Intensity (A)",
        min_value=0.000,
        max_value=60.000,
        value=10.000,
        step=0.001,
        format="%.3f"
    )

    sub_metering_1 = st.number_input(
        "Sub Metering 1 – Kitchen (Wh)",
        min_value=0.000,
        max_value=50.000,
        value=1.000,
        step=0.001,
        format="%.3f"
    )

    sub_metering_2 = st.number_input(
        "Sub Metering 2 – Laundry (Wh)",
        min_value=0.000,
        max_value=50.000,
        value=1.000,
        step=0.001,
        format="%.3f"
    )

with col2:
    sub_metering_3 = st.number_input(
        "Sub Metering 3 – Heating / AC (Wh)",
        min_value=0.000,
        max_value=100.000,
        value=10.000,
        step=0.001,
        format="%.3f"
    )

    hour = st.slider("Hour of Day", 0, 23, 12)
    day = st.slider("Day of Month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    dayofweek = st.slider("Day of Week (0 = Mon, 6 = Sun)", 0, 6, 2)
    is_weekend = st.selectbox("Is Weekend?", [0, 1])

# ==================================================
# PREDICTION
# ==================================================
st.divider()

if st.button("🔮 Predict Energy Consumption"):

    input_data = np.array([[
        global_reactive_power,
        voltage,
        global_intensity,
        sub_metering_1,
        sub_metering_2,
        sub_metering_3,
        hour,
        day,
        month,
        dayofweek,
        is_weekend
    ]], dtype=float)

    # Predict (log scale)
    pred_log = model.predict(input_data)[0]

    # Convert back to kW
    pred_kw = np.expm1(pred_log)

    # ==================================================
    # OUTPUT
    # ==================================================
    st.subheader("📊 Input Summary")

    st.table({
        "Feature": [
            "Global Reactive Power (kW)",
            "Voltage (V)",
            "Global Intensity (A)",
            "Sub Metering 1 (Wh)",
            "Sub Metering 2 (Wh)",
            "Sub Metering 3 (Wh)",
            "Hour",
            "Day",
            "Month",
            "Day of Week",
            "Is Weekend"
        ],
        "Value": [
            f"{global_reactive_power:.3f}",
            f"{voltage:.3f}",
            f"{global_intensity:.3f}",
            f"{sub_metering_1:.3f}",
            f"{sub_metering_2:.3f}",
            f"{sub_metering_3:.3f}",
            hour,
            day,
            month,
            dayofweek,
            is_weekend
        ]
    })

    st.success(f"⚡ Predicted Energy Consumption: **{pred_kw:.3f} kW**")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("📌 Machine Learning Project | Energy Consumption Prediction")