# ------------------------------
# Libraries
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ------------------------------
# Page configuration & theming
# ------------------------------
st.set_page_config(
    page_title="Greenhouse Energy and Climate Dashboard",
    page_icon="🌿",
    layout="wide",
)

# ------------------------------
# Load data and ML model
# ------------------------------
APP_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    df_path = APP_DIR / "greenhouse_data.csv"
    df = pd.read_csv(df_path, parse_dates=["Timestamp"])
    return df

@st.cache_resource
def load_model():
    model_path = APP_DIR / "rand_forest.pkl"
    model = joblib.load(model_path)
    return model

df = load_data()
model = load_model()

# features used in the Random Forest model
feature_cols = [
    "PPFD_GH", "Temperature_GH", "Humidity", "CO2", "Dew Point",
    "VPD_GH", "Pressure_GH",
    "PPFD_amb", "Temperature_amb", "Pressure_amb", "VPD_amb",
]

location_col = "Location_GH"

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.title("Plotting Panel")

min_date = df["Timestamp"].dt.date.min()
max_date = df["Timestamp"].dt.date.max()

selected_date = st.sidebar.date_input(
    "Select Date (MM/DD/YYYY)",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
)

raw_locations = sorted(df[location_col].unique())
display_locations = [loc.replace("West Cool Wall", "West") for loc in raw_locations]
loc_map = dict(zip(display_locations, raw_locations))

chosen_display_loc = st.sidebar.selectbox(
    "Select Greenhouse",
    options=display_locations,
)

view_button = st.sidebar.button("Generate Plots")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: Pick a date and greenhouse, then click **Generate Plots** to explore "
    "the full day's climate and energy profile."
)

# ------------------------------
# Main title
# ------------------------------
st.markdown(
    "<h1 style='text-align: center;'>🌿 Greenhouse Energy and Climate Dashboard</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center; color: #6b7280; font-size: 0.98rem;'>"
    "A gentle overview of your greenhouse environment and energy use, day by day."
    "</p>",
    unsafe_allow_html=True,
)

st.write(
    "Welcome to this interactive dashboard for your greenhouses. "
    "Choose a date and greenhouse in the sidebar (see icon at top left), "
    "then click 'Generate Plots' and voila- you'll see climate and energy "
    "consumption patterns for that day right before your eyes!"
)

# ------------------------------
# Daily plots
# ------------------------------

st.subheader("Daily Climate & Energy Patterns")

st.caption(
    "Below you can visualize how the greenhouse and ambient conditions evolve "
    "over the selected day, along with the associated energy consumption."
)

if view_button:
    raw_loc = loc_map[chosen_display_loc]

    mask = (
        (df["Timestamp"].dt.date == selected_date)
        & (df[location_col] == raw_loc)
    )
    day_df = df.loc[mask].sort_values("Timestamp")

    if day_df.empty:
        st.warning("No data for this date and greenhouse.")
    else:
        st.subheader(
            f"Daily time series for {chosen_display_loc} Greenhouse "
            f"on {selected_date.strftime('%m/%d/%Y')}"
        )

        st.caption("Hover over any plot to see exact values at each timestamp.")
        
        # first row
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Greenhouse Temperature (°C)")
            st.line_chart(day_df.set_index("Timestamp")["Temperature_GH"])
        with c2:
            st.caption("Greenhouse PPFD (μmol/m2/s)")
            st.line_chart(day_df.set_index("Timestamp")["PPFD_GH"])

        # second row
        c3, c4 = st.columns(2)
        with c3:
            st.caption("Greenhouse VPD (kPa)")
            st.line_chart(day_df.set_index("Timestamp")["VPD_GH"])
        with c4:
            st.caption("Greenhouse Pressure (kPa)")
            st.line_chart(day_df.set_index("Timestamp")["Pressure_GH"])

        # third row
        c5, c6 = st.columns(2)
        with c5:
            st.caption("Greenhouse CO₂ (ppm)")
            st.line_chart(day_df.set_index("Timestamp")["CO2"])
        with c6:
            st.caption("Energy Consumption (kWh)")
            st.line_chart(day_df.set_index("Timestamp")["Energy"])

        # fourth row
        c7, c8 = st.columns(2)
        with c7:
            st.caption("Ambient Temperature (°C)")
            st.line_chart(day_df.set_index("Timestamp")["Temperature_amb"])
        with c8:
            st.caption("Ambient PPFD (μmol/m2/s)")
            st.line_chart(day_df.set_index("Timestamp")["PPFD_amb"])

# ------------------------------
# Energy Consumption Prediction
# ------------------------------
st.subheader("Prediction of Energy Consumption")

st.markdown(
    "<p style='text-align: center'>"
    "<i>Hello, O' Wanderer! I, the great Voldemort, welcome you to the Random Forest. "
    "This forest can help you predict the future. What! You want to predict your "
    "electricity bill for greenhouses? What a shame wasting this forests power "
    "... So be it! "
    "Enter your data and within moments you will get a predicted hourly energy "
    "consumption data for your greenhouse! Now deal with this snake - "
    "<b>SERPENSORTIA</b> [throws a snake at you].</i>"
    "</p>",
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        ppfd_gh = st.number_input("PPFD in greenhouse (PPFD_GH)", value=0.0)
        temp_gh = st.number_input("Greenhouse temperature (°C)", value=20.0)
        humidity = st.number_input("Humidity (%)", value=90.0)
        co2 = st.number_input("CO₂ (ppm)", value=600.0)
        dew = st.number_input("Dew point (°C)", value=18.0)
        vpd_gh = st.number_input("Greenhouse VPD (kPa)", value=0.5)

    with col_right:
        pressure_gh = st.number_input("Greenhouse pressure (kPa)", value=100.0)
        ppfd_amb = st.number_input("Ambient PPFD (PPFD_amb)", value=0.0)
        temp_amb = st.number_input("Ambient temperature (°C)", value=20.0)
        pressure_amb = st.number_input("Ambient pressure (kPa)", value=100.0)
        vpd_amb = st.number_input("Ambient VPD (kPa)", value=0.5)

    submit_pred = st.form_submit_button("Generate Prediction")

    if submit_pred:
        input_row = {
            "PPFD_GH": ppfd_gh,
            "Temperature_GH": temp_gh,
            "Humidity": humidity,
            "CO2": co2,
            "Dew Point": dew,
            "VPD_GH": vpd_gh,
            "Pressure_GH": pressure_gh,
            "PPFD_amb": ppfd_amb,
            "Temperature_amb": temp_amb,
            "Pressure_amb": pressure_amb,
            "VPD_amb": vpd_amb,
        }

        input_df = pd.DataFrame([input_row])[feature_cols]
        pred_energy = model.predict(input_df)[0]

        st.success(f"Predicted energy consumption: {pred_energy:.3f} kWh")

# ------------------------------
# Footer / copyright
# ------------------------------
st.markdown(
    """
    <hr style="margin-top: 3rem; margin-bottom: 0.5rem;">
    <div style="text-align: center; color: #9ca3af; font-size: 0.8rem;">
        © 2025 Nibir Kanti Roy
    </div>
    """,
    unsafe_allow_html=True,
)