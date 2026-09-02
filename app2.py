import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "polar_energy_dataset.csv"

data = pd.read_csv(csv_path)
# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="AI Polar Energy Optimizer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI-Based Polar Energy Management System")
st.write(
    "Predict energy demand and intelligently decide whether to use "
    "renewable energy, battery power, or backup generator."
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("polar_energy_dataset.csv")

data = load_data()

features = [
    "temperature_c",
    "wind_speed_mps",
    "solar_radiation_wm2",
    "battery_level_percent",
    "hour",
    "solar_power_kwh",
    "wind_power_kwh"
]

X = data[features]
y = data["energy_demand_kwh"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions

model, X_test, y_test, predictions = train_model(X, y)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Absolute Error", round(mae, 2))

with col2:
    st.metric("R² Score", round(r2, 2))

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🔮 Predict Energy Demand")

col1, col2 = st.columns(2)

with col1:

    temperature = st.slider(
        "Temperature (°C)",
        float(data["temperature_c"].min()),
        float(data["temperature_c"].max()),
        float(data["temperature_c"].mean())
    )

    wind_speed = st.slider(
        "Wind Speed (m/s)",
        float(data["wind_speed_mps"].min()),
        float(data["wind_speed_mps"].max()),
        float(data["wind_speed_mps"].mean())
    )

    solar_radiation = st.slider(
        "Solar Radiation (W/m²)",
        float(data["solar_radiation_wm2"].min()),
        float(data["solar_radiation_wm2"].max()),
        float(data["solar_radiation_wm2"].mean())
    )

    battery_level = st.slider(
        "Battery Level (%)",
        0.0,
        100.0,
        60.0
    )

with col2:

    hour = st.slider(
        "Hour",
        0,
        23,
        12
    )

    solar_power = st.slider(
        "Solar Power (kWh)",
        float(data["solar_power_kwh"].min()),
        float(data["solar_power_kwh"].max()),
        float(data["solar_power_kwh"].mean())
    )

    wind_power = st.slider(
        "Wind Power (kWh)",
        float(data["wind_power_kwh"].min()),
        float(data["wind_power_kwh"].max()),
        float(data["wind_power_kwh"].mean())
    )

# -----------------------------
# ENERGY OPTIMIZATION FUNCTION
# -----------------------------
def optimize_energy(demand, solar_power, wind_power, battery_level):

    renewable_energy = solar_power + wind_power

    if renewable_energy >= demand:

        return {
            "source": "Solar + Wind",
            "battery_action": "Charge",
            "generator": "OFF",
            "message":
            "Renewable energy is sufficient. Extra energy can be stored in the battery."
        }

    elif battery_level > 30:

        return {
            "source": "Renewable + Battery",
            "battery_action": "Discharge",
            "generator": "OFF",
            "message":
            "Use renewable energy first and cover remaining demand using the battery."
        }

    else:

        return {
            "source": "Renewable + Generator",
            "battery_action": "Preserve",
            "generator": "ON",
            "message":
            "Battery level is low. Activate the backup generator."
        }


# -----------------------------
# PREDICTION
# -----------------------------
if st.button("⚡ Predict & Optimize", use_container_width=True):

    input_data = pd.DataFrame([{
        "temperature_c": temperature,
        "wind_speed_mps": wind_speed,
        "solar_radiation_wm2": solar_radiation,
        "battery_level_percent": battery_level,
        "hour": hour,
        "solar_power_kwh": solar_power,
        "wind_power_kwh": wind_power
    }])

    predicted_demand = model.predict(input_data)[0]

    st.success(
        f"Predicted Energy Demand: {predicted_demand:.2f} kWh"
    )

    result = optimize_energy(
        predicted_demand,
        solar_power,
        wind_power,
        battery_level
    )

    st.subheader("🤖 AI Energy Decision")

    c1, c2, c3 = st.columns(3)

    c1.metric("Energy Source", result["source"])
    c2.metric("Battery Action", result["battery_action"])
    c3.metric("Generator", result["generator"])

    st.info(result["message"])

    # Energy comparison graph
    renewable = solar_power + wind_power

    energy_values = pd.DataFrame({
        "Energy Type": [
            "Predicted Demand",
            "Solar Power",
            "Wind Power",
            "Total Renewable"
        ],
        "Energy (kWh)": [
            predicted_demand,
            solar_power,
            wind_power,
            renewable
        ]
    })

    st.subheader("⚡ Current Energy Comparison")

    st.bar_chart(
        energy_values.set_index("Energy Type")
    )


# -----------------------------
# GRAPHS
# -----------------------------
st.divider()

st.header("📈 Data Analysis")

# GRAPH 1
st.subheader("1. Actual vs Predicted Energy Demand")


actual_predicted = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

st.scatter_chart(
    actual_predicted,
    x="Actual",
    y="Predicted"
)


# GRAPH 2
st.subheader("2. Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)


# GRAPH 3
st.subheader("3. Average Energy Demand by Hour")

hourly_demand = (
    data.groupby("hour")["energy_demand_kwh"]
    .mean()
)

st.line_chart(hourly_demand)


# GRAPH 4
st.subheader("4. Solar and Wind Power")

power_data = data[
    ["solar_power_kwh", "wind_power_kwh"]
].head(100)

st.line_chart(power_data)


# -----------------------------
# DATASET
# -----------------------------
with st.expander("📂 View Dataset"):
    st.dataframe(data)
