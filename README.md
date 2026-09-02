# ⚡ AI-Based Polar Energy Management System

An AI-powered energy management system designed for extreme polar environments.  
The system predicts energy demand using Machine Learning and intelligently selects between renewable energy, battery storage, and backup generator power.

## 🌍 Problem Statement

Polar research stations operate in extreme environmental conditions where reliable energy management is critical.

Energy demand varies due to factors such as:

- Temperature
- Wind speed
- Solar radiation
- Time of day
- Battery level
- Solar power generation
- Wind power generation

Poor energy management can increase fuel consumption, operational costs, and dependence on backup generators.

## 💡 Proposed Solution

The **AI-Based Polar Energy Management System** uses Machine Learning to predict energy demand and an optimization mechanism to determine the most suitable energy source.

The system follows the workflow:

Environmental Data  
↓  
Machine Learning Model  
↓  
Energy Demand Prediction  
↓  
Renewable Energy Availability  
↓  
Energy Optimization  
↓  
Battery / Renewable / Generator Decision  
↓  
Interactive Streamlit Dashboard

## 🤖 Machine Learning Model

The project uses a **Random Forest Regressor** for energy demand prediction.

### Input Features

- Temperature (°C)
- Wind Speed (m/s)
- Solar Radiation (W/m²)
- Battery Level (%)
- Hour of Day
- Solar Power (kWh)
- Wind Power (kWh)

### Target Variable

`energy_demand_kwh`

The model predicts the expected energy demand based on environmental and energy-system conditions.

## ⚡ Energy Optimization

After predicting energy demand, the system determines the appropriate energy strategy.

### Scenario 1: Renewable Energy is Sufficient

Solar + Wind → Supply Demand  
Extra Energy → Charge Battery  
Generator → OFF

### Scenario 2: Renewable Energy is Insufficient

Solar + Wind → Partial Supply  
Battery → Supplies Remaining Demand  
Generator → OFF

### Scenario 3: Renewable Energy and Battery are Insufficient

Renewable Energy → Partial Supply  
Battery → Preserved  
Backup Generator → ON

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard that allows users to enter environmental and energy parameters and obtain real-time predictions.

The dashboard provides:

- Energy demand prediction
- AI-based energy source recommendation
- Battery management decision
- Generator ON/OFF decision
- Model performance metrics
- Actual vs Predicted Demand graph
- Feature Importance graph
- Hourly Energy Demand graph
- Solar Power Generation graph
- Wind Power Generation graph
- Solar vs Wind Power comparison

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- Matplotlib
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

AI-Polar-Energy-Management-System/

├── app3.py  
├── project 1.ipynb  
├── polar_energy_dataset.csv  
├── requirements.txt  
└── README.md

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
