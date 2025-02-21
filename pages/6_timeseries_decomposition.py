import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from statsmodels.tsa.seasonal import seasonal_decompose
from scripts.utils import load_local_files, load_uploaded_files

# --- Page Configuration ---
st.set_page_config(page_title="Time Series Decomposition", layout="wide")
st.title("Time Series Decomposition")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data Selection")
    camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]
    selected_camera = st.selectbox("Select Camera", camera_names)
    
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
    
    base_dir = os.path.join("timeseries", selected_camera)
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    csv_files = [os.path.relpath(f, base_dir) for f in csv_files]
    selected_csvs = st.multiselect("Select CSV Files", csv_files)

# --- Load and Process Data ---
dfs = []
if selected_csvs:
    dfs.extend(load_local_files(base_dir, selected_csvs))
if uploaded_files:
    dfs.extend(load_uploaded_files(uploaded_files))

if not dfs:
    st.warning("Please select or upload CSV files to analyze.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

if 'Timestamp' not in data.columns:
    st.warning("The CSV file must contain a 'Timestamp' column.")
    st.stop()

# Convert Timestamp to datetime and sort the data
data['timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.sort_values('timestamp')
data.set_index('timestamp', inplace=True)

# --- Identify Numeric Columns ---
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric columns found for decomposition analysis.")
    st.stop()

# --- User Selections for Decomposition ---
st.subheader("Time Series Decomposition Settings")
selected_variable = st.selectbox("Select Variable for Decomposition", numeric_cols)
model_type = st.selectbox("Select Decomposition Model", options=["additive", "multiplicative"], index=0)
period_input = st.text_input("Enter Seasonal Period (e.g., 7 for weekly seasonality in daily data)", value="7")

try:
    period = int(period_input)
except ValueError:
    st.error("Please enter a valid integer for the seasonal period.")
    st.stop()

# --- Perform Seasonal Decomposition ---
ts = data[selected_variable].dropna()
try:
    decomposition = seasonal_decompose(ts, model=model_type, period=period, extrapolate_trend='freq')
except Exception as e:
    st.error(f"Decomposition failed: {e}")
    st.stop()

# --- Create Decomposition Plot with Plotly ---
fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
# Observed
fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Observed'), row=1, col=1)
# Trend
fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
# Seasonal
fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
# Residual
fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)

fig.update_layout(height=800, title_text=f"Decomposition of {selected_variable} ({model_type.capitalize()} Model, Period = {period})")
fig.update_xaxes(title_text="Time", row=4, col=1)
fig.update_yaxes(title_text=selected_variable, row=1, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- Plain Language Summary ---
st.subheader("Plain Language Summary")
st.write(
    f"**Decomposition Overview for {selected_variable}:**\n\n"
    f"1. **Observed:** This is the original time series data for {selected_variable}.\n\n"
    f"2. **Trend:** The trend component reveals the underlying long-term progression of {selected_variable}. "
    f"It filters out short-term fluctuations, highlighting the overall direction (increasing, decreasing, or stable) over time.\n\n"
    f"3. **Seasonal:** The seasonal component captures regular, repeating patterns in the data. "
    f"In this analysis, a period of {period} was used, which should correspond to the expected seasonality (for example, weekly cycles in daily data).\n\n"
    f"4. **Residual:** The residual component represents the remaining variation after removing the trend and seasonal effects. "
    f"This is the noise or irregular fluctuations that are not explained by the systematic components.\n\n"
    f"By decomposing the time series, we can better understand the distinct factors influencing {selected_variable} and potentially improve forecasting or detect anomalies."
)
