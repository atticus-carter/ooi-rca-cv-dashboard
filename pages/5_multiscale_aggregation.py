import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from scripts.utils import load_local_files, load_uploaded_files

# --- Page Configuration ---
st.set_page_config(page_title="Multiscale Aggregation Analysis", layout="wide")
st.title("Multiscale Aggregation Analysis")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data Selection")
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
    
    # Optionally, load local CSV files (assumed stored in a folder called "aggregation_data")
    base_dir = os.path.join("aggregation_data")
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
    st.warning("CSV file must contain a 'Timestamp' column.")
    st.stop()

# Convert Timestamp to datetime and set it as the index
data['timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.sort_values('timestamp')
data.set_index('timestamp', inplace=True)

# Select numeric columns for aggregation (e.g., environmental metrics, species counts, etc.)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric columns found for aggregation analysis.")
    st.stop()

# --- Aggregation at Multiple Time Scales ---
daily_agg   = data[numeric_cols].resample('D').mean()
weekly_agg  = data[numeric_cols].resample('W').mean()
monthly_agg = data[numeric_cols].resample('M').mean()

# --- User Metric Selection ---
selected_metric = st.selectbox("Select Metric for Aggregation Visualization", numeric_cols)

# --- Create Tabs for Different Aggregation Scales ---
tabs = st.tabs(["Daily", "Weekly", "Monthly", "Comparison"])

with tabs[0]:
    st.subheader("Daily Aggregation")
    fig_daily = px.line(daily_agg.reset_index(), x='timestamp', y=selected_metric,
                        title=f"Daily Aggregated {selected_metric}",
                        labels={"timestamp": "Date", selected_metric: f"Mean {selected_metric}"})
    st.plotly_chart(fig_daily, use_container_width=True)
    st.write(
        "Plain Language Summary: This daily aggregation displays the average value of "
        f"{selected_metric} per day. Daily data can capture short-term fluctuations, "
        "but may also be more variable due to day-to-day differences."
    )

with tabs[1]:
    st.subheader("Weekly Aggregation")
    fig_weekly = px.line(weekly_agg.reset_index(), x='timestamp', y=selected_metric,
                         title=f"Weekly Aggregated {selected_metric}",
                         labels={"timestamp": "Week", selected_metric: f"Mean {selected_metric}"})
    st.plotly_chart(fig_weekly, use_container_width=True)
    st.write(
        "Plain Language Summary: The weekly aggregation smooths out daily variability by averaging "
        f"{selected_metric} over each week. This view can help highlight longer-term trends and reduce noise."
    )

with tabs[2]:
    st.subheader("Monthly Aggregation")
    fig_monthly = px.line(monthly_agg.reset_index(), x='timestamp', y=selected_metric,
                          title=f"Monthly Aggregated {selected_metric}",
                          labels={"timestamp": "Month", selected_metric: f"Mean {selected_metric}"})
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.write(
        "Plain Language Summary: Monthly aggregation further smooths the data by averaging over entire months. "
        "This scale is useful for identifying seasonal patterns and overall trends without short-term fluctuations."
    )

with tabs[3]:
    st.subheader("Comparative View")
    # For comparison, overlay the three aggregation scales for the selected metric.
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=daily_agg.index, y=daily_agg[selected_metric],
        mode='lines', name='Daily', line=dict(width=1)
    ))
    fig_comp.add_trace(go.Scatter(
        x=weekly_agg.index, y=weekly_agg[selected_metric],
        mode='markers+lines', name='Weekly', line=dict(width=2)
    ))
    fig_comp.add_trace(go.Scatter(
        x=monthly_agg.index, y=monthly_agg[selected_metric],
        mode='markers+lines', name='Monthly', line=dict(width=3)
    ))
    fig_comp.update_layout(title=f"Comparison of Daily, Weekly, and Monthly Aggregated {selected_metric}",
                           xaxis_title="Time",
                           yaxis_title=f"Mean {selected_metric}")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.write(
        "Plain Language Summary: This comparative plot overlays the daily, weekly, and monthly aggregated values of "
        f"{selected_metric}. Differences between these scales reveal how short-term variability is reduced when "
        "data are averaged over longer periods. Notice how the monthly trend is smoother than the daily data, indicating "
        "the damping of noise and highlighting longer-term trends."
    )
