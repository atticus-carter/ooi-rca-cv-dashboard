import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import yaml
from datetime import datetime, timedelta
import os
import glob

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

year_month = config["year_month"]

# Camera names
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

st.title("Data Analysis")

# --- Sidebar ---
st.sidebar.header("Parameters")

# Camera selection
selected_cameras = st.sidebar.multiselect("Select Cameras", camera_names, default=camera_names[:2])

# Date range selection
today = datetime.today()
default_start_date = today - timedelta(days=30)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", today)

# Class selection
all_classes = []
for camera_id in selected_cameras:
    parquet_dir = os.path.join("images", camera_id, "predictions")
    if not os.path.exists(parquet_dir):
        continue
    for date_dir in os.listdir(parquet_dir):
        if date_dir.startswith("date="):
            parquet_files = glob.glob(os.path.join(parquet_dir, date_dir, "*.parquet"))
            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    all_classes.extend(df['class_name'].unique().tolist())
                except Exception as e:
                    st.error(f"Error reading parquet file {parquet_file}: {e}")

unique_classes = list(set(all_classes))
selected_classes = st.sidebar.multiselect("Select Classes", unique_classes, default=unique_classes)

# --- Data Loading ---
@st.cache_data
def load_data(cameras, start, end, classes):
    data = []
    for camera_id in cameras:
        parquet_dir = os.path.join("images", camera_id, "predictions")
        if not os.path.exists(parquet_dir):
            continue
        for date_dir in os.listdir(parquet_dir):
            if date_dir.startswith("date="):
                date_str = date_dir.split("=")[1]
                date_dt = datetime.strptime(date_str, '%Y-%m-%d')
                if start.year <= date_dt.year <= end.year and start.month <= date_dt.month <= end.month and start.day <= date_dt.day <= end.day:
                    parquet_files = glob.glob(os.path.join(parquet_dir, date_dir, "*.parquet"))
                    for parquet_file in parquet_files:
                        try:
                            df = pd.read_parquet(parquet_file)
                            df = df[df['class_name'].isin(classes)]
                            data.append(df)
                        except Exception as e:
                            st.error(f"Error reading parquet file {parquet_file}: {e}")
    if data:
        df = pd.concat(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= pd.Timestamp(start)) & (df['timestamp'] <= pd.Timestamp(end))]
        return df
    else:
        return None

data_df = load_data(selected_cameras, start_date, end_date, selected_classes)

if data_df is None or data_df.empty:
    st.warning("No data found for the selected parameters.")
    st.stop()

# --- Plotly Plots ---

# 1. Timeline of Prediction Counts
st.subheader("Timeline of Prediction Counts")
timeline_df = data_df.groupby(['camera_id', 'timestamp']).size().reset_index(name='count')
fig_timeline = px.line(timeline_df, x='timestamp', y='count', color='camera_id', title="Prediction Counts Over Time")
st.plotly_chart(fig_timeline)

# 2. Stacked Area Chart of Class Distributions
st.subheader("Stacked Area Chart of Class Distributions")
area_df = data_df.groupby(['camera_id', 'timestamp', 'class_name']).size().reset_index(name='count')
fig_area = px.area(area_df, x='timestamp', y='count', color='class_name', facet_col='camera_id', title="Class Distributions Over Time")
st.plotly_chart(fig_area)

# 3. Stacked Bar Chart of Class Distributions
st.subheader("Stacked Bar Chart of Class Distributions")
bar_df = data_df.groupby(['camera_id', 'timestamp', 'class_name']).size().reset_index(name='count')
fig_bar = px.bar(bar_df, x='timestamp', y='count', color='class_name', facet_col='camera_id', title="Class Distributions Over Time")
st.plotly_chart(fig_bar)

# 4. Linear Regressions on Individual Species Plots
st.subheader("Linear Regressions on Individual Species Plots")
for class_name in selected_classes:
    species_df = data_df[data_df['class_name'] == class_name].groupby('timestamp').size().reset_index(name='count')
    if not species_df.empty:
        fig_species = px.scatter(species_df, x='timestamp', y='count', trendline="ols", title=f"Linear Regression for {class_name}")
        st.plotly_chart(fig_species)

# 5. Comparison Plots Between Different Cameras
st.subheader("Comparison Plots Between Different Cameras")
if len(selected_cameras) > 1:
    comparison_df = data_df.groupby(['camera_id', 'timestamp']).size().reset_index(name='count')
    fig_comparison = px.line(comparison_df, x='timestamp', y='count', color='camera_id', title="Comparison of Prediction Counts Between Cameras")
    st.plotly_chart(fig_comparison)
else:
    st.write("Select more than one camera to generate comparison plots.")
