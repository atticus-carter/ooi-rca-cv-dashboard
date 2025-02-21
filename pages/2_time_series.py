import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from st_files_connection import FilesConnection
from utils import load_local_files, load_uploaded_files

# --- Page Configuration ---
st.set_page_config(page_title="Time Series", layout="wide")
st.title("Time Series")

# Initialize connection for file loading
conn = st.connection('s3', type=FilesConnection)

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

# --- Load Data ---
dfs = []
if selected_csvs:
    dfs.extend(load_local_files(base_dir, selected_csvs))
if uploaded_files:
    dfs.extend(load_uploaded_files(uploaded_files))

if not dfs:
    st.warning("Please select or upload CSV files to analyze.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)

# --- Data Preprocessing ---
if 'class_name' in data.columns:
    data = data[data['class_name'] != 'bubble']
    # Handle timestamp creation
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    elif 'date' in data.columns and 'time' in data.columns:
        data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    elif 'date' in data.columns:
        data['timestamp'] = pd.to_datetime(data['date'])
    data = data.sort_values('timestamp')

correctly# --- Time Series Specific Visualizations ---
st.header("Time Series Visualization")

col1, col2 = st.columns(2)
with col1:
    plot_type = st.selectbox("Select Plot Type", 
                            ["Stacked Bar Chart", 
                             "Stacked Area Chart", 
                             "Average Confidence",
                             "Per-Class Time Series"])

with col2:
    if 'class_name' in data.columns:
        available_classes = data['class_name'].unique()
        selected_classes = st.multiselect("Filter Classes", 
                                        available_classes,
                                        default=available_classes)
        data_filtered = data[data['class_name'].isin(selected_classes)]
    else:
        data_filtered = data
        st.warning("No class information found in the data.")

# --- Generate Visualizations ---
if plot_type == "Stacked Bar Chart":
    fig = px.bar(data_filtered, 
                 x='timestamp', 
                 y='animal_count',
                 color='class_name',
                 title="Animal Counts Over Time")
    fig.update_layout(legend_title="Species")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Stacked Area Chart":
    # Aggregate counts by timestamp and class
    df_area = data_filtered.groupby(['timestamp', 'class_name'])['animal_count'].sum().reset_index()
    # Create percentage stacked area chart
    fig = px.area(df_area, 
                  x='timestamp', 
                  y='animal_count',
                  color='class_name',
                  title="Species Distribution Over Time")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Average Confidence":
    fig = px.line(data_filtered, 
                  x='timestamp', 
                  y='confidence',
                  color='class_name',
                  title="Detection Confidence Over Time")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Per-Class Time Series":
    # Create subplot for each class
    fig = go.Figure()
    for class_name in selected_classes:
        class_data = data_filtered[data_filtered['class_name'] == class_name]
        fig.add_trace(
            go.Scatter(x=class_data['timestamp'],
                      y=class_data['animal_count'],
                      name=class_name,
                      mode='lines+markers')
        )
    fig.update_layout(
        title="Individual Species Counts Over Time",
        xaxis_title="Time",
        yaxis_title="Count",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Summary Statistics ---
st.header("Summary Statistics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Counts by Class")
    class_totals = data_filtered.groupby('class_name')['animal_count'].sum().sort_values(ascending=False)
    st.dataframe(class_totals)

with col2:
    st.subheader("Average Confidence by Class")
    confidence_stats = data_filtered.groupby('class_name')['confidence'].agg(['mean', 'std']).round(3)
    st.dataframe(confidence_stats)

# --- Download Options ---
st.header("Download Data")
if st.button("Download Filtered Data"):
    csv = data_filtered.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"timeseries_{selected_camera}.csv",
        mime="text/csv"
    )
