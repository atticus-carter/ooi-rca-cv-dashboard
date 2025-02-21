import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from st_files_connection import FilesConnection

# Initialize connection for file loading
conn = st.connection('s3', type=FilesConnection)

# --- Page Configuration ---
st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("Time Series Analysis")

# --- Data Loading Functions ---
def load_local_files(base_dir, selected_csvs):
    dfs = []
    for csv_file in selected_csvs:
        file_path = os.path.join(base_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = csv_file
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {csv_file}: {e}")
    return dfs

def load_uploaded_files(uploaded_files):
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            df['source_file'] = uploaded_file.name
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
    return dfs

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
    data['timestamp'] = pd.to_datetime(data['timestamp'] if 'timestamp' in data.columns 
                                     else data['date'] + ' ' + data.get('time', '00:00:00'))

# --- Analysis Options ---
st.header("Analysis Options")
analysis_type = st.selectbox("Select Analysis Type", 
                           ["Basic Time Series", 
                            "Class Distribution", 
                            "Confidence Distribution",
                            "Cluster Analysis"])

# --- Visualization Functions ---
def plot_basic_timeseries():
    fig = px.line(data, x='timestamp', y='animal_count', 
                  color='class_name', title="Animal Counts Over Time")
    fig.update_layout(legend_title="Species")
    return fig

def plot_class_distribution():
    class_counts = data.groupby('class_name')['animal_count'].sum().reset_index()
    fig = px.bar(class_counts, x='class_name', y='animal_count', 
                 title="Total Counts by Class")
    fig.update_layout(xaxis_title="Species", yaxis_title="Count")
    return fig

def plot_confidence_distribution():
    fig = px.histogram(data, x='confidence', nbins=50,
                      title="Confidence Score Distribution")
    fig.update_layout(xaxis_title="Confidence Score", 
                     yaxis_title="Count")
    return fig

def plot_cluster_analysis():
    if not any("Cluster" in col for col in data.columns):
        st.warning("No cluster data available in the selected files.")
        return None
    
    cluster_cols = [col for col in data.columns if "Cluster" in col]
    cluster_data = data.melt(id_vars=['timestamp'], 
                           value_vars=cluster_cols,
                           var_name='Cluster Type',
                           value_name='Count')
    
    fig = px.line(cluster_data, x='timestamp', y='Count',
                  color='Cluster Type', title="Cluster Analysis Over Time")
    return fig

# --- Render Visualizations ---
if analysis_type == "Basic Time Series":
    st.plotly_chart(plot_basic_timeseries(), use_container_width=True)
    
elif analysis_type == "Class Distribution":
    st.plotly_chart(plot_class_distribution(), use_container_width=True)
    
elif analysis_type == "Confidence Distribution":
    st.plotly_chart(plot_confidence_distribution(), use_container_width=True)
    
elif analysis_type == "Cluster Analysis":
    fig = plot_cluster_analysis()
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# --- Download Options ---
if st.button("Download Analysis Data"):
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"analysis_{selected_camera}_{analysis_type}.csv",
        mime="text/csv"
    )
