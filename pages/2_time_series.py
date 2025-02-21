import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns, melt_species_data

# --- Page Configuration ---
st.set_page_config(page_title="Time Series", layout="wide")
st.title("Time Series")

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
if 'Timestamp' in data.columns:
    # Extract data columns
    class_names, cluster_cols, env_vars = extract_data_columns(data)
    
    data['timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data.sort_values('timestamp')

    # Instead of filtering and melting manually, use the utility function
    melted_data = melt_species_data(data, class_names)
    
    # Filter by selected classes if needed
    if selected_classes:
        melted_data = melted_data[melted_data['class_name'].isin(selected_classes)]

    # --- Time Series Specific Visualizations ---
    st.header("Time Series Visualization")

    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox("Select Plot Type", 
                                ["Stacked Bar Chart", 
                                 "Stacked Area Chart", 
                                 "Environmental Variables",
                                 "Cluster Composition",
                                 "Per-Class Time Series"])

    with col2:
        available_classes = class_names
        selected_classes = st.multiselect("Filter Classes", 
                                        available_classes,
                                        default=available_classes)

    # --- Generate Visualizations ---
    if plot_type == "Stacked Bar Chart":
        fig = px.bar(melted_data, 
                     x='timestamp', 
                     y='animal_count',
                     color='class_name',
                     title="Animal Counts Over Time")
        fig.update_layout(legend_title="Species")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Stacked Area Chart":
        # Aggregate counts by timestamp and class
        df_area = melted_data.groupby(['timestamp', 'class_name'])['animal_count'].sum().reset_index()
        # Create percentage stacked area chart
        fig = px.area(df_area, 
                      x='timestamp', 
                      y='animal_count',
                      color='class_name',
                      title="Species Distribution Over Time")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Environmental Variables":
        col1, col2 = st.columns(2)
        with col1:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
        with col2:
            rolling_window = st.slider("Rolling Average Window", 1, 100, 10)

        # Create environmental variable plot with rolling average
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data[env_var],
            name=f'Raw {env_var}',
            mode='lines',
            line=dict(color='lightgray')
        ))
        
        # Add rolling average
        rolling_avg = data[env_var].rolling(window=rolling_window).mean()
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=rolling_avg,
            name=f'{env_var} ({rolling_window}-point rolling avg)',
            line=dict(color='blue')
        ))
        
        fig.update_layout(title=f"{env_var} Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Add correlation analysis
        st.subheader("Correlation with Species Counts")
        correlations = pd.DataFrame()
        for species in class_names:
            corr = data[env_var].corr(data[species])
            correlations.loc[species, 'Correlation'] = corr
        
        correlations = correlations.sort_values('Correlation', ascending=False)
        fig_corr = px.bar(correlations, 
                         x=correlations.index, 
                         y='Correlation',
                         title=f"Species Correlation with {env_var}")
        st.plotly_chart(fig_corr, use_container_width=True)

    elif plot_type == "Cluster Composition":
        if cluster_cols:
            cluster_col = st.selectbox("Select Cluster", cluster_cols)
            fig = px.line(data, 
                          x='timestamp', 
                          y=cluster_col,
                          title=f"{cluster_col} Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cluster data found in the data.")

    elif plot_type == "Per-Class Time Series":
        # Create subplot for each class
        fig = go.Figure()
        for class_name in selected_classes:
            class_data = melted_data[melted_data['class_name'] == class_name]
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
        class_totals = melted_data.groupby('class_name')['animal_count'].sum().sort_values(ascending=False)
        st.dataframe(class_totals)

    # --- Download Options ---
    st.header("Download Data")
    if st.button("Download Filtered Data"):
        csv = melted_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"timeseries_{selected_camera}.csv",
            mime="text/csv"
        )
else:
    st.warning("Please select or upload CSV files with 'Timestamp' to analyze.")
    st.stop()
