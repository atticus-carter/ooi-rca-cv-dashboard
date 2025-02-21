import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns, melt_species_data

# --- Page Configuration ---
st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("Time Series Analysis")

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
    
    # Convert to long format for species data
    melted_data = melt_species_data(data, class_names)

    # --- Analysis Options ---
    st.header("Analysis Options")
    analysis_type = st.selectbox("Select Analysis Type", 
                               ["Basic Time Series",
                                "Stacked Visualizations", 
                                "Multi-class Timeline",
                                "Class Distribution", 
                                "Confidence Distribution",
                                "Cluster Analysis",
                                "Environmental Variable Analysis"])

    # --- Visualization Functions ---
    def plot_basic_timeseries():
        fig = px.line(melted_data, x='timestamp', y='animal_count', 
                      color='class_name', title="Animal Counts Over Time")
        fig.update_layout(legend_title="Species")
        return fig

    def plot_class_distribution():
        class_counts = melted_data.groupby('class_name')['animal_count'].sum().reset_index()
        fig = px.bar(class_counts, x='class_name', y='animal_count', 
                     title="Total Counts by Class")
        fig.update_layout(xaxis_title="Species", yaxis_title="Count")
        return fig

    def plot_confidence_distribution():
        return None

    def plot_cluster_analysis():
        if not any("Cluster" in col for col in data.columns):
            st.warning("No cluster data available in the selected files.")
            return None
        
        cluster_cols = [col for col in data.columns if "Cluster" in col]
        cluster_data = data.melt(id_vars=['timestamp'], 
                               value_vars=cluster_cols,
                               var_name='Cluster Type',
                               value_name='Count')
        
        fig_clusters = go.Figure()
        for col in cluster_cols:
            fig_clusters.add_trace(go.Bar(
                name=col,
                x=cluster_data['timestamp'],
                y=cluster_data['Count']
            ))

        fig_clusters.update_layout(
            barmode='stack',
            title="Cluster Composition Over Time",
            xaxis_title="Time",
            yaxis_title="Count"
        )
        return fig_clusters

    def plot_environmental_analysis():
        if env_vars:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
            fig = px.line(data, x='timestamp', y=env_var,
                          title=f"{env_var} Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title=env_var)
            return fig
        else:
            st.warning("No environmental variables found in the data.")
            return None

    def plot_stacked_visualizations():
        # Get unique class names for selection
        available_classes = melted_data['class_name'].unique()
        
        # Create columns for chart type and class selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            chart_type = st.radio("Chart Type", ["Stacked Bar", "Stacked Area"])
        
        with col2:
            selected_classes = st.multiselect(
                "Select Classes to Include",
                available_classes,
                default=available_classes[:3]  # Default to first 3 classes
            )
        
        if not selected_classes:
            st.warning("Please select at least one class.")
            return None

        # Filter data for selected classes
        filtered_data = melted_data[melted_data['class_name'].isin(selected_classes)]
        
        # Pivot data for stacking
        pivot_data = filtered_data.pivot_table(
            index='timestamp',
            columns='class_name',
            values='animal_count',
            fill_value=0
        ).reset_index()

        if chart_type == "Stacked Bar":
            fig = go.Figure()
            for class_name in selected_classes:
                fig.add_trace(go.Bar(
                    name=class_name,
                    x=pivot_data['timestamp'],
                    y=pivot_data[class_name]
                ))
            fig.update_layout(
                barmode='stack',
                title="Stacked Species Distribution Over Time",
                xaxis_title="Time",
                yaxis_title="Count"
            )

        else:  # Stacked Area
            fig = go.Figure()
            for class_name in selected_classes:
                fig.add_trace(go.Scatter(
                    name=class_name,
                    x=pivot_data['timestamp'],
                    y=pivot_data[class_name],
                    mode='lines',
                    stackgroup='one'
                ))
            fig.update_layout(
                title="Species Distribution Area Over Time",
                xaxis_title="Time",
                yaxis_title="Count"
            )

        return fig

    def plot_multi_class_timeline():
        available_classes = melted_data['class_name'].unique()
        selected_classes = st.multiselect(
            "Select Classes to Display",
            available_classes,
            default=available_classes[:5]  # Default to first 5 classes
        )

        if not selected_classes:
            st.warning("Please select at least one class.")
            return None

        # Create subplots, one for each class
        fig = go.Figure()
        
        for i, class_name in enumerate(selected_classes):
            class_data = melted_data[melted_data['class_name'] == class_name]
            
            fig.add_trace(go.Scatter(
                x=class_data['timestamp'],
                y=class_data['animal_count'],
                name=class_name,
                yaxis=f'y{i+1}' if i > 0 else 'y'
            ))

        # Update layout with multiple y-axes
        layout_updates = {
            'title': 'Multi-class Timeline Analysis',
            'xaxis': {'title': 'Time'},
            'height': 100 + (len(selected_classes) * 200),  # Adjust height based on number of classes
            'showlegend': True,
            'grid': {'rows': len(selected_classes), 'columns': 1, 'pattern': 'independent'},
        }

        # Add separate y-axes for each class
        for i, class_name in enumerate(selected_classes):
            if i == 0:
                layout_updates['yaxis'] = {
                    'title': class_name,
                    'domain': [(len(selected_classes)-1-i)/len(selected_classes), 
                              (len(selected_classes)-i)/len(selected_classes)]
                }
            else:
                layout_updates[f'yaxis{i+1}'] = {
                    'title': class_name,
                    'domain': [(len(selected_classes)-1-i)/len(selected_classes), 
                              (len(selected_classes)-i)/len(selected_classes)]
                }

        fig.update_layout(**layout_updates)
        return fig

    # --- Render Visualizations ---
    if analysis_type == "Basic Time Series":
        st.plotly_chart(plot_basic_timeseries(), use_container_width=True)
        
    elif analysis_type == "Class Distribution":
        st.plotly_chart(plot_class_distribution(), use_container_width=True)
        
    elif analysis_type == "Confidence Distribution":
        st.write("Confidence Distribution Removed")
        
    elif analysis_type == "Cluster Analysis":
        fig = plot_cluster_analysis()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Environmental Variable Analysis":
        fig = plot_environmental_analysis()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Stacked Visualizations":
        fig = plot_stacked_visualizations()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Multi-class Timeline":
        fig = plot_multi_class_timeline()
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
else:
    st.warning("Please select or upload CSV files with 'Timestamp' to analyze.")
    st.stop()
