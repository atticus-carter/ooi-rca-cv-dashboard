import os
import pandas as pd
import streamlit as st

def load_local_files(base_dir, selected_csvs):
    """Load data from local CSV files with the new format."""
    dfs = []
    for csv_file in selected_csvs:
        file_path = os.path.join(base_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = csv_file
            dfs.append(df)
        except FileNotFoundError:
            st.error(f"Error: File not found at {file_path}")
        except Exception as e:
            st.error(f"Error reading file {file_path}: {e}")
    return dfs

def load_uploaded_files(uploaded_files):
    """Load data from uploaded CSV files with the new format."""
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            df['source_file'] = uploaded_file.name
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading uploaded file {uploaded_file.name}: {e}")
    return dfs

def extract_data_columns(df):
    """Extract class names, cluster columns, and environmental variables from the new format."""
    class_names = []
    cluster_cols = []
    env_vars = []
    
    # Skip 'File' and 'Timestamp' columns
    for col in df.columns[2:]:
        if col.startswith('Cluster'):
            cluster_cols.append(col)
            break  # Stop adding class names once we hit clusters
        class_names.append(col)
    
    # Known environmental variables
    env_vars = [
        "Temperature", "Conductivity", "Pressure", "Salinity",
        "Oxygen Phase, usec", "Oxygen Temperature Voltage", "PressurePSI"
    ]
    
    return class_names, cluster_cols, env_vars

def melt_species_data(df, class_names):
    """Convert wide-format species data to long format."""
    melted_df = df.melt(
        id_vars=['timestamp'], 
        value_vars=class_names,
        var_name='class_name',
        value_name='animal_count'
    )
    return melted_df
