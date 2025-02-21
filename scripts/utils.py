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
    """Extract class names, cluster columns, and environmental variables."""
    class_names = []
    cluster_cols = []
    env_vars = []
    
    for col in df.columns:
        if col.lower().startswith("cluster"):
            cluster_cols.append(col)
        elif col in ["Temperature","Conductivity","Pressure","Salinity","Oxygen Phase, usec","Oxygen Temperature Voltage","PressurePSI"]:
            env_vars.append(col)
        elif col not in ['File', 'Timestamp', 'date', 'time']:
            class_names.append(col)
            
    return class_names, cluster_cols, env_vars
