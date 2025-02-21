import pandas as pd
import os
import streamlit as st

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
