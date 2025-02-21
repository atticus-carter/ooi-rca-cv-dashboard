import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from scripts.utils import load_local_files, load_uploaded_files

# --- Page Configuration ---
st.set_page_config(page_title="Interactive Statistical Tests", layout="wide")
st.title("Interactive Statistical Tests")

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
if data.empty:
    st.warning("No data found in the files.")
    st.stop()

# --- Preprocess Columns Based on CSV Header ---
# CSV header example:
# File,Timestamp,Anoplopoma,Asteroidea,Bubble,Chionoecetes,Eptatretus,
# Euphausia,Liponema,Microstomus,Sebastes,Zoarcidae,Cluster 0,Cluster 1,
# Cluster 2,Cluster 3,Temperature,Conductivity,Pressure,Salinity,
# "Oxygen Phase, usec",Oxygen Temperature Voltage,PressurePSI
#
# Exclude "File" and "Timestamp" from analysis.
exclude_cols = ["File", "Timestamp"]
numeric_cols = [col for col in data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])]

if not numeric_cols:
    st.error("No numeric columns found for statistical testing.")
    st.stop()

st.info("All columns (except 'File' and 'Timestamp') are numeric. You can compare any two variables using the tests below.")

# --- Select Statistical Test ---
st.subheader("Choose a Statistical Test")
test_options = [
    "Paired T-Test", "Pearson Correlation", "Spearman Correlation"
]
selected_test = st.selectbox("Select Test", test_options)

# Prepare a string to capture results for export.
result_str = ""

# ----------------- Paired T-Test ----------------- #
if selected_test == "Paired T-Test":
    st.write("### Paired T-Test")
    if len(numeric_cols) < 2:
        st.error("At least two numeric variables are required for a Paired T-Test.")
    else:
        var1 = st.selectbox("Select First Variable", numeric_cols)
        var2 = st.selectbox("Select Second Variable", [col for col in numeric_cols if col != var1])
        paired_data = data[[var1, var2]].dropna()
        if paired_data.empty:
            st.error("No overlapping data between the selected variables.")
        else:
            t_stat, p_val = stats.ttest_rel(paired_data[var1], paired_data[var2])
            mean_diff = (paired_data[var1] - paired_data[var2]).mean()
            result_str = (
                f"**Paired T-Test for {var1} and {var2}:**\n"
                f"- Mean difference: {mean_diff:.3f}\n"
                f"- T-statistic: {t_stat:.3f}\n"
                f"- p-value: {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: Statistically significant difference between the paired variables."
            else:
                result_str += "Result: No statistically significant difference between the paired variables."
            st.text(result_str)

# ----------------- Pearson Correlation ----------------- #
elif selected_test == "Pearson Correlation":
    st.write("### Pearson Correlation")
    if len(numeric_cols) < 2:
        st.error("At least two numeric variables are required for correlation analysis.")
    else:
        var1 = st.selectbox("Select First Numeric Variable", numeric_cols)
        var2 = st.selectbox("Select Second Numeric Variable", [col for col in numeric_cols if col != var1])
        paired_data = data[[var1, var2]].dropna()
        if paired_data.empty:
            st.error("No overlapping data between the selected variables.")
        else:
            corr_coef, p_val = stats.pearsonr(paired_data[var1], paired_data[var2])
            result_str = (
                f"**Pearson Correlation between {var1} and {var2}:**\n"
                f"- Correlation coefficient: {corr_coef:.3f}\n"
                f"- p-value: {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: A statistically significant linear relationship exists between the variables."
            else:
                result_str += "Result: No statistically significant linear relationship exists between the variables."
            st.text(result_str)

# ----------------- Spearman Correlation ----------------- #
elif selected_test == "Spearman Correlation":
    st.write("### Spearman Correlation")
    if len(numeric_cols) < 2:
        st.error("At least two numeric variables are required for correlation analysis.")
    else:
        var1 = st.selectbox("Select First Numeric Variable", numeric_cols)
        var2 = st.selectbox("Select Second Numeric Variable", [col for col in numeric_cols if col != var1])
        paired_data = data[[var1, var2]].dropna()
        if paired_data.empty:
            st.error("No overlapping data between the selected variables.")
        else:
            corr_coef, p_val = stats.spearmanr(paired_data[var1], paired_data[var2])
            result_str = (
                f"**Spearman Correlation between {var1} and {var2}:**\n"
                f"- Correlation coefficient: {corr_coef:.3f}\n"
                f"- p-value: {p_val:.3f}\n\n"
            )
            if p_val < 0.05:
                result_str += "Result: A statistically significant monotonic relationship exists between the variables."
            else:
                result_str += "Result: No statistically significant monotonic relationship exists between the variables."
            st.text(result_str)

# ----------------- Export Results ----------------- #
if result_str:
    st.subheader("Export Results")
    export_format = st.selectbox("Select Export Format", ["Text (.txt)", "CSV (.csv)"])
    if export_format == "Text (.txt)":
        export_data = result_str
        export_filename = "statistical_test_results.txt"
        mime_type = "text/plain"
    else:
        export_data = pd.DataFrame({"Result": [result_str]})
        export_filename = "statistical_test_results.csv"
        mime_type = "text/csv"
    
    st.download_button(
        label="Download Results",
        data=export_data if isinstance(export_data, str) else export_data.to_csv(index=False),
        file_name=export_filename,
        mime=mime_type
    )
