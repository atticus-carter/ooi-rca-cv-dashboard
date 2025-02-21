import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import glob
import ruptures as rpt
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns

# --- Page Configuration ---
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("Anomaly Detection")

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

if 'Timestamp' in data.columns:
    # Extract data columns
    class_names, cluster_cols, env_vars = extract_data_columns(data)
    data['timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Combine all available variables for selection
    all_variables = class_names + env_vars + cluster_cols
    
    # --- Analysis Options ---
    st.header("Anomaly Detection Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        detection_method = st.selectbox(
            "Select Detection Method",
            ["Univariate Z-Score", "Isolation Forest", "Change Point Detection", "Multivariate Analysis"]
        )
    
    with col2:
        selected_vars = st.multiselect(
            "Select Variables for Analysis",
            all_variables,
            default=[all_variables[0]] if all_variables else None
        )
    
    if not selected_vars:
        st.warning("Please select at least one variable for analysis.")
        st.stop()
    
    # Function to format anomalies for export
    def format_anomalies_for_export(anomaly_dates, variables, scores=None):
        export_df = pd.DataFrame({
            'timestamp': anomaly_dates,
            'variables': ', '.join(variables)
        })
        if scores is not None:
            export_df['anomaly_score'] = scores
        return export_df
    
    # --- Anomaly Detection Methods ---
    if detection_method == "Univariate Z-Score":
        threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0)
        
        anomalies_all = pd.DataFrame()
        for var in selected_vars:
            z_scores = np.abs(stats.zscore(data[var].fillna(method='ffill')))
            anomaly_mask = z_scores > threshold
            if anomaly_mask.any():
                anomalies = data[anomaly_mask][['timestamp', var]]
                anomalies['anomaly_score'] = z_scores[anomaly_mask]
                anomalies_all = pd.concat([anomalies_all, anomalies])
        
        if not anomalies_all.empty:
            # Visualization
            fig = go.Figure()
            for var in selected_vars:
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=data[var],
                    name=var,
                    mode='lines'
                ))
                anomaly_points = anomalies_all[anomalies_all[var].notna()]
                fig.add_trace(go.Scatter(
                    x=anomaly_points['timestamp'],
                    y=anomaly_points[var],
                    mode='markers',
                    name=f'{var} Anomalies',
                    marker=dict(size=10, color='red')
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            export_df = format_anomalies_for_export(
                anomalies_all['timestamp'].unique(),
                selected_vars,
                anomalies_all['anomaly_score']
            )
            
    elif detection_method == "Isolation Forest":
        contamination = st.slider("Contamination Factor", 0.01, 0.1, 0.05)
        
        # Prepare data for Isolation Forest
        X = data[selected_vars].fillna(method='ffill')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_mask = anomaly_labels == -1
        
        # Visualization
        fig = go.Figure()
        for var in selected_vars:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data[var],
                name=var,
                mode='lines'
            ))
            # Add anomaly points
            fig.add_trace(go.Scatter(
                x=data.loc[anomaly_mask, 'timestamp'],
                y=data.loc[anomaly_mask, var],
                mode='markers',
                name=f'{var} Anomalies',
                marker=dict(size=10, color='red')
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        export_df = format_anomalies_for_export(
            data.loc[anomaly_mask, 'timestamp'],
            selected_vars
        )
        
    elif detection_method == "Change Point Detection":
        penalty_value = st.slider("Penalty Value", 1, 100, 10)
        min_size = st.slider("Minimum Segment Size", 5, 50, 20)
        
        anomaly_points = []
        for var in selected_vars:
            signal = data[var].fillna(method='ffill').values
            algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(signal)
            change_points = algo.predict(pen=penalty_value)
            
            if change_points:
                anomaly_points.extend(data.iloc[change_points]['timestamp'])
        
        if anomaly_points:
            # Visualization
            fig = go.Figure()
            for var in selected_vars:
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=data[var],
                    name=var,
                    mode='lines'
                ))
                for cp in anomaly_points:
                    fig.add_vline(x=cp, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            export_df = format_anomalies_for_export(
                pd.Series(anomaly_points).unique(),
                selected_vars
            )
            
    elif detection_method == "Multivariate Analysis":
        window_size = st.slider("Rolling Window Size", 5, 50, 20)
        threshold = st.slider("Mahalanobis Distance Threshold", 2.0, 10.0, 3.0)
        
        # Prepare data
        X = data[selected_vars].fillna(method='ffill')
        
        # Calculate Mahalanobis distance
        def mahalanobis(x, data):
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)
            diff = x - mean
            return np.sqrt(diff.dot(np.linalg.inv(cov)).dot(diff))
        
        distances = []
        for i in range(len(X)):
            start = max(0, i - window_size)
            window = X.iloc[start:i]
            if len(window) >= 2:  # Need at least 2 points for covariance
                distances.append(mahalanobis(X.iloc[i], window))
            else:
                distances.append(0)
        
        anomaly_mask = np.array(distances) > threshold
        
        # Visualization
        fig = go.Figure()
        for var in selected_vars:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data[var],
                name=var,
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=data.loc[anomaly_mask, 'timestamp'],
                y=data.loc[anomaly_mask, var],
                mode='markers',
                name=f'{var} Anomalies',
                marker=dict(size=10, color='red')
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        export_df = format_anomalies_for_export(
            data.loc[anomaly_mask, 'timestamp'],
            selected_vars,
            np.array(distances)[anomaly_mask]
        )
    
    # --- Export Options ---
    if 'export_df' in locals() and not export_df.empty:
        st.subheader("Export Anomalies")
        
        # Display detected anomalies
        st.write("Detected Anomalies:", export_df.shape[0])
        st.dataframe(export_df)
        
        # Download options
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download Anomalies CSV",
            data=csv,
            file_name=f"anomalies_{selected_camera}_{detection_method}.csv",
            mime="text/csv"
        )
        
else:
    st.warning("Please select or upload CSV files with a 'Timestamp' column to analyze.")
    st.stop()
