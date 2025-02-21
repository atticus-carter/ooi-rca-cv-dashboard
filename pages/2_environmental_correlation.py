import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns
import os
import glob

# --- Page Configuration ---
st.set_page_config(page_title="Environmental Correlation Analysis", layout="wide")
st.title("Environmental Correlation Analysis")

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

# --- Load and Process Data ---
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
    # Extract columns
    class_names, cluster_cols, env_vars = extract_data_columns(data)
    data['timestamp'] = pd.to_datetime(data['Timestamp'])

    # --- Analysis Type Selection ---
    analysis_type = st.selectbox("Select Analysis Type", [
        "Correlation Matrix",
        "Environmental Response Curves",
        "PCA Analysis",
        "Time-lagged Correlations",
        "Threshold Analysis"
    ])

    if analysis_type == "Correlation Matrix":
        # Create correlation matrix between environmental variables and species
        corr_df = data[env_vars + class_names].corr()
        
        # Filter to show only env vars vs species correlations
        env_species_corr = corr_df.loc[env_vars, class_names]
        
        fig = px.imshow(
            env_species_corr,
            title="Environment-Species Correlation Matrix",
            labels=dict(x="Species", y="Environmental Variables", color="Correlation"),
            aspect="auto",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add statistical significance
        st.subheader("Statistical Significance")
        p_values = pd.DataFrame(index=env_vars, columns=class_names)
        for env in env_vars:
            for species in class_names:
                _, p_val = stats.pearsonr(data[env], data[species])
                p_values.loc[env, species] = p_val
        
        # Display significant correlations
        sig_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
        significant_corr = (p_values < sig_level) & (abs(env_species_corr) > 0.3)
        if significant_corr.any().any():
            st.write("Significant correlations (p < {:.2f}):".format(sig_level))
            for env in env_vars:
                for species in class_names:
                    if significant_corr.loc[env, species]:
                        st.write(f"{env} vs {species}: r = {env_species_corr.loc[env, species]:.3f}")

    elif analysis_type == "Environmental Response Curves":
        col1, col2 = st.columns(2)
        with col1:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
        with col2:
            species = st.selectbox("Select Species", class_names)
        
        # Create scatter plot with trend line
        fig = px.scatter(
            data, 
            x=env_var, 
            y=species,
            trendline="lowess",
            title=f"{species} Response to {env_var}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add density distribution
        fig2 = go.Figure()
        fig2.add_histogram2d(
            x=data[env_var],
            y=data[species],
            colorscale="Viridis",
            nbinsx=30,
            nbinsy=30
        )
        fig2.update_layout(title=f"Density Distribution: {species} vs {env_var}")
        st.plotly_chart(fig2, use_container_width=True)

    elif analysis_type == "PCA Analysis":
        # Standardize environmental variables
        scaler = StandardScaler()
        env_scaled = scaler.fit_transform(data[env_vars])
        
        # Perform PCA
        pca = PCA()
        env_pca = pca.fit_transform(env_scaled)
        
        # Create scree plot
        explained_variance = pca.explained_variance_ratio_ * 100
        fig = px.line(
            x=range(1, len(explained_variance) + 1),
            y=explained_variance,
            markers=True,
            title="PCA Scree Plot",
            labels={"x": "Principal Component", "y": "Explained Variance (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show PCA loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(env_vars))],
            index=env_vars
        )
        st.write("PCA Loadings:")
        st.dataframe(loadings)

    elif analysis_type == "Time-lagged Correlations":
        col1, col2, col3 = st.columns(3)
        with col1:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
        with col2:
            species = st.selectbox("Select Species", class_names)
        with col3:
            max_lag = st.slider("Maximum Lag (hours)", 1, 48, 24)
        
        # Calculate lagged correlations
        correlations = []
        for lag in range(max_lag + 1):
            corr = data[env_var].shift(lag).corr(data[species])
            correlations.append({"lag": lag, "correlation": corr})
        
        lag_df = pd.DataFrame(correlations)
        fig = px.line(
            lag_df,
            x="lag",
            y="correlation",
            title=f"Time-lagged Correlation: {species} vs {env_var}",
            labels={"lag": "Lag (hours)", "correlation": "Correlation Coefficient"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Threshold Analysis":
        col1, col2 = st.columns(2)
        with col1:
            env_var = st.selectbox("Select Environmental Variable", env_vars)
        with col2:
            species = st.selectbox("Select Species", class_names)
        
        # Calculate percentiles for threshold analysis
        percentiles = np.percentile(data[env_var], [25, 50, 75])
        thresholds = {
            "Low": (data[env_var] <= percentiles[0]),
            "Medium": (data[env_var] > percentiles[0]) & (data[env_var] <= percentiles[2]),
            "High": (data[env_var] > percentiles[2])
        }
        
        # Calculate species statistics for each threshold
        threshold_stats = []
        for threshold_name, mask in thresholds.items():
            stats_dict = {
                "threshold": threshold_name,
                "mean": data.loc[mask, species].mean(),
                "std": data.loc[mask, species].std(),
                "count": mask.sum()
            }
            threshold_stats.append(stats_dict)
        
        stats_df = pd.DataFrame(threshold_stats)
        fig = px.bar(
            stats_df,
            x="threshold",
            y="mean",
            error_y="std",
            title=f"{species} Abundance by {env_var} Thresholds",
            labels={"threshold": f"{env_var} Level", "mean": f"Mean {species} Count"}
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Please select or upload CSV files with 'Timestamp' column to analyze.")
    st.stop()
