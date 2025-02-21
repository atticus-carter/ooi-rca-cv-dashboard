import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
from scipy.stats import f_oneway
import pymannkendall as mk
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import ruptures
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns

# --- Page Configuration ---
st.set_page_config(page_title="Ecological Metrics", layout="wide")
st.title("Ecological Metrics")

# Initialize connection for file loading

# --- Shared Data Loading Functions ---
# Note: These functions are identical across pages and could be moved to a utils.py file

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
    data['date'] = data['timestamp'].dt.date
    
    # Create pivot table for species counts
    species_pivot = data.groupby(['date'])[class_names].sum()

    # --- Ecological Metrics Options ---
    st.header("Ecological Metrics Analysis")
    metric_type = st.selectbox(
        "Select Analysis Type",
        ["Diversity Indices", "Species Accumulation", "Community Analysis", 
         "Network Analysis", "Change Point Analysis"]
    )

    # --- Ecological Metrics Specific Functions ---
    def calculate_diversity_indices(df):
        total = df.sum()
        props = df[df > 0] / total
        
        # Calculate indices
        richness = (df > 0).sum()
        shannon = -np.sum(props * np.log(props))
        simpson = 1 - np.sum((props * (props - 1)))
        evenness = shannon / np.log(richness) if richness > 1 else 0
        
        return pd.Series({
            'Richness': richness,
            'Shannon': shannon,
            'Simpson': simpson,
            'Evenness': evenness
        })

    # --- Metric-Specific Visualizations ---
    if metric_type == "Diversity Indices":
        st.subheader("Diversity Indices Over Time")
        
        # Calculate rolling diversity indices
        window_size = st.slider("Rolling Window (days)", 1, 30, 7)
        diversity_indices = species_pivot.rolling(window=window_size, min_periods=1).apply(calculate_diversity_indices)
        
        # Plot indices
        fig = go.Figure()
        for col in diversity_indices.columns:
            fig.add_trace(go.Scatter(x=diversity_indices.index, y=diversity_indices[col], name=col))
        fig.update_layout(title="Diversity Indices Over Time", xaxis_title="Date", yaxis_title="Index Value")
        st.plotly_chart(fig, use_container_width=True)

    elif metric_type == "Species Accumulation":
        st.subheader("Species Accumulation Curve")
        
        dates_sorted = sorted(species_pivot.index)
        cumulative_species = []
        species_set = set()
        
        for d in dates_sorted:
            present = species_pivot.loc[d][species_pivot.loc[d] > 0].index.tolist()
            species_set.update(present)
            cumulative_species.append(len(species_set))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(dates_sorted)+1)), 
                                y=cumulative_species,
                                mode='lines+markers'))
        fig.update_layout(title="Species Accumulation Curve",
                         xaxis_title="Number of Samples",
                         yaxis_title="Cumulative Species Count")
        st.plotly_chart(fig, use_container_width=True)

    elif metric_type == "Community Analysis":
        st.subheader("Community Composition Analysis")
        
        # Calculate Bray-Curtis dissimilarity matrix
        bray_curtis = pdist(species_pivot, metric='braycurtis')
        dissimilarity_matrix = squareform(bray_curtis)
        
        # Plot heatmap
        fig = px.imshow(dissimilarity_matrix,
                        labels=dict(x="Time Point", y="Time Point", color="Bray-Curtis Dissimilarity"),
                        title="Community Dissimilarity Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # NMDS plot
        from sklearn.manifold import MDS
        nmds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        nmds_coords = nmds.fit_transform(dissimilarity_matrix)
        
        fig = px.scatter(x=nmds_coords[:, 0], y=nmds_coords[:, 1],
                         labels={'x': 'NMDS1', 'y': 'NMDS2'},
                         title="NMDS Plot of Community Composition")
        st.plotly_chart(fig, use_container_width=True)

    elif metric_type == "Network Analysis":
        st.subheader("Species Co-occurrence Network")
        
        # Calculate correlations between species
        species_corr = species_pivot.corr()
        
        # Create network
        correlation_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5, 0.1)
        G = nx.Graph()
        
        for i in range(len(species_corr)):
            for j in range(i+1, len(species_corr)):
                corr = abs(species_corr.iloc[i,j])
                if corr >= correlation_threshold:
                    G.add_edge(species_corr.index[i], species_corr.index[j], weight=corr)
        
        if len(G.edges()) == 0:
            st.warning("No species pairs meet the correlation threshold. Try lowering the threshold.")
        else:
            # Create network visualization
            pos = nx.spring_layout(G)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                    line=dict(width=0.5, color='#888'), hoverinfo='none'))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                    text=list(G.nodes()), textposition="top center"))
            fig.update_layout(title="Species Co-occurrence Network",
                             showlegend=False, height=600)
            st.plotly_chart(fig, use_container_width=True)

    elif metric_type == "Change Point Analysis":
        st.subheader("Community Change Point Detection")
        
        # Prepare data for change point detection
        total_abundance = species_pivot.sum(axis=1).values.reshape(-1, 1)
        
        # Detect change points
        n_changepoints = st.slider("Number of Change Points", 1, 10, 3)
        algo = ruptures.Binseg(model="l2").fit(total_abundance)
        change_points = algo.predict(n_bkps=n_changepoints)
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=species_pivot.index, y=total_abundance.flatten(),
                                mode='lines', name='Total Abundance'))
        
        # Add vertical lines for change points
        for cp in change_points[:-1]:
            fig.add_vline(x=species_pivot.index[cp], line_dash="dash", line_color="red")
        
        fig.update_layout(title="Change Points in Community Composition",
                         xaxis_title="Date",
                         yaxis_title="Total Abundance")
        st.plotly_chart(fig, use_container_width=True)

    # --- Environmental Correlation Analysis ---
    st.header("Environmental Correlation Analysis")
    if env_vars:
        env_var = st.selectbox("Select Environmental Variable", env_vars)
        
        # Calculate correlation with diversity indices
        if 'diversity_indices' in locals():
            env_data = data.groupby('date')[env_var].mean()
            combined_data = diversity_indices.join(env_data, how='inner')
            
            correlations = combined_data.corr()[env_var].drop(env_var)
            st.write("Correlations with Diversity Indices:")
            st.dataframe(correlations)
            
            # Plot scatter plot
            fig = px.scatter(combined_data, x=env_var, y=correlations.index,
                             title=f"Correlation between {env_var} and Diversity Indices")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Diversity indices not calculated. Please select 'Diversity Indices' analysis first.")
    else:
        st.warning("No environmental variables found in the data.")

    # --- Download Options ---
    st.header("Download Results")
    if st.button("Download Analysis Results"):
        results = {
            "Diversity_Indices": calculate_diversity_indices(species_pivot.mean()),
            "Species_Richness": len(species_pivot.columns),
            "Total_Observations": species_pivot.sum().sum()
        }
        results_df = pd.DataFrame(results)
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ecological_metrics_{selected_camera}.csv",
            mime="text/csv"
        )
else:
    st.warning("Please select or upload CSV files with 'Timestamp' to analyze.")
    st.stop()
