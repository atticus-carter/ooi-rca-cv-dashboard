import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import networkx as nx
from scripts.utils import load_local_files, load_uploaded_files, extract_data_columns

# --- Page Configuration ---
st.set_page_config(page_title="Ecological Metrics Analysis", layout="wide")
st.title("Ecological Metrics Analysis")

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
    data['timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Extract columns using the provided utility (assume species columns are returned as class_names)
    class_names, cluster_cols, env_vars = extract_data_columns(data)
    
    # For ecological analysis, we assume that class_names hold species counts/abundances.
    species_data = data[class_names].fillna(0)
    
    # --- Analysis Type Selection ---
    analysis_options = [
        "Diversity Indices Analysis",
        "Species Accumulation Curve",
        "Rank Abundance Curve",
        "Community NMDS Analysis",
        "Network Analysis",
        "Change Point Analysis"
    ]
    analysis_type = st.selectbox("Select Analysis Type", analysis_options)
    
    # --------------------- Diversity Indices Analysis --------------------- #
    if analysis_type == "Diversity Indices Analysis":
        st.subheader("Diversity Indices")
        
        # Define functions for Shannon and Simpson indices
        def shannon_index(row):
            counts = row[row > 0]
            proportions = counts / counts.sum()
            return -np.sum(proportions * np.log(proportions))
        
        def simpson_index(row):
            counts = row[row > 0]
            proportions = counts / counts.sum()
            return 1 - np.sum(proportions**2)
        
        # Calculate indices for each sample
        data['Richness'] = (species_data > 0).sum(axis=1)
        data['Shannon'] = species_data.apply(shannon_index, axis=1)
        data['Simpson'] = species_data.apply(simpson_index, axis=1)
        
        # Plot indices over time
        fig = px.line(data, x='timestamp', y=['Richness', 'Shannon', 'Simpson'], 
                      title="Diversity Indices Over Time",
                      labels={"value": "Index Value", "timestamp": "Time"})
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Plain Language Summary: This analysis plots three common diversity indices over time. "
                 "Species richness counts the number of species present, while the Shannon and Simpson indices "
                 "capture both richness and evenness. Shifts in these curves may indicate changes in community structure.")

    # --------------------- Species Accumulation Curve --------------------- #
    elif analysis_type == "Species Accumulation Curve":
        st.subheader("Species Accumulation Curve")
        
        # Randomize sample order and calculate cumulative species count
        shuffled_data = data.sample(frac=1, random_state=42)
        cumulative_species = []
        species_set = set()
        for i, (_, row) in enumerate(shuffled_data.iterrows(), start=1):
            present = row[class_names][row[class_names] > 0].index.tolist()
            species_set.update(present)
            cumulative_species.append(len(species_set))
            
        accumulation_df = pd.DataFrame({
            "Sample Number": range(1, len(cumulative_species) + 1),
            "Cumulative Species": cumulative_species
        })
        fig = px.line(accumulation_df, x="Sample Number", y="Cumulative Species",
                      title="Species Accumulation Curve")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Plain Language Summary: The species accumulation curve shows how the total number of unique species "
                 "increases as more samples are added. A plateau suggests that most species in the community have been observed.")

    # --------------------- Rank Abundance Curve --------------------- #
    elif analysis_type == "Rank Abundance Curve":
        st.subheader("Rank Abundance Curve")
        
        # Aggregate species counts across all samples and rank them
        total_counts = species_data.sum(axis=0).sort_values(ascending=False)
        rank = np.arange(1, len(total_counts) + 1)
        fig = px.line(x=rank, y=total_counts.values, markers=True,
                      title="Rank Abundance Curve",
                      labels={"x": "Species Rank", "y": "Total Abundance"})
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Plain Language Summary: The rank abundance curve orders species by total abundance. A steep decline indicates that few species dominate, while a gradual slope suggests a more even distribution among species.")

    # --------------------- Community NMDS Analysis --------------------- #
    elif analysis_type == "Community NMDS Analysis":
        st.subheader("Non-Metric Multidimensional Scaling (NMDS) Analysis")
        
        # Define Bray-Curtis dissimilarity between two samples
        def bray_curtis(u, v):
            if np.sum(u + v) == 0:
                return 0
            return np.sum(np.abs(u - v)) / np.sum(u + v)
        
        # Compute pairwise Bray-Curtis distance matrix
        species_array = species_data.values
        distance_matrix = squareform(pdist(species_array, metric=bray_curtis))
        
        # Run NMDS (non-metric MDS)
        mds = MDS(n_components=2, metric=False, dissimilarity="precomputed", random_state=42, n_init=10, max_iter=300)
        nmds_coords = mds.fit_transform(distance_matrix)
        nmds_df = pd.DataFrame(nmds_coords, columns=["NMDS1", "NMDS2"])
        if 'timestamp' in data.columns:
            nmds_df['timestamp'] = data['timestamp']
        
        fig = px.scatter(nmds_df, x="NMDS1", y="NMDS2", 
                         title="NMDS Plot of Community Composition",
                         hover_data=['timestamp'] if 'timestamp' in nmds_df.columns else None)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Plain Language Summary: NMDS reduces complex community data to two dimensions while preserving the rank order "
                 "of differences. Clusters in the NMDS plot may indicate samples with similar species composition.")

    # --------------------- Network Analysis --------------------- #
    elif analysis_type == "Network Analysis":
        st.subheader("Species Co-occurrence Network Analysis")
        
        # Compute Spearman correlations among species
        corr_matrix = species_data.corr(method='spearman')
        threshold = st.slider("Correlation Threshold for Network", 0.3, 1.0, 0.5)
        G = nx.Graph()
        for sp in class_names:
            G.add_node(sp)
        for i, sp1 in enumerate(class_names):
            for sp2 in class_names[i+1:]:
                corr_value = corr_matrix.loc[sp1, sp2]
                if abs(corr_value) >= threshold:
                    G.add_edge(sp1, sp2, weight=corr_value)
        
        # Compute layout for network visualization
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            marker=dict(size=10, color='skyblue'),
            hoverinfo='text'
        )
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title="Species Co-occurrence Network",
                                         showlegend=False,
                                         hovermode='closest',
                                         margin=dict(b=20, l=5, r=5, t=40)))
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Plain Language Summary: This network represents species as nodes and draws an edge between species when their co-occurrence (measured via Spearman correlation) exceeds the chosen threshold. Densely connected nodes suggest species that may interact or share similar habitat preferences.")

    # --------------------- Change Point Analysis --------------------- #
    elif analysis_type == "Change Point Analysis":
        st.subheader("Community Change Point Analysis")
        
        # Use the Shannon diversity index time series for change point detection.
        if 'Shannon' not in data.columns:
            def shannon_index(row):
                counts = row[row > 0]
                proportions = counts / counts.sum()
                return -np.sum(proportions * np.log(proportions))
            data['Shannon'] = species_data.apply(shannon_index, axis=1)
        
        ts = data.sort_values('timestamp')
        window_size = st.slider("Rolling Window Size", 5, 50, 10)
        rolling_mean = ts['Shannon'].rolling(window=window_size).mean()
        threshold = st.slider("Change Point Detection Threshold", 0.01, 1.0, 0.1)
        diff = rolling_mean.diff().abs()
        change_points = diff[diff > threshold].index
        
        fig = px.line(ts, x='timestamp', y='Shannon', title="Shannon Diversity Over Time with Change Points")
        fig.add_scatter(x=ts['timestamp'], y=rolling_mean, mode='lines', name='Rolling Mean')
        for cp in ts.loc[change_points, 'timestamp']:
            fig.add_vline(x=cp, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Plain Language Summary: This analysis detects moments when the community diversity (measured by the Shannon index) shifts significantly over time. Red dashed vertical lines mark potential change points that may reflect major ecological changes or disturbances.")

else:
    st.warning("Please select or upload CSV files with a 'Timestamp' column to analyze.")
    st.stop()
