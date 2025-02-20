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

# --- Page Title ---
st.title("Timeseries Data View")

# --- Camera Selection ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]
selected_camera = st.selectbox("Select Camera", camera_names)

# --- File Upload ---
uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

# --- List Available CSV Files ---
base_dir = os.path.join("timeseries", selected_camera)
st.write(f"Searching for CSV files in: {base_dir}")

csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
st.write(f"Found CSV files: {csv_files}")

csv_files = [os.path.relpath(f, base_dir) for f in csv_files]

if not csv_files and not uploaded_files:
    st.warning("No CSV files found for the selected camera and no files uploaded.")
    st.stop()

selected_csvs = st.multiselect("Select CSV Files", csv_files)

if not selected_csvs and not uploaded_files:
    st.warning("Please select at least one CSV file or upload files.")
    st.stop()

# --- Load and Concatenate Selected CSV Files ---
dfs = []

# Load data from selected CSV files
for csv_file in selected_csvs:
    file_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(file_path)
    df['source_file'] = csv_file
    dfs.append(df)

# Load data from uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            df['source_file'] = uploaded_file.name  # Use filename as source
            dfs.append(df)
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {e}")

if not dfs:
    st.warning("No data loaded. Please select CSV files or upload them.")
    st.stop()

data = pd.concat(dfs)

# --- CSV Schema Detection and Processing ---
if any("Cluster" in col for col in data.columns):
    schema_option = st.radio("Select CSV Schema", options=["Standard", "Cluster"], index=0)
else:
    schema_option = "Standard"

if schema_option == "Cluster":
    # Process cluster CSV: extract datetime from "File" column; assume filename pattern has YYYYMMDDTHHMMSS.
    import re
    def extract_datetime(filename):
        m = re.search(r'(\d{8}T\d{6})', filename)
        if m:
            return pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S")
        else:
            return pd.NaT
    data["timestamp"] = data["File"].apply(extract_datetime)
    data = data.dropna(subset=["timestamp"])
    # Convert any column that contains "Cluster" in its header to numeric.
    cluster_cols = [col for col in data.columns if "Cluster" in col]
    for col in cluster_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        
    # --- Plot Cluster Counts Over Time ---
    st.subheader("Cluster Counts Over Time")
    fig_cluster = go.Figure()
    for col in cluster_cols:
        fig_cluster.add_trace(go.Scatter(x=data["timestamp"], y=data[col], mode="lines+markers", name=col))
    fig_cluster.update_layout(title="Clusters Over Time", xaxis_title="Time", yaxis_title="Count")
    st.plotly_chart(fig_cluster)
    
    # --- Process and Plot Object Detection Data ---
    # Identify object detection columns (assumed to be those whose header is a digit)
    obj_detect_cols = [col for col in data.columns if col.isdigit()]
    if obj_detect_cols:
        st.subheader("Object Detections as Percentage Over Time")
        # Group by date: use the 'timestamp' column already extracted
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        obj_daily = data.groupby('date')[obj_detect_cols].sum()
        # For each date, convert counts to percentages (i.e. out of 100)
        obj_percent = obj_daily.div(obj_daily.sum(axis=1), axis=0) * 100
        obj_percent = obj_percent.reset_index().melt(id_vars='date', value_vars=obj_detect_cols,
                                                     var_name='class', value_name='percentage')
        fig_obj = px.area(obj_percent, x='date', y='percentage', color='class',
                          title="Object Detections (% of Total) Over Time")
        st.plotly_chart(fig_obj)

else:
    # Standard schema: create timestamp from 'date' (and 'time' if present)
    if 'time' in data.columns:
        data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    else:
        data['timestamp'] = pd.to_datetime(data['date'])
    data.sort_values('timestamp', inplace=True)

# --- If Cluster schema is active, graph cluster counts over time ---
if schema_option == "Cluster":
    st.subheader("Cluster Counts Over Time")
    fig_cluster = go.Figure()
    for col in cluster_cols:
        fig_cluster.add_trace(go.Scatter(x=data["timestamp"], y=data[col], mode="lines+markers", name=col))
    fig_cluster.update_layout(title="Clusters Over Time", xaxis_title="Time", yaxis_title="Count")
    st.plotly_chart(fig_cluster)

# --- Process Cluster CSV Schema if detected ---
unique_classes = data['class_name'].unique().tolist()
selected_classes = st.multiselect("Select Classes", unique_classes, default=unique_classes)
if selected_classes:
    data = data[data['class_name'].isin(selected_classes)]

# --- Confidence Threshold ---
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
data = data[data['confidence'] >= conf_threshold]

# --- Create Timestamp Column ---
# If a time column is present, combine with date; otherwise use date only.
if 'time' in data.columns:
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
else:
    data['timestamp'] = pd.to_datetime(data['date'])
data.sort_values('timestamp', inplace=True)

# --- Plotting Options ---
plot_type = st.selectbox("Select Plot Type", ["Stacked Bar Chart", "Stacked Area Chart", "Average Confidence"])

if plot_type == "Stacked Bar Chart":
    fig = px.bar(data, x='timestamp', y='animal_count', color='class_name', title="Stacked Bar Chart")
elif plot_type == "Stacked Area Chart":
    # Aggregate counts by timestamp and class
    df_area = data.groupby(['timestamp', 'class_name'])['animal_count'].sum().reset_index()
    # Pivot to have timestamps as index and class_names as columns
    pivot = df_area.pivot(index='timestamp', columns='class_name', values='animal_count').fillna(0)
    # Convert raw counts to percentages per timestamp row
    pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_percent = pivot_percent.reset_index()
    # Melt back to long format for Plotly Express
    df_melted = pivot_percent.melt(id_vars='timestamp', var_name='class_name', value_name='percentage')
    fig = px.area(df_melted, x='timestamp', y='percentage', color='class_name', title="Stacked Area Chart (Percentage)")
elif plot_type == "Average Confidence":
    fig = px.line(data, x='timestamp', y='confidence', color='class_name', title="Average Confidence Over Time")
st.plotly_chart(fig)

# --- Ecology Metrics ---
st.subheader("Ecology Metrics")
col1, col2 = st.columns(2)
with col1:
    st.write("Total Counts by Class:")
    total_counts = data.groupby('class_name')['animal_count'].sum()
    st.write(total_counts)
with col2:
    st.write("Average Confidence by Class:")
    average_confidences = data.groupby('class_name')['confidence'].mean()
    st.write(average_confidences)

# --- Ecological Metrics Plot ---
st.subheader("Ecological Metrics Over Time")

# Create pivot table for species counts
df_ecol = data.copy()
df_ecol['date'] = pd.to_datetime(df_ecol['timestamp']).dt.date
species_pivot = df_ecol.groupby(['date', 'class_name'])['animal_count'].sum().unstack(fill_value=0)

# Initialize the diversity metrics DataFrame with basic counts
diversity_metrics = pd.DataFrame(index=species_pivot.index)
diversity_metrics['Total_Annotations'] = species_pivot.sum(axis=1)
diversity_metrics['Species_Richness'] = (species_pivot > 0).sum(axis=1)

# Calculate additional diversity metrics
def calculate_diversity_metrics(row):
    counts = row[row > 0]  # Only consider non-zero counts
    total = counts.sum()
    if total == 0:
        return pd.Series({
            'Shannon_Wiener': 0,
            'Simpson': 0,
            'Pielou': 0,
            'Chao1': 0,
            'Berger_Parker': 0,
            'Hill_N1': 0
        })
    
    # Species richness
    richness = len(counts)
    
    # Shannon-Wiener Index
    proportions = counts / total
    shannon = -np.sum(proportions * np.log(proportions))
    
    # Simpson's Index
    simpson = 1 - np.sum((counts * (counts - 1)) / (total * (total - 1)))
    
    # Pielou's Evenness
    pielou = shannon / np.log(richness) if richness > 1 else 0
    
    # Chao1 Richness
    singletons = len(counts[counts == 1])
    doubletons = len(counts[counts == 2])
    chao1 = richness + ((singletons * singletons) / (2 * doubletons)) if doubletons > 0 else richness
    
    # Berger-Parker Dominance
    berger_parker = counts.max() / total
    
    # Hill Numbers (q=1)
    hill_1 = np.exp(shannon)
    
    return pd.Series({
        'Shannon_Wiener': shannon,
        'Simpson': simpson,
        'Pielou': pielou,
        'Chao1': chao1,
        'Berger_Parker': berger_parker,
        'Hill_N1': hill_1
    })

# Calculate additional metrics and combine with basic metrics
additional_metrics = species_pivot.apply(calculate_diversity_metrics, axis=1)
diversity_metrics = pd.concat([diversity_metrics, additional_metrics], axis=1)

# Add rolling averages
window_size = 7
for col in diversity_metrics.columns:
    diversity_metrics[f'{col}_7d'] = diversity_metrics[col].rolling(window=window_size).mean()

# Let user select which metrics to display
available_metrics = [
    'Species_Richness', 'Shannon_Wiener', 'Simpson', 'Pielou', 
    'Chao1', 'Berger_Parker', 'Hill_N1', 'Total_Annotations'
]

selected_metrics = st.multiselect(
    "Choose diversity metrics to plot",
    available_metrics,
    default=['Species_Richness', 'Shannon_Wiener', 'Simpson']
)

# Create plot with selected metrics
if selected_metrics:
    fig_ecol = go.Figure()
    for metric in selected_metrics:
        # Plot both raw and smoothed data
        fig_ecol.add_trace(go.Scatter(
            x=diversity_metrics.index,
            y=diversity_metrics[metric],
            mode='lines',
            name=f'{metric} (Raw)',
            line=dict(width=1)
        ))
        fig_ecol.add_trace(go.Scatter(
            x=diversity_metrics.index,
            y=diversity_metrics[f'{metric}_7d'],
            mode='lines',
            name=f'{metric} (7-day Avg)',
            line=dict(width=2)
        ))
    
    fig_ecol.update_layout(
        title='Ecological Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Metric Value',
        legend_title='Metrics',
        template='plotly_white',
        showlegend=True
    )
    st.plotly_chart(fig_ecol)

    # Add explanatory text
    st.markdown("""
    **Metric Descriptions:**
    - **Species Richness**: Number of different species present
    - **Shannon-Wiener Index**: Measure of diversity considering both abundance and evenness
    - **Simpson's Index**: Probability that two randomly selected individuals belong to different species
    - **Pielou's Evenness**: How evenly individuals are distributed among species
    - **Chao1**: Estimated true species richness including unobserved species
    - **Berger-Parker Dominance**: Relative abundance of the most abundant species
    - **Hill Number (N1)**: Effective number of species based on Shannon entropy
    - **Total Annotations**: Total number of annotations across all species
    """)

# --- Advanced Ecological Analysis ---
st.subheader("Advanced Ecological Analysis")

# Mann-Kendall Trend Test on Total Annotations (7-day Avg)
try:
    import pymannkendall as mk
    ts = diversity_metrics['Total_Annotations_7d'].dropna()
    if len(ts) >= 10:
        result = mk.original_test(ts)
        st.write("Mann-Kendall Trend Test on Total Annotations (7d Avg):")
        st.write(f"Trend: {result.trend}, p-value: {result.p}")
    else:
        st.warning("Not enough data for Mann-Kendall test (need at least 10 data points)")
except ImportError:
    st.warning("pymannkendall library not installed. Skipping Mann-Kendall trend test.")
except Exception as e:
    st.error(f"Error performing Mann-Kendall test: {e}")

# ARIMA Forecasting for Total Annotations (7-day Avg)
st.subheader("ARIMA Forecasting")
try:
    from statsmodels.tsa.arima.model import ARIMA
    ts = diversity_metrics['Total_Annotations_7d'].dropna()
    if len(ts) >= 10:
        # Convert index to datetime with frequency
        ts.index = pd.DatetimeIndex(ts.index).to_period('D').to_timestamp()
        model_arima = ARIMA(ts, order=(1,1,1)).fit()
        forecast = model_arima.get_forecast(steps=10)
        forecast_index = pd.date_range(ts.index[-1], periods=10, freq='D')
        forecast_series = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Observed'))
        fig_arima.add_trace(go.Scatter(x=forecast_index, y=forecast_series, mode='lines', name='Forecast'))
        fig_arima.add_trace(go.Scatter(
            x=forecast_index, y=conf_int.iloc[:, 0],
            mode='lines', line=dict(color='gray'), name='Lower CI'))
        fig_arima.add_trace(go.Scatter(
            x=forecast_index, y=conf_int.iloc[:, 1],
            mode='lines', line=dict(color='gray'), name='Upper CI'))
        fig_arima.update_layout(
            title="ARIMA Forecast: Total Annotations (7d Avg)",
            xaxis_title="Date",
            yaxis_title="Total Annotations"
        )
        st.plotly_chart(fig_arima)
    else:
        st.warning("Not enough data for ARIMA forecast (need at least 10 data points)")
except Exception as e:
    st.error(f"Error performing ARIMA forecasting: {e}")

# --- Fit Statistical Model if Data is Sufficient ---
try:
    X = species_pivot[['Total_Annotations_7d', 'Species_Richness_7d']].dropna()
    y = diversity_metrics['Shannon_Wiener'].dropna()
    X, y = X.align(y, join='inner', axis=0)
    if not X.empty and not y.empty and X.shape[0] >= 2:
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.write("Statistical Model Summary:")
        st.write(model.summary())
    else:
        st.write("Insufficient data for statistical model (need at least 2 data points).")
except Exception as e:
    st.error(f"Error fitting statistical model: {e}")

# --- Per Class Graphs ---
st.subheader("Per Class Graphs")

from plotly.subplots import make_subplots  # Import make_subplots

unique_classes = data['class_name'].unique()
num_classes = len(unique_classes)

# Create subplots: one row per unique class, shared x-axis
fig_class = make_subplots(rows=num_classes, cols=1, subplot_titles=unique_classes, shared_xaxes=True)

for i, class_name in enumerate(unique_classes):
    class_data = data[data['class_name'] == class_name]
    fig_class.add_trace(
        go.Scatter(x=class_data['timestamp'], y=class_data['animal_count'], mode='lines', name=class_name),
        row=i+1, col=1
    )

fig_class.update_layout(
    title="Animal Counts Over Time by Class",
    xaxis_title="Time",
    yaxis_title="Animal Count",
    template='plotly_white',
    height=300*num_classes  # Adjust height per subplot
)

st.plotly_chart(fig_class)

# --- Interspecies Occurrence Correlation Heatmap ---
st.subheader("Interspecies Occurrence Correlation Heatmap")
try:
    # Create a daily pivot table of animal counts for each species.
    species_daily = data.copy()
    species_daily['date'] = pd.to_datetime(species_daily['timestamp']).dt.date
    species_pivot = species_daily.groupby(['date', 'class_name'])['animal_count'].sum().unstack(fill_value=0)
    # Compute the correlation matrix among species.
    corr_species = species_pivot.corr()
    # Plot the correlation heatmap with increased size.
    fig_species_corr = px.imshow(
        corr_species, 
        text_auto=True, 
        color_continuous_scale='RdBu_r', 
        title="Interspecies Occurrence Correlation Heatmap"
    )
    fig_species_corr.update_layout(width=800, height=800)
    st.plotly_chart(fig_species_corr)
except Exception as e:
    st.error(f"Error generating species correlation heatmap: {e}")

# --- Per Species ARIMA Forecasting ---
st.subheader("Per Species ARIMA Forecasting")
from statsmodels.tsa.arima.model import ARIMA  # Ensure ARIMA is imported

unique_species = data['class_name'].unique()
for species in unique_species:
    try:
        # Group by date for the species and sum animal counts
        df_species = data[data['class_name'] == species].copy()
        df_species['date'] = pd.to_datetime(df_species['timestamp']).dt.date
        ts_species = df_species.groupby('date')['animal_count'].sum().dropna()
        if len(ts_species) < 10:
            st.write(f"Not enough data for ARIMA forecast for species: {species}")
            continue
        model_arima_species = ARIMA(ts_species, order=(1,1,1)).fit()
        forecast_species = model_arima_species.get_forecast(steps=10)
        forecast_index_species = pd.date_range(pd.to_datetime(ts_species.index[-1]), periods=10, freq='D')
        forecast_series_species = forecast_species.predicted_mean
        conf_int_species = forecast_species.conf_int()
        fig_species = go.Figure()
        fig_species.add_trace(go.Scatter(x=ts_species.index, y=ts_species, mode='lines', name='Observed'))
        fig_species.add_trace(go.Scatter(x=forecast_index_species, y=forecast_series_species, mode='lines', name='Forecast'))
        fig_species.add_trace(go.Scatter(
            x=forecast_index_species, y=conf_int_species.iloc[:, 0],
            mode='lines', line=dict(color='gray'), name='Lower CI'))
        fig_species.add_trace(go.Scatter(
            x=forecast_index_species, y=conf_int_species.iloc[:, 1],
            mode='lines', line=dict(color='gray'), name='Upper CI'))
        fig_species.update_layout(
            title=f"ARIMA Forecast for {species}",
            xaxis_title="Date",
            yaxis_title="Animal Count",
            template='plotly_white'
        )
        st.plotly_chart(fig_species)
    except Exception as e:
        st.error(f"Error forecasting for species {species}: {e}")

# --- Seasonal Decomposition ---
from statsmodels.tsa.seasonal import seasonal_decompose

st.subheader("Seasonal Decomposition of Total Annotations")
ts_decomp = diversity_metrics['Total_Annotations'].dropna()  # Using raw total annotations
if len(ts_decomp) >= 14:  # need at least two periods (e.g., 7-day period)
    decomp_result = seasonal_decompose(ts_decomp, model='additive', period=7)
    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Scatter(x=ts_decomp.index, y=decomp_result.trend, mode='lines', name='Trend'))
    fig_decomp.add_trace(go.Scatter(x=ts_decomp.index, y=decomp_result.seasonal, mode='lines', name='Seasonal'))
    fig_decomp.add_trace(go.Scatter(x=ts_decomp.index, y=decomp_result.resid, mode='lines', name='Residual'))
    fig_decomp.update_layout(title="Seasonal Decomposition (7-day Period)", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig_decomp)
    
    # --- Plain Language Summary ---
    # Compute variance explained by seasonal component
    variance_total = np.var(ts_decomp)
    var_seasonal = np.var(decomp_result.seasonal.dropna())
    explained = (var_seasonal / variance_total) * 100
    st.write(f"Plain Language Summary: The seasonal component explains approximately {explained:.1f}% of the total variation in total annotations.")

    # --- FFT Analysis ---
    # Detrend the time series by subtracting its mean
    ts_detrended = ts_decomp - np.mean(ts_decomp)
    N = len(ts_detrended)
    fft_vals = np.fft.fft(ts_detrended)
    fft_freq = np.fft.fftfreq(N, d=1)  # Assuming a daily sampling period
    
    # Keep only the positive frequencies
    positive = fft_freq > 0
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=fft_freq[positive], y=np.abs(fft_vals)[positive], mode='lines+markers', name='FFT Amplitude'))
    fig_fft.update_layout(title="FFT of Total Annotations", xaxis_title="Frequency (cycles per day)", yaxis_title="Amplitude")
    st.plotly_chart(fig_fft)

else:
    st.write("Not enough data for seasonal decomposition.")

# --- Species Accumulation and Rarefaction ---
st.subheader("Species Accumulation Curve")
# Using species_daily from earlier; create a pivot table per date.
species_pivot = species_daily.groupby(['date', 'class_name'])['animal_count'].sum().unstack(fill_value=0)
dates_sorted = sorted(species_pivot.index)
cumulative_species = []
species_set = set()
for d in dates_sorted:
    present = species_pivot.loc[d][species_pivot.loc[d] > 0].index.tolist()
    species_set.update(present)
    cumulative_species.append(len(species_set))
fig_accum = go.Figure()
fig_accum.add_trace(go.Scatter(x=list(range(1, len(dates_sorted)+1)), y=cumulative_species,
                               mode='lines+markers', name='Accumulation'))
fig_accum.update_layout(title="Species Accumulation Curve", xaxis_title="Number of Samples", yaxis_title="Cumulative Species Count")
st.plotly_chart(fig_accum)

# --- NMDS and PCoA for Interspecies Composition ---
st.subheader("NMDS and PCoA of Interspecies Composition")
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# Compute Bray-Curtis dissimilarities from the species pivot table.
bray_distance = squareform(pdist(species_pivot, metric='braycurtis'))
# NMDS (non-metric MDS)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, metric=False)
nmds_coords = mds.fit_transform(bray_distance)
fig_nmds = go.Figure(data=go.Scatter(x=nmds_coords[:,0], y=nmds_coords[:,1],
                                      mode='markers', text=[str(d) for d in species_pivot.index]))
fig_nmds.update_layout(title="NMDS of Interspecies Composition", xaxis_title="NMDS1", yaxis_title="NMDS2")
st.plotly_chart(fig_nmds)
# PCoA via PCA on the dissimilarity matrix
pcoa = PCA(n_components=2)
pcoa_coords = pcoa.fit_transform(bray_distance)
fig_pcoa = go.Figure(data=go.Scatter(x=pcoa_coords[:,0], y=pcoa_coords[:,1],
                                      mode='markers', text=[str(d) for d in species_pivot.index]))
fig_pcoa.update_layout(title="PCoA of Interspecies Composition", xaxis_title="PCoA1", yaxis_title="PCoA2")
st.plotly_chart(fig_pcoa)

# --- Interactive Per Class ARIMA Forecasting ---
st.subheader("Interactive Per Class ARIMA Forecasting")
# Let user select a species (class) to generate its ARIMA forecast
unique_species = data['class_name'].unique()
selected_arima_class = st.selectbox("Select a Class for ARIMA Forecast", unique_species)
if st.button("Generate ARIMA Forecast for Selected Class"):
    try:
        df_species = data[data['class_name'] == selected_arima_class].copy()
        df_species['date'] = pd.to_datetime(df_species['timestamp']).dt.date
        ts_species = df_species.groupby('date')['animal_count'].sum().dropna()
        if len(ts_species) < 10:
            st.write(f"Not enough data for ARIMA forecast for species: {selected_arima_class}")
        else:
            from statsmodels.tsa.arima.model import ARIMA
            model_arima_species = ARIMA(ts_species, order=(1,1,1)).fit()
            forecast_species = model_arima_species.get_forecast(steps=10)
            forecast_index_species = pd.date_range(pd.to_datetime(ts_species.index[-1]), periods=10, freq='D')
            forecast_series_species = forecast_species.predicted_mean
            conf_int_species = forecast_species.conf_int()
            fig_species = go.Figure()
            fig_species.add_trace(go.Scatter(x=ts_species.index, y=ts_species, mode='lines', name='Observed'))
            fig_species.add_trace(go.Scatter(x=forecast_index_species, y=forecast_series_species, mode='lines', name='Forecast'))
            fig_species.add_trace(go.Scatter(
                x=forecast_index_species, y=conf_int_species.iloc[:, 0],
                mode='lines', line=dict(color='gray'), name='Lower CI'))
            fig_species.add_trace(go.Scatter(
                x=forecast_index_species, y=conf_int_species.iloc[:, 1],
                mode='lines', line=dict(color='gray'), name='Upper CI'))
            fig_species.update_layout(
                title=f"ARIMA Forecast for {selected_arima_class}",
                xaxis_title="Date",
                yaxis_title="Animal Count",
                template='plotly_white'
            )
            st.plotly_chart(fig_species, key=f"arima_{selected_arima_class}")
    except Exception as e:
        st.error(f"Error forecasting for species {selected_arima_class}: {e}")

# --- Note ---
# The average annotations ARIMA forecast is still generated in the earlier ARIMA section.
# The per-class ARIMA forecast is generated in the interactive section.
# The ARIMA forecasts are generated for the next 10 days.

# --- Additional Diversity Indices ---
st.subheader("Additional Diversity Indices")

# Calculate Simpson's Diversity Index
def simpsons_diversity(counts):
    N = sum(counts)
    return 1 - sum((n/N) * ((n-1)/(N-1)) for n in counts if n > 0)

# Calculate Pielou's Evenness
def pielou_evenness(counts):
    shannon = -sum((n/sum(counts)) * np.log(n/sum(counts)) for n in counts if n > 0)
    return shannon / np.log(len([n for n in counts if n > 0]))

# Calculate Chao1 Richness
def chao1_richness(counts):
    singletons = sum(1 for n in counts if n == 1)
    doubletons = sum(1 for n in counts if n == 2)
    observed_richness = len([n for n in counts if n > 0])
    if doubletons == 0:
        return observed_richness
    return observed_richness + (singletons * singletons)/(2 * doubletons)

# Create time series of diversity indices
diversity_metrics = pd.DataFrame(index=species_pivot.index)
diversity_metrics['simpsons'] = species_pivot.apply(simpsons_diversity, axis=1)
diversity_metrics['pielou'] = species_pivot.apply(pielou_evenness, axis=1)
diversity_metrics['chao1'] = species_pivot.apply(chao1_richness, axis=1)

fig_div = go.Figure()
for col in diversity_metrics.columns:
    fig_div.add_trace(go.Scatter(x=diversity_metrics.index, y=diversity_metrics[col], 
                                mode='lines', name=col.capitalize()))
fig_div.update_layout(title="Diversity Indices Over Time", xaxis_title="Date", yaxis_title="Index Value")
st.plotly_chart(fig_div)

# --- Community Analysis ---
st.subheader("Community Analysis")

# Beta Diversity over time
from scipy.spatial.distance import pdist
beta_div = pdist(species_pivot, metric='braycurtis')
beta_matrix = squareform(beta_div)
fig_beta = px.imshow(beta_matrix, 
                     labels=dict(x="Time Point", y="Time Point", color="Bray-Curtis Dissimilarity"),
                     title="Beta Diversity (Bray-Curtis) Between Time Points")
st.plotly_chart(fig_beta)

# Species turnover rate
def calculate_turnover(df):
    turnover = []
    for i in range(1, len(df)):
        prev = set(df.iloc[i-1][df.iloc[i-1] > 0].index)
        curr = set(df.iloc[i][df.iloc[i] > 0].index)
        gain = len(curr - prev)
        loss = len(prev - curr)
        turnover.append((gain + loss) / 2)
    return turnover

turnover_rates = calculate_turnover(species_pivot)
fig_turnover = go.Figure()
fig_turnover.add_trace(go.Scatter(x=species_pivot.index[1:], y=turnover_rates, 
                                 mode='lines', name='Turnover Rate'))
fig_turnover.update_layout(title="Species Turnover Rate", 
                          xaxis_title="Date", yaxis_title="Turnover Rate")
st.plotly_chart(fig_turnover)

# Rank-abundance curves
st.subheader("Rank-Abundance Curves")
# Calculate mean abundance for each species
mean_abundances = species_pivot.mean().sort_values(ascending=False)
fig_rank = go.Figure()
fig_rank.add_trace(go.Scatter(x=list(range(1, len(mean_abundances) + 1)), 
                             y=mean_abundances, mode='lines+markers'))
fig_rank.update_layout(title="Rank-Abundance Curve",
                      xaxis_title="Rank", yaxis_title="Mean Abundance (log scale)",
                      yaxis_type="log")
st.plotly_chart(fig_rank)

# --- Temporal Patterns ---
st.subheader("Temporal Patterns")

# Wavelet Analysis
from scipy import signal
st.write("Wavelet Analysis")
# Perform continuous wavelet transform on total annotations
total_annotations = species_pivot.sum(axis=1)
widths = np.arange(1, 31)  # Range of periods to analyze

# Create a simple wavelet function (Mexican Hat / Ricker wavelet)
def ricker_wavelet(points, a):
    vec = np.arange(0, points) - (points - 1.0) / 2
    return (1 - (vec * vec) / (a * a)) * np.exp(-(vec * vec) / (2 * a * a))

# Perform the wavelet transform
cwtmatr = np.zeros((len(widths), len(total_annotations)))
for ind, width in enumerate(widths):
    wavelet = ricker_wavelet(min(10 * width, len(total_annotations)), width)
    cwtmatr[ind, :] = np.convolve(total_annotations, wavelet, mode='same')

fig_wavelet = px.imshow(np.abs(cwtmatr), 
                       labels=dict(x="Time", y="Scale", color="Power"),
                       title="Wavelet Transform of Total Annotations")
st.plotly_chart(fig_wavelet)

# Change Point Detection
st.write("Change Point Detection")
from ruptures import Binseg

# Detect change points in total annotations
change_detector = Binseg(model="l2").fit(total_annotations.values.reshape(-1, 1))
change_points = change_detector.predict(n_bkps=3)
fig_change = go.Figure()

# Plot the total annotations line
fig_change.add_trace(go.Scatter(
    x=total_annotations.index,
    y=total_annotations,
    mode='lines',
    name='Total Annotations'
))

# Add vertical lines for change points
for cp in change_points[:-1]:
    cp_date = total_annotations.index[cp]
    # Use correct shape properties (x0, x1, y0, y1)
    fig_change.add_shape(
        type="line",
        x=cp_date,
        x1=cp_date,
        y0=0,
        y1=total_annotations.max(),
        line=dict(
            color="red",
            width=2,
            dash="dash"
        )
    )
    # Add annotation with correct position
    fig_change.add_annotation(
        x=cp_date,
        y=total_annotations.max(),
        text="Change Point",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

fig_change.update_layout(
    title="Change Points in Community Composition",
    xaxis_title="Date",
    yaxis_title="Total Annotations",
    showlegend=True
)

st.plotly_chart(fig_change)

# Change Point Detection and Community Analysis
st.subheader("Detailed Change Point Analysis")

try:
    # Detect change points using Ruptures
    change_detector = Binseg(model="l2").fit(total_annotations.values.reshape(-1, 1))
    change_points = change_detector.predict(n_bkps=3)
    
    # Create segments based on change points
    segments = []
    start_idx = 0
    dates = total_annotations.index
    
    for cp in change_points[:-1]:
        segment = {
            'start_date': dates[start_idx],
            'end_date': dates[cp],
            'data': species_pivot.iloc[start_idx:cp]
        }
        segments.append(segment)
        start_idx = cp
    
    # Add the last segment
    segments.append({
        'start_date': dates[start_idx],
        'end_date': dates[-1],
        'data': species_pivot.iloc[start_idx:]
    })
    
    # Analyze each segment
    st.write("### Community Segments Analysis")
    
    for i, segment in enumerate(segments):
        st.write(f"\n#### Segment {i+1}: {segment['start_date'].strftime('%Y-%m-%d')} to {segment['end_date'].strftime('%Y-%m-%d')}")
        
        # Calculate segment statistics
        segment_data = segment['data']
        total_observations = segment_data.sum().sum()
        species_present = (segment_data.sum() > 0).sum()
        dominant_species = segment_data.sum().idxmax()
        
        # Calculate diversity metrics for segment
        shannon = -np.sum((segment_data.sum() / total_observations) * 
                         np.log(segment_data.sum() / total_observations))
        simpson = 1 - np.sum((segment_data.sum() * (segment_data.sum() - 1)) / 
                            (total_observations * (total_observations - 1)))
        
        # Display basic statistics
        st.write("**Basic Statistics:**")
        st.write(f"- Duration: {len(segment_data)} days")
        st.write(f"- Total observations: {total_observations}")
        st.write(f"- Number of species: {species_present}")
        st.write(f"- Dominant species: {dominant_species}")
        st.write(f"- Shannon diversity: {shannon:.3f}")
        st.write(f"- Simpson diversity: {simpson:.3f}")
        
        # Species composition
        st.write("\n**Species Composition:**")
        species_comp = segment_data.sum().sort_values(ascending=False)
        species_percent = (species_comp / total_observations * 100).round(2)
        
        # Create composition DataFrame
        comp_df = pd.DataFrame({
            'Count': species_comp,
            'Percentage': species_percent,
            'Rank': range(1, len(species_comp) + 1)
        })
        st.dataframe(comp_df)
        
        # Visualize segment composition
        fig_comp = px.pie(values=species_comp, names=species_comp.index,
                         title=f"Species Composition - Segment {i+1}")
        st.plotly_chart(fig_comp)
        
        # Statistical tests between adjacent segments
        if i > 0:
            prev_segment = segments[i-1]['data']
            st.write("\n**Statistical Comparison with Previous Segment:**")
            
            # Perform PERMANOVA test
            try:
                from scipy.stats import f_oneway
                
                # Prepare data for PERMANOVA
                segment1_data = prev_segment.values.flatten()
                segment2_data = segment_data.values.flatten()
                
                # Perform F-test
                f_stat, p_val = f_oneway(segment1_data, segment2_data)
                st.write(f"PERMANOVA test: F-statistic = {f_stat:.3f}, p-value = {p_val:.3f}")
                
                # Interpretation
                if p_val < 0.05:
                    st.write(" The communities are significantly different (p < 0.05)")
                else:
                    st.write(" No significant difference between communities (p >= 0.05)")
                
                # Calculate and display community turnover
                species_before = set(prev_segment.columns[prev_segment.sum() > 0])
                species_after = set(segment_data.columns[segment_data.sum() > 0])
                
                appeared = species_after - species_before
                disappeared = species_before - species_after
                
                st.write("\n**Community Turnover:**")
                if len(appeared) > 0:
                    st.write("Species appeared:", ', '.join(appeared))
                if len(disappeared) > 0:
                    st.write("Species disappeared:", ', '.join(disappeared))
                
                # Calculate similarity indices
                jaccard = len(species_before & species_after) / len(species_before | species_after)
                st.write(f"Jaccard similarity index: {jaccard:.3f}")
                
            except Exception as e:
                st.error(f"Error performing statistical tests: {e}")
        
        # Trend analysis within segment
        st.write("\n**Trend Analysis:**")
        try:
            # Mann-Kendall trend test
            import pymannkendall as mk
            segment_total = segment_data.sum(axis=1)
            trend_result = mk.original_test(segment_total)
            st.write(f"Mann-Kendall trend test: {trend_result.trend} (p-value: {trend_result.p:.3f})")
            
            # Plot trend
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=segment_data.index,
                y=segment_total,
                mode='lines+markers',
                name='Observations'
            ))
            
            # Add trend line
            z = np.polyfit(range(len(segment_total)), segment_total, 1)
            p = np.poly1d(z)
            fig_trend.add_trace(go.Scatter(
                x=segment_data.index,
                y=p(range(len(segment_total))),
                mode='lines',
                name='Trend',
                line=dict(dash='dash')
            ))
            
            fig_trend.update_layout(
                title=f"Temporal Trend - Segment {i+1}",
                xaxis_title="Date",
                yaxis_title="Total Observations"
            )
            st.plotly_chart(fig_trend)
            
        except Exception as e:
            st.error(f"Error performing trend analysis: {e}")
    
    # Overall change point significance
    st.write("### Overall Change Point Significance")
    
    # Calculate BIC scores for different numbers of change points
    bic_scores = []
    for n_bkps in range(1, 6):
        score = change_detector.score(n_bkps=n_bkps)
        bic_scores.append(score)
    
    # Plot BIC scores
    fig_bic = go.Figure()
    fig_bic.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=bic_scores,
        mode='lines+markers'
    ))
    fig_bic.update_layout(
        title="Model Selection - BIC Scores",
        xaxis_title="Number of Change Points",
        yaxis_title="BIC Score"
    )
    st.plotly_chart(fig_bic)
    
    # Recommend optimal number of change points
    optimal_cp = np.argmin(bic_scores) + 1
    st.write(f"Optimal number of change points (based on BIC): {optimal_cp}")

except Exception as e:
    st.error(f"Error performing detailed change point analysis: {e}")

# --- Overall Change Point Significance ---
st.write("### Overall Change Point Significance")
def compute_total_cost(signal, bkps, cost_func):
    total_cost = 0
    start = 0
    for bp in bkps:
        total_cost += cost_func.error(signal, start, bp)
        start = bp
    return total_cost

cost_scores = []
for k in range(1, 6):
    bkps = change_detector.predict(n_bkps=k)
    score = compute_total_cost(ts.values, bkps, change_detector.cost)
    cost_scores.append(score)

fig_cost = go.Figure()
fig_cost.add_trace(go.Scatter(x=list(range(1, 6)), y=cost_scores, mode='lines+markers'))
fig_cost.update_layout(
    title="Total Cost Scores vs. Number of Change Points",
    xaxis_title="Number of Change Points",
    yaxis_title="Total Cost Score"
)
st.plotly_chart(fig_cost)

optimal_cp = cost_scores.index(min(cost_scores)) + 1
st.write(f"Optimal number of change points (based on minimal total cost): {optimal_cp}")

# Time-lag Analysis
st.write("Time-lag Analysis")
# Calculate community dissimilarity at different time lags
max_lag = 10
lag_dissim = []
for lag in range(1, max_lag + 1):
    lagged_dist = np.mean([pdist([species_pivot.iloc[i], species_pivot.iloc[i+lag]], 
                                'braycurtis')[0] 
                          for i in range(len(species_pivot)-lag)])
    lag_dissim.append(lagged_dist)

fig_lag = go.Figure()
fig_lag.add_trace(go.Scatter(x=list(range(1, max_lag + 1)), y=lag_dissim, 
                            mode='lines+markers'))
fig_lag.update_layout(title="Time-lag Analysis",
                     xaxis_title="Time Lag (days)", 
                     yaxis_title="Mean Bray-Curtis Dissimilarity")
st.plotly_chart(fig_lag)

# Species Co-occurrence Network
st.write("Species Co-occurrence Network")
import networkx as nx
# Calculate species correlations
species_corr = species_pivot.corr()
# Create network from significant correlations
G = nx.Graph()
for i in range(len(species_corr)):
    for j in range(i+1, len(species_corr)):
        if abs(species_corr.iloc[i,j]) > 0.5:  # Correlation threshold
            G.add_edge(species_corr.index[i], species_corr.index[j], 
                      weight=species_corr.iloc[i,j])

# Convert network to plotly figure
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

fig_network = go.Figure()
fig_network.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                line=dict(width=0.5), hoverinfo='none'))
fig_network.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                text=list(G.nodes()), textposition="top center"))
fig_network.update_layout(title="Species Co-occurrence Network",
                         showlegend=False)
st.plotly_chart(fig_network)

# --- Species Co-occurrence Network ---
st.subheader("Species Co-occurrence Network")

# Calculate correlations between species
species_corr = species_pivot.corr()

# Create network
G = nx.Graph()

# Add edges for significantly correlated species pairs
correlation_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5, 0.1)
for i in range(len(species_corr.columns)):
    for j in range(i+1, len(species_corr.columns)):
        corr = abs(species_corr.iloc[i,j])
        if corr >= correlation_threshold:
            species1 = species_corr.columns[i]
            species2 = species_corr.columns[j]
            G.add_edge(species1, species2, weight=corr)

if len(G.edges()) == 0:
    st.warning("No species pairs meet the correlation threshold. Try lowering the threshold.")
else:
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"Correlation: {edge[2]['weight']:.2f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            size=10,
            line_width=2))

    # Create the figure
    fig_network = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title="Species Co-occurrence Network",
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )
    
    st.plotly_chart(fig_network)
    
    # Add network statistics
    st.write("Network Statistics:")
    st.write(f"Number of species (nodes): {G.number_of_nodes()}")
    st.write(f"Number of connections (edges): {G.number_of_edges()}")
    if G.number_of_nodes() > 1:
        st.write(f"Network density: {nx.density(G):.3f}")
        st.write(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")

# --- Anomaly Detection ---
st.subheader("Anomaly Detection")
window_size = st.slider("Rolling Window Size (days)", min_value=3, max_value=30, value=7, step=1)
z_threshold = st.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

# Aggregate data by date
daily_counts = data.copy()
daily_counts['date'] = pd.to_datetime(daily_counts['timestamp']).dt.date
daily_agg = daily_counts.groupby('date')['animal_count'].sum().reset_index()
daily_agg['date'] = pd.to_datetime(daily_agg['date'])
daily_agg.sort_values('date', inplace=True)

# Compute rolling statistics
daily_agg['rolling_mean'] = daily_agg['animal_count'].rolling(window=window_size, min_periods=1, center=True).mean()
daily_agg['rolling_std'] = daily_agg['animal_count'].rolling(window=window_size, min_periods=1, center=True).std().fillna(0)

# Calculate z-scores and flag anomalies
daily_agg['z_score'] = (daily_agg['animal_count'] - daily_agg['rolling_mean']) / daily_agg['rolling_std']
anomalies = daily_agg[abs(daily_agg['z_score']) > z_threshold]

# Plotting the results
fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['animal_count'],
                              mode='lines', name='Daily Count'))
fig_anom.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['rolling_mean'],
                              mode='lines', name='Rolling Mean'))
if not anomalies.empty:
    fig_anom.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['animal_count'],
                                  mode='markers', marker=dict(color='red', size=10),
                                  name='Anomaly'))
fig_anom.update_layout(title="Anomaly Detection in Daily Animal Counts",
                      xaxis_title="Date", yaxis_title="Animal Count",
                      template='plotly_white')
st.plotly_chart(fig_anom)
st.write("Anomalous Days Detected:")
st.write(anomalies[['date', 'animal_count', 'z_score']])

# --- Plain Language Summary ---
st.markdown("## Plain Language Summary")
summary_text = f"""
After analyzing the data across multiple dimensions, several key observations emerge:

1. **Overall Trends and Fluctuations:**  
   The daily total animal counts exhibit periods of significant variability. The rolling average computed over a window of {st.session_state.get('window_size', 7) if 'window_size' in st.session_state else 7} days smooths short-term noise, while days with z-scores above the chosen threshold (set at {st.session_state.get('z_threshold', 2.5) if 'z_threshold' in st.session_state else 2.5}) have been flagged as anomalies. These anomalies may indicate unusual events or shifts in underlying behaviors.
"""
st.markdown(summary_text)
