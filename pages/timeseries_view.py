import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm

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

# Aggregate species prediction counts by date (using the date portion of timestamp)
df_ecol = data.copy()
df_ecol['date'] = pd.to_datetime(df_ecol['timestamp']).dt.date
species_columns = ['animal_count']
species_counts = df_ecol.groupby('date')[species_columns].sum()

# Calculate ecological metrics
species_counts['total_annotations'] = species_counts.sum(axis=1)
species_counts['species_richness'] = (species_counts[species_columns] > 0).sum(axis=1)
species_proportions = species_counts[species_columns].div(species_counts['total_annotations'], axis=0)
species_counts['shannon_wiener'] = - (species_proportions * np.log(species_proportions)).sum(axis=1)

# Apply a 7-day rolling average to smooth the time series
species_counts['total_annotations_7d_avg'] = species_counts['total_annotations'].rolling(window=7).mean()
species_counts['species_richness_7d_avg'] = species_counts['species_richness'].rolling(window=7).mean()
species_counts['shannon_wiener_7d_avg'] = species_counts['shannon_wiener'].rolling(window=7).mean()

# Create line chart
fig_ecol = go.Figure()
fig_ecol.add_trace(go.Scatter(x=species_counts.index, y=species_counts['total_annotations_7d_avg'], mode='lines', name='Total Annotations (7-day Avg)'))
fig_ecol.add_trace(go.Scatter(x=species_counts.index, y=species_counts['species_richness_7d_avg'], mode='lines', name='Species Richness (7-day Avg)'))
fig_ecol.add_trace(go.Scatter(x=species_counts.index, y=species_counts['shannon_wiener_7d_avg'], mode='lines', name='Shannon-Wiener Index (7-day Avg)'))
fig_ecol.update_layout(
    title='Ecological Metrics Over Time',
    xaxis_title='Date',
    yaxis_title='Metric Value',
    legend_title='Metrics',
    template='plotly_white'
)
st.plotly_chart(fig_ecol)

# --- Fit Statistical Model if Data is Sufficient ---
X = species_counts[['total_annotations_7d_avg', 'species_richness_7d_avg']].dropna()
y = species_counts['shannon_wiener_7d_avg'].dropna()
X, y = X.align(y, join='inner', axis=0)
if X.empty or y.empty or X.shape[0] < 2:
    st.write("Insufficient data for statistical model (need at least 2 data points).")
else:
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.write("Statistical Model Summary:")
    st.write(model.summary())

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

# --- Advanced Ecological Analysis ---

st.subheader("Advanced Ecological Analysis")

# Mann-Kendall Trend Test on Total Annotations (7-day Avg)
try:
    import pymannkendall as mk
    result = mk.original_test(species_counts['total_annotations_7d_avg'].dropna())
    st.write("Mann-Kendall Trend Test on Total Annotations (7d Avg):")
    st.write(f"Trend: {result.trend}, p-value: {result.p}")
except ImportError:
    st.warning("pymannkendall library not installed. Skipping Mann-Kendall trend test.")

# ARIMA Forecasting for Total Annotations (7-day Avg)
try:
    from statsmodels.tsa.arima.model import ARIMA
    ts = species_counts['total_annotations_7d_avg'].dropna()
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
except Exception as e:
    st.error(f"Error performing ARIMA forecasting: {e}")

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
ts_decomp = species_counts['total_annotations'].dropna()  # Using raw total annotations
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
# Use continuous wavelet transform from scipy.signal
wavelet = signal.morlet2  # Using Morlet wavelet instead of deprecated ricker
frequencies = np.linspace(1, 10, 30)
cwtmatr = signal.cwt(total_annotations.values, wavelet, widths)
fig_wavelet = px.imshow(np.abs(cwtmatr), 
                       labels=dict(x="Time", y="Scale", color="Power"),
                       title="Wavelet Transform of Total Annotations")
st.plotly_chart(fig_wavelet)

# Change Point Detection
st.write("Change Point Detection")
from ruptures import Binseg  # You'll need to add 'ruptures' to requirements.txt
# Detect change points in total annotations
change_detector = Binseg(model="l2").fit(total_annotations.values.reshape(-1, 1))
change_points = change_detector.predict(n_bkps=3)
fig_change = go.Figure()
fig_change.add_trace(go.Scatter(x=total_annotations.index, y=total_annotations, 
                               mode='lines', name='Total Annotations'))
for cp in change_points[:-1]:  # Exclude last point
    fig_change.add_vline(x=total_annotations.index[cp], line_dash="dash", 
                        annotation_text="Change Point")
fig_change.update_layout(title="Change Points in Community Composition")
st.plotly_chart(fig_change)

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
