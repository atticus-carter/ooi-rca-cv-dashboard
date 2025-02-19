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

# --- Class Filtering ---
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

# Correlation Heatmap for Ecological Metrics
try:
    metrics = species_counts[['total_annotations', 'species_richness', 'shannon_wiener']]
    corr = metrics.corr()
    fig_heatmap = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
    st.plotly_chart(fig_heatmap)
except Exception as e:
    st.error(f"Error generating correlation heatmap: {e}")
