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

# --- List Available CSV Files ---
base_dir = os.path.join("timeseries", selected_camera)
st.write(f"Searching for CSV files in: {base_dir}")

csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
st.write(f"Found CSV files: {csv_files}")

csv_files = [os.path.relpath(f, base_dir) for f in csv_files]

if not csv_files:
    st.warning("No CSV files found for the selected camera.")
    st.stop()

selected_csvs = st.multiselect("Select CSV Files", csv_files)

if not selected_csvs:
    st.warning("Please select at least one CSV file.")
    st.stop()

# --- Load and Concatenate Selected CSV Files ---
dfs = []
for csv_file in selected_csvs:
    file_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(file_path)
    df['source_file'] = csv_file
    dfs.append(df)

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
    fig = px.area(data, x='timestamp', y='animal_count', color='class_name', title="Stacked Area Chart")
elif plot_type == "Average Confidence":
    fig = px.line(data, x='timestamp', y='confidence', color='class_name', title="Average Confidence Over Time")
st.plotly_chart(fig)

# --- Ecology Metrics ---
st.subheader("Ecology Metrics")
total_counts = data.groupby('class_name')['animal_count'].sum()
st.write("Total Counts by Class:")
st.write(total_counts)
average_confidences = data.groupby('class_name')['confidence'].mean()
st.write("Average Confidence by Class:")
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
