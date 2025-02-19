import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px

# --- Page Title ---
st.title("Timeseries Data View")

# --- Camera Selection ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]
selected_camera = st.selectbox("Select Camera", camera_names)

# --- List Available CSV Files ---
base_dir = os.path.join("timeseries", selected_camera)
st.write(f"Searching for CSV files in: {base_dir}")  # Debugging information

csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
st.write(f"Found CSV files: {csv_files}")  # Debugging information

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
    df['source_file'] = csv_file  # Add a column to identify the source file
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

# --- Granularity Selection ---
granularity = st.selectbox("Select Granularity", ["Hourly", "Daily", "Monthly"])

if granularity == "Hourly":
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data.set_index('timestamp', inplace=True)
    
    # Separate numeric and non-numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    non_numeric_data = data.select_dtypes(exclude=['number'])
    
    # Resample numeric data and calculate the mean
    resampled_numeric_data = numeric_data.resample('H').mean()
    
    # Merge the resampled numeric data with the non-numeric data
    resampled_data = pd.merge(resampled_numeric_data, non_numeric_data.reset_index(), left_index=True, right_on='timestamp', how='left')
    data = resampled_data.reset_index()

elif granularity == "Daily":
    data['timestamp'] = pd.to_datetime(data['date'])
    data.set_index('timestamp', inplace=True)
    
    # Separate numeric and non-numeric columns
    numeric_data = data.select_dtypes(include(['number']))
    non_numeric_data = data.select_dtypes(exclude(['number']))
    
    # Resample numeric data and calculate the mean
    resampled_numeric_data = numeric_data.resample('D').mean()
    
    # Merge the resampled numeric data with the non-numeric data
    resampled_data = pd.merge(resampled_numeric_data, non_numeric_data.reset_index(), left_index=True, right_on='timestamp', how='left')
    data = resampled_data.reset_index()

elif granularity == "Monthly":
    data['timestamp'] = pd.to_datetime(data['date'])
    data.set_index('timestamp', inplace=True)
    
    # Separate numeric and non-numeric columns
    numeric_data = data.select_dtypes(include(['number']))
    non_numeric_data = data.select_dtypes(exclude(['number']))
    
    # Resample numeric data and calculate the mean
    resampled_numeric_data = numeric_data.resample('M').mean()
    
    # Merge the resampled numeric data with the non-numeric data
    resampled_data = pd.merge(resampled_numeric_data, non_numeric_data.reset_index(), left_index=True, right_on='timestamp', how='left')
    data = resampled_data.reset_index()

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
