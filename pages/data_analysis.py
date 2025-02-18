import streamlit as st
import pandas as pd
import boto3
import duckdb
import plotly.express as px
import yaml
from datetime import datetime, timedelta

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

bucket_name = config["bucket_name"]
region_name = config["region_name"]

# Camera names
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# --- AWS Authentication ---
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),  # Use environment variables
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name,
    )
except Exception as e:
    st.error(f"Error during AWS authentication: {e}")
    st.stop()

st.title("Data Analysis")

# --- Sidebar ---
st.sidebar.header("Parameters")

# Camera selection
selected_cameras = st.sidebar.multiselect("Select Cameras", camera_names, default=camera_names[:2])

# Date range selection
today = datetime.today()
default_start_date = today - timedelta(days=30)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", today)

# Class selection
all_classes = []
for camera_id in selected_cameras:
    prefix = f"{camera_id}/predictions/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        dates = [d['Prefix'].split('/')[-2] for d in response.get('CommonPrefixes', [])]
        for date in dates:
            parquet_prefix = f"{camera_id}/predictions/{date}/"
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=parquet_prefix)
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    parquet_file = f"s3://{bucket_name}/{obj['Key']}"
                    try:
                        df = pd.read_parquet(parquet_file)
                        all_classes.extend(df['class_name'].unique().tolist())
                    except Exception as e:
                        st.error(f"Error reading parquet file {parquet_file}: {e}")
    except Exception as e:
        st.error(f"Error listing objects in S3: {e}")

unique_classes = list(set(all_classes))
selected_classes = st.sidebar.multiselect("Select Classes", unique_classes, default=unique_classes)

# --- Data Loading ---
@st.cache_data
def load_data(cameras, start, end, classes):
    data = []
    for camera_id in cameras:
        prefix = f"{camera_id}/predictions/"
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
            dates = [d['Prefix'].split('/')[-2] for d in response.get('CommonPrefixes', [])]
            for date in dates:
                date_dt = datetime.strptime(date, '%Y-%m')
                if start.year <= date_dt.year <= end.year and start.month <= date_dt.month <= end.month:
                    parquet_prefix = f"{camera_id}/predictions/{date}/"
                    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=parquet_prefix)
                    for obj in response.get('Contents', []):
                        if obj['Key'].endswith('.parquet'):
                            parquet_file = f"s3://{bucket_name}/{obj['Key']}"
                            try:
                                df = pd.read_parquet(parquet_file)
                                df = df[df['class_name'].isin(classes)]
                                data.append(df)
                            except Exception as e:
                                st.error(f"Error reading parquet file {parquet_file}: {e}")
        except Exception as e:
            st.error(f"Error listing objects in S3: {e}")
    if data:
        df = pd.concat(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= pd.Timestamp(start)) & (df['timestamp'] <= pd.Timestamp(end))]
        return df
    else:
        return None

data_df = load_data(selected_cameras, start_date, end_date, selected_classes)

if data_df is None or data_df.empty:
    st.warning("No data found for the selected parameters.")
    st.stop()

# --- Plotly Plots ---

# 1. Timeline of Prediction Counts
st.subheader("Timeline of Prediction Counts")
timeline_df = data_df.groupby(['camera_id', 'timestamp']).size().reset_index(name='count')
fig_timeline = px.line(timeline_df, x='timestamp', y='count', color='camera_id', title="Prediction Counts Over Time")
st.plotly_chart(fig_timeline)

# 2. Stacked Area Chart of Class Distributions
st.subheader("Stacked Area Chart of Class Distributions")
area_df = data_df.groupby(['camera_id', 'timestamp', 'class_name']).size().reset_index(name='count')
fig_area = px.area(area_df, x='timestamp', y='count', color='class_name', facet_col='camera_id', title="Class Distributions Over Time")
st.plotly_chart(fig_area)

# 3. Stacked Bar Chart of Class Distributions
st.subheader("Stacked Bar Chart of Class Distributions")
bar_df = data_df.groupby(['camera_id', 'timestamp', 'class_name']).size().reset_index(name='count')
fig_bar = px.bar(bar_df, x='timestamp', y='count', color='class_name', facet_col='camera_id', title="Class Distributions Over Time")
st.plotly_chart(fig_bar)

# 4. Linear Regressions on Individual Species Plots
st.subheader("Linear Regressions on Individual Species Plots")
for class_name in selected_classes:
    species_df = data_df[data_df['class_name'] == class_name].groupby('timestamp').size().reset_index(name='count')
    if not species_df.empty:
        fig_species = px.scatter(species_df, x='timestamp', y='count', trendline="ols", title=f"Linear Regression for {class_name}")
        st.plotly_chart(fig_species)

# 5. Comparison Plots Between Different Cameras
st.subheader("Comparison Plots Between Different Cameras")
if len(selected_cameras) > 1:
    comparison_df = data_df.groupby(['camera_id', 'timestamp']).size().reset_index(name='count')
    fig_comparison = px.line(comparison_df, x='timestamp', y='count', color='camera_id', title="Comparison of Prediction Counts Between Cameras")
    st.plotly_chart(fig_comparison)
else:
    st.write("Select more than one camera to generate comparison plots.")
