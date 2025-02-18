import os
import glob
import pandas as pd
import duckdb
import streamlit as st
from scripts.model_generation import generate_predictions, model_urls
import re
import boto3
import torch
import time  # Import the time module
import yaml  # Import the YAML module
import subprocess  # Import subprocess

# --- Load Configuration ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        st.error("Configuration file 'config.yaml' is empty or contains invalid YAML.")
        st.stop()
    bucket_name = config["bucket_name"]
    year_month = config["year_month"]
except FileNotFoundError:
    st.error("Configuration file 'config.yaml' not found.  Please create one.")
    st.stop()
except yaml.YAMLError as e:
    st.error(f"Error parsing 'config.yaml': {e}")
    st.stop()
except KeyError as e:
    st.error(f"Missing key in 'config.yaml': {e}")
    st.stop()

# --- Camera Names ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# --- Function to extract timestamp from filename ---
def extract_timestamp_from_filename(filename):
    """Extracts timestamp from the filename."""
    match = re.search(r"(\d{8}T\d{6})", filename)  # Matches the timestamp format YYYYMMDDTHHMMSS
    if match:
        timestamp_str = match.group(1)  # Extract the matched timestamp string
        timestamp = pd.to_datetime(timestamp_str, format="%Y%m%dT%H%M%S")
        return timestamp
    else:
        return None

# --- Main Streamlit App ---
st.title("OOI RCA CV Dashboard")

# --- Ensure bucket name is saved in session state ---
if "bucket_name" not in st.session_state:
    st.session_state.bucket_name = bucket_name

# --- Create S3 Client ---
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["connections.s3"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["connections.s3"]["aws_secret_access_key"],
        region_name=st.secrets["connections.s3"]["region_name"],
    )
    print("Successfully created S3 client.")
except Exception as e:
    st.error(f"Error creating S3 client: {e}. Check your AWS credentials in Streamlit secrets.")
    st.stop()

# Connect to DuckDB (in-memory for this example)
con = duckdb.connect(database=':memory:', read_only=False)

# --- Camera Selection ---
for camera_id in camera_names:
    # --- Set local directory
    local_image_dir = f"{camera_id}_data_{year_month}"

    # --- Define data S3 path
    data_s3_path = f"s3://{bucket_name}/{camera_id}/data_{year_month}"

    # --- check if s3 has data for camera ---
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{camera_id}/data_{year_month}/")
        has_data = "Contents" in response
    except Exception as e:
        st.error(f"Error accessing S3 bucket: {e}.  Check bucket name and permissions.")
        has_data = False

    # --- Check if data exists in s3 ---
    if has_data:
        # --- Set local directory
        local_image_dir = f"{camera_id}_data_{year_month}"

        # --- Check if the Parquet File has been created. If not create it ---
        parquet_file_path = f"{local_image_dir}/predictions.parquet"

        if not os.path.exists(parquet_file_path):
            if not os.path.exists(local_image_dir):
                os.makedirs(local_image_dir)
            # 1. List image files
            image_files = glob.glob(os.path.join(local_image_dir, "*.jpg")) # Adjust for .png, etc.
            if not image_files:
                st.warning(f"No images found in local directory: {local_image_dir}.  Please verify that image directory was correctly loaded in")
                continue  # Skip to the next camera

            # 2. Create a list to hold the data
            data = []

            # 3. Iterate through the images, generate predictions, and create rows for the Parquet file
            image_files_len = len(image_files)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, image_file in enumerate(image_files):
                image_name = os.path.basename(image_file)
                timestamp = extract_timestamp_from_filename(image_name)
                if timestamp is None:
                    print(f"Warning: Could not extract timestamp from filename {image_name}. Skipping.")
                    continue
                image_s3_path = f"{data_s3_path}/{image_name}"
                try:
                    predictions = generate_predictions(image_file, "SHR_DSCAM")
                except Exception as e:
                    st.error(f"Error generating predictions for {image_file}: {e}")
                    continue

                for prediction in predictions:
                    data.append({
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "image_path": image_s3_path,
                        "class_id": prediction["class_id"],
                        "class_name": prediction["class_name"],
                        "bbox_x": prediction["bbox"][0],
                        "bbox_y": prediction["bbox"][1],
                        "bbox_width": prediction["bbox"][2],
                        "bbox_height": prediction["bbox"][3],
                        "confidence": prediction["confidence"],
                    })
                progress = (i + 1) / image_files_len
                status_text.text(f"{progress:.2%} Complete")
                progress_bar.progress(progress)

            # 4. Create a Pandas DataFrame from the data
            if data:
                df = pd.DataFrame(data)

                # 5. Define the S3 path for the Parquet file
                parquet_s3_path = f"{data_s3_path}/predictions.parquet"

                # 6. Upload images to S3 with retry
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=st.secrets["connections.s3"]["aws_access_key_id"],
                    aws_secret_access_key=st.secrets["connections.s3"]["aws_secret_access_key"],
                    region_name=st.secrets["connections.s3"]["region_name"],
                )

                def upload_with_retry(filename, bucket, key, retries=3, delay=2):
                    for attempt in range(retries):
                        try:
                            s3_client.upload_file(filename, bucket, key)
                            print(f"Uploaded {filename} to s3://{bucket}/{key}")
                            return True
                        except Exception as e:
                            print(f"Error uploading {filename} (attempt {attempt + 1}/{retries}): {e}")
                            if attempt < retries - 1:
                                time.sleep(delay)  # Wait before retrying
                            else:
                                st.error(f"Failed to upload {filename} to S3 after multiple retries: {e}")
                                return False

                for image_file in image_files:
                    image_name = os.path.basename(image_file)
                    s3_key = f"{camera_id}/data_{year_month}/{image_name}"
                    upload_with_retry(image_file, bucket_name, s3_key)

                # 7. Save to Parquet directly to S3
                try:
                    df.to_parquet(parquet_s3_path, engine='fastparquet')
                    print(f"Parquet file saved to {parquet_s3_path}")
                except Exception as e:
                    st.error(f"Error saving Parquet file to S3: {e}")

                # 8. Save Parquet to local directory for streamlit connection
                try:
                    df.to_parquet(parquet_file_path, engine='fastparquet')
                    print(f"Parquet file saved to {parquet_file_path}")
                except Exception as e:
                    st.error(f"Error saving Parquet file locally: {e}")
            else:
                st.warning(f"No predictions generated for camera {camera_id}. Skipping")
                continue
        else:
            print("Parquet files found!")

        st.subheader(f"Camera {camera_id}")

        # Add Model Selection dropdown
        available_models = list(model_urls.keys())
        default_model = "SHR_DSCAM"  # Set the default model
        selected_model = st.selectbox(
            f"See predictions from:",
            options=available_models,
            index=available_models.index(default_model),  # Set the default selection
            key=f"model_{camera_id}",
        )

        # Dummy image for now
        st.image("https://placehold.co/1024x1024", caption=f"Latest Image - {camera_id}")

        # Fetch last month's predictions for time series
        query = f"""
        SELECT timestamp, prediction_count
        FROM '{parquet_file_path}'
        WHERE camera_id = '{camera_id}'
        AND timestamp BETWEEN date('now', '-1 month') AND date('now')
        """
        try:
            df = con.execute(query).fetchdf()
            df['prediction_count'] = 1
            #fig = px.line(df, x="timestamp", y="prediction_count", title=f"Prediction Count Over Time - {camera_id}")
            #st.plotly_chart(fig)
        except Exception as e:
            st.write(f"Failed to fetch predictions: {e}")
    else:
        st.warning(f"No data found in S3 for camera: {camera_id}")

# --- Dataview Button ---
camera_option = st.selectbox("Select Camera for Detailed View", [cam for cam in camera_names if os.path.exists(f"{cam}_data_{year_month}/predictions.parquet")])  # Use camera_names list

if st.button("Go to Dataview"):
    st.session_state.camera = camera_option
    st.session_state.selected_model = selected_model
    st.session_state.bucket_name = bucket_name # Passing the bucket name
    st.switch_page("pages/dataview.py") # You would need to create dataview.py in a pages directory

# --- Batch Processing Button ---
if st.button("Run Batch Processing"):
    try:
        subprocess.run(["python", "batch_processing.py"], check=True)
        st.success("Batch processing completed successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Batch processing failed: {e}")

# --- Data Analysis Page Link ---
if st.button("Go to Data Analysis"):
    st.switch_page("pages/data_analysis.py")

# --- Camera Map Page Link ---
if st.button("View Camera Map"):
    st.switch_page("pages/camera_map.py")
