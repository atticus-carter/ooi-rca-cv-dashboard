import os
import glob
import pandas as pd
import duckdb
import streamlit as st  # Import Streamlit
from scripts.model_generation import generate_predictions, model_urls # Replace with your path
import re

# --- Camera Names ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# --- Configuration (Adjust these) ---
bucket_name = "ooi-rca-cv-data"  # Your GCS bucket name
year_month = "2021-08"  # The year and month of the data

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

# --- Create Connection Object ---
try:
    from st_files_connection import FilesConnection
    conn = st.connection('gcs', type=FilesConnection)
    print("Successfully created GCS connection.")
except Exception as e:
    st.error(f"Error creating GCS connection: {e}")
    st.stop()

# Connect to DuckDB (in-memory for this example)
con = duckdb.connect(database=':memory:', read_only=False)

# --- Camera Selection ---
for camera_id in camera_names:  # Changed to iterate over camera_names list

    # --- Define data GCS path
    data_gcs_path = f"gs://{bucket_name}/{camera_id}/data_{year_month}"

    # --- Check if data exists in gcs ---
    try:
        blobs = conn.ls(f"{bucket_name}/{camera_id}/data_{year_month}/")  # Use conn.ls for listing
        has_data = len(blobs) > 0  # Check if the directory has files
    except Exception as e:
        st.error(f"Error accessing GCS: {e}")
        has_data = False

    # --- Check if data exists in gcs ---
    if has_data:
        # --- Set local directory
        local_image_dir = f"{camera_id}_data_{year_month}"  # Use data_YYYY-MM format

        # --- Check if the Parquet File has been created. If not create it ---
        parquet_file_path = f"{local_image_dir}/predictions.parquet"

        if not os.path.exists(parquet_file_path):
            if not os.path.exists(local_image_dir):
                os.makedirs(local_image_dir)
            # 1. List image files
            #image_files = glob.glob(os.path.join(local_image_dir, "*.jpg")) # Adjust for .png, etc.

            #Check what's in GCS before generating local file
            try:
                gcs_files = conn.ls(f"{bucket_name}/{camera_id}/data_{year_month}/")
                image_files = [f for f in gcs_files if f.endswith(".jpg")]
                if not image_files:
                    st.warning(f"No images found in GCS for camera {camera_id}.")
                    continue
            except Exception as e:
                st.error(f"Error listing GCS files: {e}")
                continue

            #Download images from GCS and place them in the directory
            for image_file in image_files:
                try:
                    local_file = os.path.join(local_image_dir, image_file.split("/")[-1])
                    conn.get(image_file, output_format="path", filename=local_file)
                    print(f"File was succesfully downloaded to {local_file}")
                except Exception as e:
                    st.error(f"Failed to download {image_file} from GCS: {e}")

            image_files = glob.glob(os.path.join(local_image_dir, "*.jpg"))  # Now read local directory for image
            if not image_files:
                st.warning(f"No images found in local directory: {local_image_dir}. Please verify that image directory was correctly loaded in")
                continue  # Skip to the next camera

            # 2. Create a list to hold the data
            data = []

            # 3. Iterate through the images, generate predictions, and create rows for the Parquet file
            for image_file in image_files:
                image_name = os.path.basename(image_file)
                timestamp = extract_timestamp_from_filename(image_name) # Extract timestamp from filename
                if timestamp is None:
                    print(f"Warning: Could not extract timestamp from filename {image_name}. Skipping.")
                    continue
                image_gcs_path = f"{data_gcs_path}/{image_name}"
                predictions = generate_predictions(image_file, "SHR_DSCAM") # Replace with your desired model

                for prediction in predictions:
                    data.append({
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "image_path": image_gcs_path,
                        "class_id": prediction["class_id"],
                        "class_name": prediction["class_name"],
                        "bbox_x": prediction["bbox"][0],
                        "bbox_y": prediction["bbox"][1],
                        "bbox_width": prediction["bbox"][2],
                        "bbox_height": prediction["bbox"][3],
                        "confidence": prediction["confidence"],
                    })

            # 4. Create a Pandas DataFrame from the data
            if data: # Check to ensure data is not empty, skip to next if so
                df = pd.DataFrame(data)

                # 5. Define the GCS path for the Parquet file
                parquet_gcs_path = f"{data_gcs_path}/predictions.parquet"

                # 6. Save Parquet directly to GCS
                try:
                    #If writing out to GCS does not work, perhaps create a file and then copy. This ensures that the file exists
                    df.to_parquet(parquet_file_path, engine='fastparquet')
                    conn.upload(parquet_file_path, parquet_gcs_path)
                    os.remove(parquet_file_path)

                except Exception as e:
                    print(f"Failed to upload parquet to gcs, however, the code now saves it locally. Error: {e}")

                    print(f"Parquet file saved to {parquet_gcs_path}")
                # 7. Save Parquet to local directory for streamlit connection
                #df.to_parquet(parquet_file_path, engine='fastparquet') # Important:  Install fastparquet!
                #print(f"Parquet file saved to {parquet_file_path}")
            else:
                st.warning(f"No predictions generated for camera {camera_id}. Skipping")
                continue # Skip to the next camera
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
        st.warning(f"No data found in GCS for camera: {camera_id}")

# --- Dataview Button ---
camera_option = st.selectbox("Select Camera for Detailed View", [cam for cam in camera_names if os.path.exists(f"{cam}_data_{year_month}/predictions.parquet")])  # Use camera_names list

if st.button("Go to Dataview"):
    st.session_state.camera = camera_option
    st.session_state.selected_model = selected_model
    st.session_state.bucket_name = bucket_name # Passing the bucket name
    st.switch_page("pages/dataview.py") # You would need to create dataview.py in a pages directory

con.close()
