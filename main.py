import os
import glob
import pandas as pd
import duckdb
from google.cloud import storage
import streamlit as st  # Import Streamlit
from scripts.model_generation import generate_predictions, model_urls # Replace with your path

# --- Camera Names ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# --- Configuration (Adjust these) ---
bucket_name = "ooi-rca-cv-data-yourname"  # Your GCS bucket name
#camera_id = "PC01A_CAMDSC102"  # The ID of the camera # Removed this
year_month = "2025-01"  # The year and month of the data
#local_image_dir = "path/to/your/local/camera1/2024-01/images/"  # Path to your local images folder  # Removed this

# Connect to DuckDB (in-memory for this example)
con = duckdb.connect(database=':memory:', read_only=False)

# --- Main Streamlit App ---
st.title("OOI RCA CV Dashboard")

# --- Ensure bucket name is saved in session state ---
if "bucket_name" not in st.session_state:
  st.session_state.bucket_name = bucket_name

# --- Camera Selection ---
for camera_id in camera_names:  # Changed to iterate over camera_names list
    # --- Set local directory
    local_image_dir = f"{camera_id}_{year_month}_images"

    # --- Check if the Parquet File has been created. If not create it ---
    if not os.path.exists(f"{local_image_dir}/predictions.parquet"):
        if not os.path.exists(local_image_dir):
            os.makedirs(local_image_dir)
        # 1. List image files
        image_files = glob.glob(os.path.join(local_image_dir, "*.jpg")) # Adjust for .png, etc.

        # 2. Create a list to hold the data
        data = []

        # 3. Iterate through the images, generate predictions, and create rows for the Parquet file
        for image_file in image_files:
            image_name = os.path.basename(image_file)
            timestamp = pd.to_datetime(year_month + "-" + image_name.split(".")[0].split("_")[-1]) # Creating dummy timestamp. Can be improved
            image_gcs_path = f"gs://{bucket_name}/{camera_id}/{year_month}/images/{image_name}"
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
        df = pd.DataFrame(data)

        # 5. Define the GCS path for the Parquet file
        parquet_gcs_path = f"gs://{bucket_name}/{camera_id}/{year_month}/predictions.parquet"

        # 6. Upload images to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for image_file in image_files:
            image_name = os.path.basename(image_file)
            blob = bucket.blob(f"{camera_id}/{year_month}/images/{image_name}")
            blob.upload_from_filename(image_file)
            print(f"Uploaded {image_file} to gs://{bucket_name}/{camera_id}/{year_month}/images/{image_name}")

        # 7. Save to Parquet directly to GCS
        df.to_parquet(parquet_gcs_path, engine='fastparquet') # Important:  Install fastparquet!
        print(f"Parquet file saved to {parquet_gcs_path}")

        # 8. Save Parquet to local directory for streamlit connection
        df.to_parquet(f"{local_image_dir}/predictions.parquet", engine='fastparquet') # Important:  Install fastparquet!
        print(f"Parquet file saved to {f"{local_image_dir}/predictions.parquet"}")
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
    FROM '{local_image_dir}/predictions.parquet'
    WHERE camera_id = '{camera_id}'
    AND timestamp BETWEEN date('now', '-1 month') AND date('now')
    """
    try:
        df = con.execute(query).fetchdf()
        df['prediction_count'] = 1
        fig = px.line(df, x="timestamp", y="prediction_count", title=f"Prediction Count Over Time - {camera_id}")
        st.plotly_chart(fig)
    except Exception as e:
        st.write(f"Failed to fetch predictions: {e}")


# --- Dataview Button ---
camera_option = st.selectbox("Select Camera for Detailed View", camera_names)  # Use camera_names list

if st.button("Go to Dataview"):
    st.session_state.camera = camera_option
    st.session_state.selected_model = selected_model
    st.session_state.bucket_name = bucket_name # Passing the bucket name
    st.switch_page("pages/dataview.py") # You would need to create dataview.py in a pages directory

con.close()
