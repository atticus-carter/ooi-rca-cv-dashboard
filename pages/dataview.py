import streamlit as st
import duckdb
import pandas as pd
from google.cloud import storage  # For accessing GCS images
from PIL import Image
import io
import cv2  # OpenCV
import numpy as np
import hashlib

# --- Camera Names ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# Function to generate color dynamically
def generate_color(class_name):
    """Generates a unique BGR color for a given class name."""
    # Use hashlib to generate a unique (but deterministic) number for each class
    hash_object = hashlib.md5(class_name.encode())
    hex_digest = hash_object.hexdigest()

    # Take the first 6 characters of the hex digest and convert them to an integer
    color_int = int(hex_digest[:6], 16)

    # Extract BGR components
    blue = (color_int & 0xFF)
    green = ((color_int >> 8) & 0xFF)
    red = ((color_int >> 16) & 0xFF)

    return (blue, green, red)  # BGR format


# --- Dataview Page ---
if 'camera' not in st.session_state:
    st.write("Please select camera from main page")
else:
    st.title(f"Dataview - {st.session_state.camera}")

    # --- Load Variables from Session State ---
    bucket_name = st.session_state.get("bucket_name")  # Ensure bucket_name is passed
    camera_id = st.session_state.camera
    selected_model = st.session_state.get("selected_model", "SHR_DSCAM")
    year_month = "2025-01"  # Test data is in January 2025

    st.write(f"Using model: {selected_model}")

    # --- Get the parquet path ---
    parquet_gcs_path = f"gs://{bucket_name}/{camera_id}/{year_month}/predictions.parquet"

    # Connect to DuckDB (in-memory for this example)
    con = duckdb.connect(database=':memory:', read_only=False)

    # Install and configure the httpfs extension for GCS access
    con.sql("INSTALL httpfs;")
    con.sql("LOAD httpfs;")
    con.sql(f"SET s3_endpoint='storage.googleapis.com';")
    con.sql(f"SET s3_region='auto';")
    #You may need these lines depending on your duckdb version and gcloud credentials setup
    #con.sql("SET s3_access_key_id='';")
    #con.sql("SET s3_secret_access_key='';")
    #con.sql("SET s3_session_token='';")


    # Load the dataframe from the parquet file
    query = f"SELECT * FROM '{parquet_gcs_path}'"
    try:
        df = con.execute(query).fetchdf()
        if df.empty:
            st.warning("No data found for the selected camera and time period.")
        else:
            # Convert timestamp to datetime objects if it's not already
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp
            df = df.sort_values(by='timestamp')

            # --- Time Selection with Slider ---
            min_ts = df['timestamp'].min()
            max_ts = df['timestamp'].max()

            time_selection = st.slider(
                "Select Timestamp:",
                min_value=min_ts,
                max_value=max_ts,
                value=min_ts  # Initial Value
            )

            # Filter the DataFrame to get the closest matching record
            closest_record = df.iloc[(df['timestamp'] - time_selection).abs().argsort()[:1]]

            # Load the closest image and get its path
            image_gcs_path = closest_record['image_path'].iloc[0]
            st.write(f"Using image: {image_gcs_path}")

            # Connect to the GCS
            client = storage.Client()
            bucket = client.bucket(st.session_state.bucket_name)

            # Extract the image from the GCS
            try:
                blob = bucket.blob(image_gcs_path.replace(f"gs://{st.session_state.bucket_name}/", ""))
                image_bytes = blob.download_as_bytes()
                image = Image.open(io.BytesIO(image_bytes))
                img_width, img_height = image.size
                image_np = np.array(image)  # Convert PIL Image to NumPy array
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR

            except Exception as e:
                st.error(f"Error loading image: {e}")
                image_cv = None


            # Draw Bounding Boxes
            if image_cv is not None:
                st.write("Displaying bounding boxes...")
                # Get predictions for the selected image
                predictions = closest_record.to_dict('records')

                # --- Draw Bounding Boxes ---
                for prediction in predictions:
                    # Extract bounding box coordinates
                    x_center = prediction['bbox_x']
                    y_center = prediction['bbox_y']
                    box_width = prediction['bbox_width']
                    box_height = prediction['bbox_height']
                    confidence = prediction['confidence']
                    class_name = prediction['class_name']

                    # Convert normalized coordinates to pixel coordinates
                    x1 = int((x_center - box_width / 2) * img_width)
                    y1 = int((y_center - box_height / 2) * img_height)
                    x2 = int((x_center + box_width / 2) * img_width)
                    y2 = int((y_center + box_height / 2) * img_height)


                    # Dynamically generate color for the class
                    color = generate_color(class_name)

                    # Draw the bounding box and label
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


                # Convert back to RGB for Streamlit
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption=f"Image with Bounding Boxes from {time_selection}")


            # --- Download Predictions ---
            if st.button("Download Predictions as CSV"):
                # Convert the closest_record (which is already a DataFrame) to CSV
                csv_data = closest_record.to_csv(index=False)

                st.download_button(
                    label="Download",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Error querying DuckDB: {e}")

    con.close()
