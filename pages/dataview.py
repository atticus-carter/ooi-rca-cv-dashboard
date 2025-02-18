import streamlit as st
import duckdb
import pandas as pd
import re
import os
from scripts.model_generation import model_urls, generate_predictions  # Import generate_predictions
import io
import cv2  # opencv
import boto3
from PIL import Image  # Import Pillow for image handling

# --- Camera Names ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# --- Dataview Page ---
if 'camera' not in st.session_state:
    st.write("Please select camera from main page")
else:
    st.title(f"Dataview - {st.session_state.camera}")

    # --- Load variables from Session State ---
    bucket_name = st.session_state.get("bucket_name")
    camera_id = st.session_state.camera
    selected_model = st.session_state.get("selected_model", "SHR_DSCAM")
    year_month = "2021-08"

    # --- AWS Authentication ---
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["connections.s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["connections.s3"]["aws_secret_access_key"],
            region_name=st.secrets["connections.s3"]["region_name"],
        )
    except Exception as e:
        st.error(f"Error during AWS authentication: {e}")
        st.stop()

    try:
        parquet_file_path = f"{camera_id}_data_{year_month}/predictions.parquet"
        df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        st.error(f"Error loading Parquet file: {e}")
        st.stop()

    # --- Model Selection ---
    available_models = list(model_urls.keys())
    selected_model = st.selectbox("Select Model", available_models, index=available_models.index(selected_model))

    # --- Class Name Filtering ---
    unique_class_names = df['class_name'].unique().tolist()
    selected_class = st.selectbox("Filter by Class", ['All'] + unique_class_names)

    if selected_class != 'All':
        df = df[df['class_name'] == selected_class]

    # --- Pagination ---
    items_per_page = 10
    total_items = len(df)
    num_pages = (total_items + items_per_page - 1) // items_per_page
    current_page = st.number_input("Page", min_value=1, max_value=num_pages, value=1)
    start_index = (current_page - 1) * items_per_page
    end_index = min(start_index + items_per_page, total_items)
    df_page = df.iloc[start_index:end_index]

    # --- Display Images with Bounding Boxes ---
    for index, row in df_page.iterrows():
        image_path = row['image_path']
        class_name = row['class_name']
        confidence = row['confidence']
        bbox_x, bbox_y, bbox_width, bbox_height = row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']

        try:
            # Download the image from S3
            image_name = os.path.basename(image_path)
            local_image_path = os.path.join(f"{camera_id}_data_{year_month}", image_name)

            if not os.path.exists(local_image_path):
                s3_client.download_file(bucket_name, f"{camera_id}/data_{year_month}/{image_name}", local_image_path)

            # Open the image using PIL
            img = Image.open(local_image_path)
            img_width, img_height = img.size

            # Calculate bounding box coordinates
            x1 = int((bbox_x - bbox_width / 2) * img_width)
            y1 = int((bbox_y - bbox_height / 2) * img_height)
            x2 = int((bbox_x + bbox_width / 2) * img_width)
            y2 = int((bbox_y + bbox_height / 2) * img_height)

            # Draw bounding box and label on the image using OpenCV
            img_cv = cv2.imread(local_image_path)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert back to PIL image for Streamlit
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

            st.image(img_pil, caption=f"Class: {class_name}, Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Error processing image {image_path}: {e}")
