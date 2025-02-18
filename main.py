import os
import glob
import pandas as pd
import duckdb
import streamlit as st
from scripts.model_generation import generate_predictions, model_urls, load_model
import re
import time  # Import the time module
import yaml  # Import the YAML module
import subprocess  # Import subprocess
import logging
import cv2
from PIL import Image
import numpy as np

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Configuration ---
try:
    if not os.path.exists("config.yaml"):
        st.error("Configuration file 'config.yaml' not found.")
        st.stop()

    if os.stat("config.yaml").st_size == 0:
        st.error("Configuration file 'config.yaml' is empty.")
        st.stop()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        st.error("Configuration file 'config.yaml' contains invalid YAML.")
        st.stop()

    year_month = config.get("year_month")

    if not year_month:
        st.error("Missing 'year_month' in 'config.yaml'.")
        st.stop()

except FileNotFoundError:
    st.error("Configuration file 'config.yaml' not found.")
    st.stop()
except yaml.YAMLError as e:
    st.error(f"Error parsing 'config.yaml': {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the configuration: {e}")
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

# --- Color Palette for Bounding Boxes ---
def get_color_for_class(class_name):
    """Generates a unique color for each class name."""
    # Use a hash function to generate a unique integer for each class name
    hash_value = hash(class_name) % 360
    # Map the hash value to a color in the HSL color space
    hue = hash_value
    saturation = 75  # You can adjust the saturation
    lightness = 50  # You can adjust the lightness
    # Convert HSL to BGR (OpenCV uses BGR)
    hsv_color = np.uint8([[[hue, saturation, lightness]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
    return bgr_color

# --- Main Streamlit App ---
st.title("OOI RCA CV Dashboard")

# --- Find Most Recent Image and Display Predictions ---
def display_latest_image_with_predictions(camera_id, selected_model):
    image_dir = os.path.join("images", camera_id, year_month)
    if not os.path.exists(image_dir):
        st.warning(f"No data found in local directory: {image_dir}")
        return None

    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_files:
        st.warning(f"No images found in local directory: {image_dir}")
        return None

    # Find the most recent image
    most_recent_image = max(image_files, key=os.path.getmtime)

    # Generate predictions for the most recent image
    predictions = generate_predictions(most_recent_image, selected_model)

    # Load the image using OpenCV
    img_cv = cv2.imread(most_recent_image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.open(most_recent_image)
    img_width, img_height = img_pil.size

    # Overlay predictions on the image
    for prediction in predictions:
        if len(prediction["bbox"]) < 4:
            logging.warning(f"Skipping prediction due to insufficient bbox elements: {prediction}")
            continue

        class_name = prediction["class_name"]
        confidence = prediction["confidence"]
        bbox_x, bbox_y, bbox_width, bbox_height = prediction["bbox"][0], prediction["bbox"][1], prediction["bbox"][2], prediction["bbox"][3]

        x1 = int((bbox_x - bbox_width / 2) * img_width)
        y1 = int((bbox_y - bbox_height / 2) * img_height)
        x2 = int((bbox_x + bbox_width / 2) * img_width)
        y2 = int((bbox_y + bbox_height / 2) * img_height)

        color = get_color_for_class(class_name)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img_cv

# --- Display Images in a 2x2 Grid ---
cols = st.columns(2)
for i in range(2):
    with cols[i]:
        st.subheader(camera_names[i])
        available_models = list(model_urls.keys())
        selected_model = st.selectbox(
            f"Select Model for {camera_names[i]}",
            options=available_models,
            index=0,
            key=f"model_{camera_names[i]}",
        )
        img = display_latest_image_with_predictions(camera_names[i], selected_model)
        if img is not None:
            st.image(img, use_column_width=True)

cols2 = st.columns(2)
for i in range(2):
    with cols2[i]:
        st.subheader(camera_names[i+2])
        available_models = list(model_urls.keys())
        selected_model = st.selectbox(
            f"Select Model for {camera_names[i+2]}",
            options=available_models,
            index=0,
            key=f"model_{camera_names[i+2]}",
        )
        img = display_latest_image_with_predictions(camera_names[i+2],selected_model)
        if img is not None:
            st.image(img, use_column_width=True)

# --- Dataview Button ---
camera_option = st.selectbox("Select Camera for Detailed View", camera_names)  # Use camera_names list

if st.button("Go to Dataview"):
    st.session_state.camera = camera_option
    st.session_state.selected_model = "SHR_DSCAM"
    st.switch_page("pages/dataview.py")

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
