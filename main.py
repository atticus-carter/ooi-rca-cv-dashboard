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
from ultralytics import YOLO

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

# --- Default Models ---
default_models = {
    "PC01A_CAMDSC102": "Megalodon",
    "LV01C_CAMDSB106": "315K",
    "MJ01C_CAMDSB107": "315K",
    "MJ01B_CAMDSB103": "SHR_DSCAM"
}

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
    """Generates a unique vibrant color for each class name."""
    # Predefined vibrant colors (RGB format)
    vibrant_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 0, 128),  # Pink
        (0, 255, 128),  # Spring Green
    ]
    # Use hash of class name to pick a color deterministically
    color_index = hash(class_name) % len(vibrant_colors)
    return vibrant_colors[color_index]

# --- Main Streamlit App ---
st.title("OOI RCA CV Dashboard")

# --- Find Most Recent Image and Display Predictions ---
def display_latest_image_with_predictions(camera_id, selected_model=None, conf_thres=0.25, iou_thres=0.45):
    # Build the absolute path so images are correctly found in deployment
    base_dir = os.path.join(os.getcwd(), "images")
    image_dir = os.path.join(base_dir, camera_id, year_month)
    if not os.path.exists(base_dir):
        st.warning(f"No 'images' directory found in {os.getcwd()}")
        return None
    if not os.path.exists(image_dir):
        st.warning(f"No data found in local directory: {image_dir}")
        return None

    # Use a glob pattern that matches both ".jpg" and ".png"
    image_files = glob.glob(os.path.join(image_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(image_dir, "*.[pP][nN][gG]"))
    st.sidebar.text(f"Found files in {image_dir}: {image_files}")
    
    if not image_files:
        st.warning(f"No images found in local directory: {image_dir}")
        return None

    # Find the most recent image
    most_recent_image = max(image_files, key=os.path.getmtime)

    # Load the image using OpenCV
    img_cv = cv2.imread(most_recent_image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.open(most_recent_image)
    img_width, img_height = img_pil.size

    # Determine the model to use
    model_to_use = selected_model if selected_model != 'None' else None

    # Generate predictions for the most recent image
    if model_to_use:
        predictions = generate_predictions(most_recent_image, model_to_use, conf_thres, iou_thres)

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
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)  # Thicker lines
            
            # Add a filled rectangle behind the text for better visibility
            label = f"{class_name} {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(img_cv, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return img_cv

# --- Display Images in a 2x2 Grid ---
# Add global controls for confidence and IoU thresholds
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

# Initialize session state for images
if 'images' not in st.session_state:
    st.session_state.images = {camera_id: None for camera_id in camera_names}

cols = st.columns(2)
for i in range(2):
    with cols[i]:
        camera_id = camera_names[i]
        st.subheader(camera_id)
        available_models = list(model_urls.keys())
        selected_model = st.selectbox(
            f"Select Model for {camera_id}",
            options=['None'] + available_models,
            index=available_models.index(default_models[camera_id]) + 1 if default_models[camera_id] in available_models else 0,
            key=f"model_{camera_id}",
        )
        
        # Display initial image with default model predictions
        if st.session_state.images[camera_id] is None:
            st.session_state.images[camera_id] = display_latest_image_with_predictions(
                camera_id,
                default_models[camera_id],
                conf_threshold,
                iou_threshold
            )
        if st.session_state.images[camera_id] is not None:
            st.image(st.session_state.images[camera_id], use_container_width=True)

        if st.button(f"Generate Predictions", key=f"generate_{camera_id}"):
            with st.spinner('Generating predictions...'):
                st.session_state.images[camera_id] = display_latest_image_with_predictions(
                    camera_id,
                    selected_model if selected_model != 'None' else None,
                    conf_threshold,
                    iou_threshold
                )
                if st.session_state.images[camera_id] is not None:
                    st.image(st.session_state.images[camera_id], use_container_width=True)

cols2 = st.columns(2)
for i in range(2):
    with cols2[i]:
        camera_id = camera_names[i+2]
        st.subheader(camera_id)
        available_models = list(model_urls.keys())
        selected_model = st.selectbox(
            f"Select Model for {camera_id}",
            options=['None'] + available_models,
            index=available_models.index(default_models[camera_id]) + 1 if default_models[camera_id] in available_models else 0,
            key=f"model_{camera_id}",
        )
        
        # Display initial image with default model predictions
        if st.session_state.images[camera_id] is None:
            st.session_state.images[camera_id] = display_latest_image_with_predictions(
                camera_id,
                default_models[camera_id],
                conf_threshold,
                iou_threshold
            )
        if st.session_state.images[camera_id] is not None:
            st.image(st.session_state.images[camera_id], use_container_width=True)

        if st.button(f"Generate Predictions", key=f"generate_{camera_id}"):
            with st.spinner('Generating predictions...'):
                st.session_state.images[camera_id] = display_latest_image_with_predictions(
                    camera_id,
                    selected_model if selected_model != 'None' else None,
                    conf_threshold,
                    iou_threshold
                )
                if st.session_state.images[camera_id] is not None:
                    st.image(st.session_state.images[camera_id], use_container_width=True)

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
