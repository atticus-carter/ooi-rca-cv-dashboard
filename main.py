import sys
import os
sys.path.insert(0, os.getcwd())

import glob
import pandas as pd
import duckdb
import streamlit as st
# from .scripts.model_generation import generate_predictions, model_urls, load_model  # Removed relative import
from scripts.model_generation import generate_predictions, model_urls, load_model # Added absolute import
import re
import time  # Import the time module
import yaml  # Import the YAML module
import subprocess  # Import subprocess
import logging
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import urllib.request

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

# --- Camera Names and URLs ---
camera_data = {
    "PC01A_CAMDSC102": "https://rawdata.oceanobservatories.org/files/RS01SBPS/PC01A/CAMDSC102_10.33.3.146/2025/01/09/CAMDSC102_10.33.3.146_20250109T154628_UTC.jpg",
    "MJ01B_CAMDSB103": "https://rawdata.oceanobservatories.org/files/RS01SUM2/MJ01B/CAMDSB103_10.33.7.5/2025/02/18/CAMDSB103_10.33.7.5_20250218T234538_UTC.jpg",
    "LV01C_CAMDSB106": "https://rawdata.oceanobservatories.org/files/CE04OSBP/LV01C/CAMDSB106_10.33.9.6/2025/02/18/CAMDSB106_10.33.9.6_20250218T234536_UTC.jpg",
    "MJ01C_CAMDSB107": "https://rawdata.oceanobservatories.org/files/CE02SHBP/MJ01C/CAMDSB107_10.33.13.8/2025/02/18/CAMDSB107_10.33.13.8_20250218T234535_UTC.jpg"
}

camera_names = list(camera_data.keys())

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
    image_url = camera_data[camera_id]
    try:
        # Download the image
        with urllib.request.urlopen(image_url) as response:
            img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None:
                st.warning(f"Could not decode image from URL: {image_url}")
                return None, None

        # Resize the image using bicubic interpolation
        img_cv = cv2.resize(img_cv, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Extract timestamp from filename
        image_name = os.path.basename(image_url)
        timestamp = extract_timestamp_from_filename(image_name)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if timestamp else "Timestamp N/A"

        return img_pil, timestamp_str

    except Exception as e:
        st.error(f"Error processing image from URL {image_url}: {e}")
        return None, None

# --- Display Images in a 2x2 Grid ---
# Add global controls for confidence and IoU thresholds
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

# Initialize session state for images and selected models
if 'images' not in st.session_state:
    st.session_state.images = {camera_id: None for camera_id in camera_names}
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = {camera_id: None for camera_id in camera_names}
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = {camera_id: default_models[camera_id] for camera_id in camera_names}

cols = st.columns(2)
for i in range(2):
    with cols[i]:
        camera_id = camera_names[i]
        st.subheader(f"{camera_id}")
        
        # Display initial image without predictions
        if st.session_state.images[camera_id] is None:
            img, timestamp = display_latest_image_with_predictions(camera_id)
            st.session_state.images[camera_id] = img
            st.session_state.timestamps[camera_id] = timestamp
        
        st.text(f"{st.session_state.timestamps[camera_id]}" if st.session_state.timestamps[camera_id] else "Loading...")
        
        # Model selection dropdown
        available_models = list(model_urls.keys())
        selected_model = st.selectbox(
            f"Select Model for {camera_id}",
            options=['None'] + available_models,
            index=available_models.index(st.session_state.selected_models[camera_id]) + 1 if st.session_state.selected_models[camera_id] in available_models else 0,
            key=f"model_{camera_id}"
        )
        
        if st.session_state.images[camera_id] is not None:
            st.image(st.session_state.images[camera_id], use_container_width=True)

        # Generate predictions button
        if st.button(f"Generate Predictions", key=f"generate_{camera_id}"):
            with st.spinner('Generating predictions...'):
                model_to_use = st.session_state[f"model_{camera_id}"] if st.session_state[f"model_{camera_id}"] != 'None' else None
                if model_to_use:
                    # Get the original image
                    img_pil = st.session_state.images[camera_id]
                    if img_pil is not None:
                        # Generate predictions
                        predictions = generate_predictions(np.array(img_pil), model_to_use, conf_threshold, iou_threshold)
                        # Load model and generate visualization
                        model = load_model(model_to_use)
                        results = model(img_pil, imgsz=1024, conf=conf_threshold, iou=iou_threshold)
                        # Update the image with predictions
                        for r in results:
                            im_bgr = r.plot()
                            im_rgb = Image.fromarray(im_bgr[..., ::-1])
                            st.session_state.images[camera_id] = im_rgb
                            st.image(im_rgb, use_container_width=True)

cols2 = st.columns(2)
for i in range(2):
    with cols2[i]:
        camera_id = camera_names[i+2]
        st.subheader(f"{camera_id}")
        
        # Display initial image without predictions
        if st.session_state.images[camera_id] is None:
            img, timestamp = display_latest_image_with_predictions(camera_id)
            st.session_state.images[camera_id] = img
            st.session_state.timestamps[camera_id] = timestamp
        
        st.text(f"{st.session_state.timestamps[camera_id]}" if st.session_state.timestamps[camera_id] else "Loading...")
        
        # Model selection dropdown
        available_models = list(model_urls.keys())
        selected_model = st.selectbox(
            f"Select Model for {camera_id}",
            options=['None'] + available_models,
            index=available_models.index(st.session_state.selected_models[camera_id]) + 1 if st.session_state.selected_models[camera_id] in available_models else 0,
            key=f"model_{camera_id}"
        )
        
        if st.session_state.images[camera_id] is not None:
            st.image(st.session_state.images[camera_id], use_container_width=True)

        # Generate predictions button
        if st.button(f"Generate Predictions", key=f"generate_{camera_id}"):
            with st.spinner('Generating predictions...'):
                model_to_use = st.session_state[f"model_{camera_id}"] if st.session_state[f"model_{camera_id}"] != 'None' else None
                if model_to_use:
                    # Get the original image
                    img_pil = st.session_state.images[camera_id]
                    if img_pil is not None:
                        # Generate predictions
                        predictions = generate_predictions(np.array(img_pil), model_to_use, conf_threshold, iou_threshold)
                        # Load model and generate visualization
                        model = load_model(model_to_use)
                        results = model(img_pil, imgsz=1024, conf=conf_threshold, iou=iou_threshold)
                        # Update the image with predictions
                        for r in results:
                            im_bgr = r.plot()
                            im_rgb = Image.fromarray(im_bgr[..., ::-1])
                            st.session_state.images[camera_id] = im_rgb
                            st.image(im_rgb, use_container_width=True)

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

# --- Timeseries View Page Link ---
if st.button("View Timeseries Data"):
    st.switch_page("pages/timeseries_view.py")
