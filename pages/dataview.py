import streamlit as st
import pandas as pd
import os
from PIL import Image
import cv2
import re

# --- Camera Names ---
camera_names = ["PC01A_CAMDSC102", "LV01C_CAMDSB106", "MJ01C_CAMDSB107", "MJ01B_CAMDSB103"]

# --- Dataview Page ---
if 'camera' not in st.session_state:
    st.write("Please select camera from main page")
else:
    st.title(f"Dataview - {st.session_state.camera}")

    # --- Load variables from Session State ---
    camera_id = st.session_state.camera
    year_month = "2021-08"  # This might not be relevant anymore

    # --- File Selection ---
    data_dir = os.path.join("timeseries", camera_id)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        st.error(f"No CSV files found in {data_dir}")
        st.stop()

    selected_csv = st.selectbox("Select CSV File", csv_files)

    try:
        df = pd.read_csv(selected_csv)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

    # --- Extract Class Names ---
    class_names = []
    for col in df.columns:
        if col == "File" or col == "Timestamp":
            continue
        if "Cluster" in col:
            break
        class_names.append(col)

    # --- Class Name Filtering ---
    selected_class = st.selectbox("Filter by Class", ['All'] + class_names)

    if selected_class != 'All':
        if selected_class not in class_names:
            st.error(f"Class '{selected_class}' not found in the data.")
            st.stop()
        df = df[df[selected_class] > 0]  # Filter rows where the class count is greater than 0

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
        image_filename = row['File']
        timestamp = row['Timestamp']
        
        # Construct image path (assuming images are in a directory relative to the CSV)
        image_path = os.path.join("images", camera_id, os.path.splitext(image_filename)[0] + ".jpg")  # Adjust extension if needed

        if not os.path.exists(image_path):
            st.warning(f"Image not found: {image_path}")
            continue

        try:
            img = Image.open(image_path)
            img_cv = cv2.imread(image_path)

            # Display image
            st.image(img, caption=f"File: {image_filename}, Timestamp: {timestamp}")

        except Exception as e:
            st.error(f"Error processing image {image_path}: {e}")
