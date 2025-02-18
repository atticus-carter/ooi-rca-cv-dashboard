import os
import glob
import pandas as pd
from ultralytics import YOLO
from scripts.model_generation import generate_predictions  # Import your prediction function
import re
import yaml
from datetime import datetime
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_timestamp_from_filename(filename):
    """Extracts timestamp from the filename."""
    match = re.search(r"(\d{8}T\d{6})", filename)  # Matches the timestamp format YYYYMMDDTHHMMSS
    if match:
        timestamp_str = match.group(1)  # Extract the matched timestamp string
        timestamp = pd.to_datetime(timestamp_str, format="%Y%m%dT%H%M%S")
        return timestamp
    else:
        return None

def batch_process_camera_data(camera_id, year_month, model_name, config):
    """
    Processes all images for a given camera and year/month, generates predictions,
    and saves the predictions in Parquet format locally, partitioned by date.
    """
    image_dir = os.path.join("images", camera_id, year_month)
    parquet_prefix = os.path.join("images", camera_id, "predictions")  # Local prefix for partitioned Parquet files

    # List image files in the local directory
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))  # Adjust for other image formats
    if not image_files:
        logging.warning(f"No images found in local directory: {image_dir}")
        print(f"No images found in local directory: {image_dir}")
        return

    # Process each image and generate predictions
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        timestamp = extract_timestamp_from_filename(image_name)
        if timestamp is None:
            logging.warning(f"Could not extract timestamp from filename {image_name}. Skipping.")
            print(f"Warning: Could not extract timestamp from filename {image_name}. Skipping.")
            continue

        try:
            predictions = generate_predictions(image_file, model_name)
        except Exception as e:
            logging.error(f"Error generating predictions for {image_file}: {e}", exc_info=True)
            print(f"Error generating predictions for {image_file}: {e}")
            continue

        # Create a DataFrame for the predictions
        data = []
        for prediction in predictions:
            data.append({
                "camera_id": camera_id,
                "timestamp": timestamp,
                "image_path": image_file,  # Store local image path
                "class_id": prediction["class_id"],
                "class_name": prediction["class_name"],
                "bbox_x": prediction["bbox"][0],
                "bbox_y": prediction["bbox"][1],
                "bbox_width": prediction["bbox"][2],
                "bbox_height": prediction["bbox"][3],
                "confidence": prediction["confidence"],
            })
        df = pd.DataFrame(data)

        if not df.empty:
            # Partition the Parquet file by date
            date_partition = timestamp.strftime("%Y-%m-%d")
            parquet_file_name = f"{camera_id}_predictions_{timestamp.strftime('%Y%m%d')}.parquet"
            parquet_dir = os.path.join(parquet_prefix, f"date={date_partition}")
            os.makedirs(parquet_dir, exist_ok=True)  # Ensure the directory exists
            parquet_file_path = os.path.join(parquet_dir, parquet_file_name)

            # Save the DataFrame to Parquet locally
            try:
                df.to_parquet(parquet_file_path, engine='fastparquet')
                logging.info(f"Predictions saved to {parquet_file_path}")
                print(f"Predictions saved to {parquet_file_path}")
            except Exception as e:
                logging.error(f"Error saving predictions to {parquet_file_path}: {e}", exc_info=True)
                print(f"Error saving predictions to {parquet_file_path}: {e}")
        else:
            logging.warning(f"No predictions generated for {image_name}")
            print(f"No predictions generated for {image_name}")

if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set parameters
    camera_id = "PC01A_CAMDSC102"  # Replace with the desired camera ID
    year_month = "2021-08"  # Replace with the desired year and month
    model_name = "SHR_DSCAM"

    # Run batch processing
    batch_process_camera_data(camera_id, year_month, model_name, config)
