import os
import glob
import pandas as pd
import boto3
import torch
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
    and saves the predictions in Parquet format on S3, partitioned by date.
    """
    bucket_name = config["bucket_name"]
    data_s3_path = f"s3://{bucket_name}/{camera_id}/data_{year_month}"
    local_image_dir = f"{camera_id}_data_{year_month}"
    parquet_s3_prefix = f"{camera_id}/predictions/{year_month}"  # S3 prefix for partitioned Parquet files

    # Create S3 client
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),  # Use environment variables
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=config["region_name"],
        )
        logging.info("Successfully created S3 client.")
    except Exception as e:
        logging.error(f"Error creating S3 client: {e}", exc_info=True)
        print(f"Error creating S3 client: {e}")
        return

    # List image files in the local directory
    image_files = glob.glob(os.path.join(local_image_dir, "*.jpg"))  # Adjust for other image formats
    if not image_files:
        logging.warning(f"No images found in local directory: {local_image_dir}")
        print(f"No images found in local directory: {local_image_dir}")
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
                "image_path": f"{data_s3_path}/{image_name}",
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
            parquet_s3_key = f"{parquet_s3_prefix}/date={date_partition}/{parquet_file_name}"
            parquet_s3_path = f"s3://{bucket_name}/{parquet_s3_key}"

            # Save the DataFrame to Parquet on S3
            try:
                df.to_parquet(parquet_s3_path, engine='fastparquet')
                logging.info(f"Predictions saved to {parquet_s3_path}")
                print(f"Predictions saved to {parquet_s3_path}")
            except Exception as e:
                logging.error(f"Error saving predictions to {parquet_s3_path}: {e}", exc_info=True)
                print(f"Error saving predictions to {parquet_s3_path}: {e}")
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
