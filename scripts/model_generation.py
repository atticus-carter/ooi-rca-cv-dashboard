import os
from ultralytics import YOLO
import logging
import urllib.request  # Import urllib
import glob  # Import glob
import cv2
import numpy as np
from PIL import Image

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Management ---
model_urls = {
    "SHR_DSCAM": "https://huggingface.co/atticus-carter/SHR_DSCAM/raw/main/best.pt",
    "Megalodon": "https://huggingface.co/FathomNet/megalodon/resolve/main/mbari-megalodon-yolov8x.pt?download=true",
    "315K": "https://huggingface.co/FathomNet/MBARI-315k-yolov8/resolve/main/mbari_315k_yolov8.pt?download=true"
    # Add other models here with their URLs
}

models = {}  # Store loaded models

def load_model(model_name):
    """Loads a YOLO model from a URL or cache."""
    if model_name in models:
        return models[model_name]

    url = model_urls.get(model_name)
    if not url:
        raise ValueError(f"Model '{model_name}' not found. Check model_urls dictionary.")

    try:
        # Check if the model is a local file or URL
        # Search for a .pt file in the weights directory
        weights_dir = "."  # Assuming the weights are in the root directory
        pt_files = glob.glob(os.path.join(weights_dir, "*.pt"))

        if pt_files:
            model_path = pt_files[0]  # Take the first .pt file found
            model = YOLO(model_path)  # Load local model
            logging.info(f"Model '{model_name}' loaded from local file: {model_path}")
        else:
            # Download the model to a temporary file
            temp_model_path = "best.pt"
            try:
                urllib.request.urlretrieve(url, temp_model_path)
            except Exception as e:
                logging.error(f"Error downloading model '{model_name}' from URL: {url}", exc_info=True)
                raise Exception(f"Error downloading model '{model_name}' from URL: {e}")

            if not temp_model_path.endswith(".pt"):
                os.remove(temp_model_path)
                raise ValueError(f"Downloaded model file is not a valid PyTorch model (.pt): {temp_model_path}")

            model = YOLO(temp_model_path)  # Load from temp file
            logging.info(f"Model '{model_name}' loaded from URL: {url}")
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {e}", exc_info=True)
        raise Exception(f"Error loading model '{model_name}': {e}")

    models[model_name] = model
    return model

def generate_predictions(img_array, model_name="SHR_DSCAM", conf_thres=0.25, iou_thres=0.45): # Default model
    """Generates YOLO predictions for a given image using the specified model."""
    try:
        model = load_model(model_name)
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        print(f"Error loading model: {e}")
        return []

    try:
        logging.info(f"Running YOLO inference on image")
        # Convert the image array to a PIL Image
        img_pil = Image.fromarray(img_array)
        results = model(img_pil, imgsz=1024, conf=conf_thres, iou=iou_thres)  # Run inference with thresholds and image size

        predictions = []
        for result in results:
            boxes = result.boxes # Get boxes from the result
            xywhn = result.boxes.xywhn  # normalized xywh
            names = result.names  # class names
            confs = result.boxes.conf  # confidence scores
            classes = result.boxes.cls.int() # class indices

            for i in range(len(xywhn)):
                class_id = classes[i].item()
                class_name = names[class_id]
                confidence = confs[i].item()
                bbox = xywhn[i].tolist() # Normalized xywh (center x, center y, width, height)
                predictions.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": bbox, # normalized
                    "confidence": confidence,
                })
        logging.info(f"Generated {len(predictions)} predictions for image")
        return predictions
    except Exception as e:
        logging.error(f"Error during inference on image: {e}", exc_info=True)
        print(f"Error during inference: {e}")
        return []
