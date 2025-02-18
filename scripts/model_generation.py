import os
from ultralytics import YOLO
import logging
import urllib.request  # Import urllib
import glob  # Import glob

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Management ---
model_urls = {
    "SHR_DSCAM": "https://huggingface.co/atticus-carter/SHR_DSCAM/raw/main/best.pt",
    "Megalodon": "https://huggingface.co/FathomNet/megalodon/resolve/main/best.pt?download=true",
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

def generate_predictions(image_path, model_name="SHR_DSCAM", conf_thres=0.25, iou_thres=0.45): # Default model
    """Generates YOLO predictions for a given image using the specified model."""
    try:
        model = load_model(model_name)
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        print(f"Error loading model: {e}")
        return []

    try:
        logging.info(f"Running YOLO inference on image: {image_path}")
        results = model(image_path, conf=conf_thres, iou=iou_thres)  # Run inference with thresholds

        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xywhn[0].tolist() # Normalized xywh (center x, center y, width, height)
                predictions.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": bbox, # normalized
                    "confidence": confidence,
                })
        logging.info(f"Generated {len(predictions)} predictions for image: {image_path}")
        return predictions
    except Exception as e:
        logging.error(f"Error during inference on image {image_path}: {e}", exc_info=True)
        print(f"Error during inference: {e}")
        return []
