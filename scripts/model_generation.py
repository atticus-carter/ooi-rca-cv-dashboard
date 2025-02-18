import os
from ultralytics import YOLO
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Management ---
model_urls = {
    "SHR_DSCAM": "https://huggingface.co/atticus-carter/SHR_DSCAM/resolve/main/best.pt",
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
        if os.path.exists(url):
            model = YOLO(url)  # Load local model
            logging.info(f"Model '{model_name}' loaded from local file: {url}")
        else:
            # Download the model to a temporary file
            import urllib.request
            temp_model_path = "temp_model.pt"
            urllib.request.urlretrieve(url, temp_model_path)
            model = YOLO(temp_model_path)  # Load from temp file
            logging.info(f"Model '{model_name}' loaded from URL: {url}")
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {e}", exc_info=True)
        raise Exception(f"Error loading model '{model_name}': {e}")

    models[model_name] = model
    return model

def generate_predictions(image_path, model_name="SHR_DSCAM"): # Default model
    """Generates YOLO predictions for a given image using the specified model."""
    try:
        model = load_model(model_name)
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        print(f"Error loading model: {e}")
        return []

    try:
        logging.info(f"Running YOLO inference on image: {image_path}")
        results = model(image_path)  # Run inference

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
