import torch
import os
from ultralytics import YOLO

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
        raise ValueError(f"Model '{model_name}' not found.  Check model_urls dictionary.")

    try:
        # Check if the model is a local file or URL
        if os.path.exists(url):
            model = YOLO(url)  # Load local model
        else:
            torch.hub.download_url_to_file(url, 'temp_model.pt')  # Download to temp file
            device = torch.device('cpu')
            model = YOLO('temp_model.pt', device=device)  # Load from temp file
    except Exception as e:
        raise Exception(f"Error loading model '{model_name}': {e}")

    models[model_name] = model
    return model

def generate_predictions(image_path, model_name="SHR_DSCAM"): # Default model
    """Generates YOLO predictions for a given image using the specified model."""
    try:
        model = load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    try:
        print("Running YOLO, just saying this to ensure this code actually runs")
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
        return predictions
    except Exception as e:
        print(f"Error during inference: {e}")
        return []
