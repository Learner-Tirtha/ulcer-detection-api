import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define model paths
ULCER_MODEL_PATH = "models/best_ulcer_classifier.h5"
SEVERITY_MODEL_PATH = "models/dfu_severity_final.h5"

def load_models():
    if not os.path.exists(ULCER_MODEL_PATH) or not os.path.exists(SEVERITY_MODEL_PATH):
        raise FileNotFoundError("Model files not found. Ensure they are uploaded to Render.")
    
    ulcer_model = load_model(ULCER_MODEL_PATH)
    severity_model = load_model(SEVERITY_MODEL_PATH)
    
    return ulcer_model, severity_model
