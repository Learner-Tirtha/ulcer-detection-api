from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
from app.model_loader import load_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")  # Move up a level
ULCER_MODEL_PATH = os.path.join(MODELS_DIR, "best_ulcer_classifier.h5")
SEVERITY_MODEL_PATH = os.path.join(MODELS_DIR, "dfu_severity_final.h5")


# Load models
ulcer_model, severity_model = load_models(ULCER_MODEL_PATH, SEVERITY_MODEL_PATH)

# Define class labels
ulcer_labels = ["Non-Ulcer", "Ulcer"]
severity_labels = ["Mild", "Moderate", "Severe"]  # Adjust based on model output classes

# Initialize FastAPI app
app = FastAPI(title="Ulcer Detection & Severity API", description="Upload an image to detect ulcers and assess severity.", version="1.1")

# Enable CORS (Allow frontend/mobile access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image preprocessing function
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.get("/")
def home():
    return {"message": "Ulcer Detection & Severity API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image_data = await file.read()
        
        # Preprocess the image
        img_array = preprocess_image(image_data)
        
        # Ulcer classification
        ulcer_prediction = ulcer_model.predict(img_array)[0][0]
        ulcer_result = ulcer_labels[int(ulcer_prediction > 0.5)]
        
        response = {"prediction": ulcer_result}

        # If ulcer detected, predict severity
        if ulcer_result == "Ulcer":
            severity_prediction = severity_model.predict(img_array)
            severity_class = np.argmax(severity_prediction)  # Get class with highest probability
            severity_result = severity_labels[severity_class]
            response["severity"] = severity_result

        return response

    except Exception as e:
        return {"error": str(e)}
