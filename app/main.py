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
from app.image_processor import preprocess_image

# Load trained models
ulcer_model, severity_model = load_models()

# Define class labels
ulcer_labels = ["Non-Ulcer", "Ulcer"]
severity_labels = ["Mild", "Moderate", "Severe"]

# Initialize FastAPI app
app = FastAPI(
    title="Ulcer Detection & Severity API",
    description="Upload an image to detect ulcers and assess severity.",
    version="1.1"
)

# ✅ Enable CORS (Allow mobile app to make API requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your mobile app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Ulcer Detection & Severity API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img_array = preprocess_image(image_data)
        
        # Ulcer classification
        ulcer_prediction = ulcer_model.predict(img_array)[0][0]
        ulcer_result = ulcer_labels[int(ulcer_prediction > 0.5)]
        
        response = {"prediction": ulcer_result}

        # If ulcer detected, predict severity
        if ulcer_result == "Ulcer":
            severity_prediction = severity_model.predict(img_array)
            severity_class = np.argmax(severity_prediction)
            severity_result = severity_labels[severity_class]
            response["severity"] = severity_result

        return response

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
