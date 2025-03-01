import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from dotenv import load_dotenv
from app.image_processor import preprocess_image
from app.model_loader import load_models

# Load environment variables
load_dotenv()

# Get model paths from environment variables
ULCER_MODEL_PATH = os.getenv("ULCER_MODEL_PATH", "models/best_ulcer_classifier.h5")
SEVERITY_MODEL_PATH = os.getenv("SEVERITY_MODEL_PATH", "models/dfu_severity_final.h5")

# Load models
ulcer_model, severity_model = load_models(ULCER_MODEL_PATH, SEVERITY_MODEL_PATH)

# Class labels
ULCER_LABELS = ["Non-Ulcer", "Ulcer"]
SEVERITY_LABELS = ["Mild", "Moderate", "Severe"]

# Initialize FastAPI app
app = FastAPI(
    title="Ulcer Detection & Severity API",
    description="Upload an image to detect ulcers and assess severity.",
    version="1.1",
)

# Enable CORS for frontend/mobile integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
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
        # Read image file
        image_data = await file.read()
        
        # Preprocess the image
        img_array = preprocess_image(image_data)
        
        # Ulcer classification
        ulcer_prediction = ulcer_model.predict(img_array)[0][0]
        ulcer_result = ULCER_LABELS[int(ulcer_prediction > 0.5)]
        
        response = {"prediction": ulcer_result}

        # If ulcer detected, predict severity
        if ulcer_result == "Ulcer":
            severity_prediction = severity_model.predict(img_array)
            severity_class = np.argmax(severity_prediction)  # Get class with highest probability
            severity_result = SEVERITY_LABELS[severity_class]
            response["severity"] = severity_result

        return response

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
