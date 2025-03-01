import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

def preprocess_image(image_data):
    """
    Preprocess uploaded image for model prediction.
    """
    img = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
