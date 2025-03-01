from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
