import tensorflow as tf
from tensorflow.keras.models import load_model

def load_models(ulcer_model_path, severity_model_path):
    """
    Load and return ulcer and severity classification models.
    """
    ulcer_model = load_model(ulcer_model_path)
    severity_model = load_model(severity_model_path)
    return ulcer_model, severity_model
