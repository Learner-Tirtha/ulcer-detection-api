from tensorflow.keras.models import load_model

def load_models(ulcer_model_path, severity_model_path):
    """
    Load Keras models from .h5 files.
    """
    try:
        ulcer_model = load_model(ulcer_model_path)  # Ensure .h5 format
        severity_model = load_model(severity_model_path)  # Ensure .h5 format
        return ulcer_model, severity_model
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")
