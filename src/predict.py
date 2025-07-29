# src/predict.py

import os
import joblib
import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image
from src.features import extract_features
from src.config import MODEL_DIR, IMAGE_SIZE

# mapping from model name to filename
MODEL_PATHS = {
    "hog_pca": os.path.join(MODEL_DIR, "hog_pca_logreg.pkl"),
    "hog_edge_pca": os.path.join(MODEL_DIR, "hog_edge_pca_logreg.pkl"),
    "hog": os.path.join(MODEL_DIR, "hog_logreg.pkl"),
    "hog_edge": os.path.join(MODEL_DIR, "hog_edge_logreg.pkl"),
    "edge": os.path.join(MODEL_DIR, "edge_logreg.pkl"),
    "raw": os.path.join(MODEL_DIR, "raw_logreg.pkl"),
}

# class names
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def classify_image(image: Image.Image, feature_combo: str):
    """Classify an image using the specified feature set + logistic regression model."""
    # load the corresponding trained model
    model_path = MODEL_PATHS.get(feature_combo)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file for '{feature_combo}' not found at {model_path}."
        )

    model = joblib.load(model_path)

    # preprocess the image (resize, grayscale, etc.)
    img_arr = preprocess_image(image, size=IMAGE_SIZE)

    # extract features (based on combo: hog, hog_pca, edge, etc.)
    features = extract_features(img_arr, combo=feature_combo)

    # get prediction
    probs = model.predict_proba(features)[0]  # shape: (num_classes,)
    predicted_index = np.argmax(probs)
    predicted_label = CLASS_NAMES[predicted_index]

    # build a dictionary to map labels to probabilities
    class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return predicted_label, class_probs
