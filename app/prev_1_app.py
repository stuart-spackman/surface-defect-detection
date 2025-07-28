# app/app.py

# ========================================
# Surface Defect Classifier Web Interface
# ========================================
# 1. upload and image or choose from test set dropdown
# 2. preprocess the image (grayscale, resize)
# 3. extract features with HOG and PCA
# 4. load the trained model
# 5. make prediction
# 6. show
#     predicted class
#     prediction probabilities as a bar chart
#     image preview

import os
import cv2
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from skimage.feature import hog

# ----------------------
# Configuration Settings
# ----------------------
MODEL_DIR = "../models"  # Path to trained models
SAMPLE_DIR = "../data/raw/NEU-DET/validation/images"  # Path to sample images
COMBO = "hog_pca"  # Which feature combo to use (name of model files)
IMG_SIZE = (128, 128)  # Input image dimensions (should match training)

# ----------------------
# Load Trained Model
# ----------------------
model = joblib.load(os.path.join(MODEL_DIR, f"{COMBO}_logreg.pkl"))

# Class label names (extracted from model training)
label_encoder = model.classes_

# ----------------------------
# Feature Extraction Pipeline
# ----------------------------


def extract_hog(img):
    """
    Extract Histogram of Oriented Gradients (HOG) features from a grayscale image.
    """
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def preprocess_image(img):
    """
    Convert image to grayscale, resize to training size, and extract HOG features.
    Returns a 1D feature vector ready for prediction.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    features = extract_hog(resized)
    return np.array([features])  # Wrap in array to match scikit-learn input shape


def load_image(path):
    """
    Read an image from a file path (used for dropdown selection).
    """
    return cv2.imread(path)


# ----------------------------
# Streamlit App User Interface
# ----------------------------

st.title("🖥 Surface Defect Classifier")
st.markdown(f"Using feature combination: **`{COMBO}`**")

# ---- Upload Widget ----
uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# ---- Dropdown for Sample Images ----
sample_files = []
for root, _, files in os.walk(SAMPLE_DIR):
    for file in files:
        if file.lower().endswith(".jpg"):
            rel_path = os.path.relpath(os.path.join(root, file), SAMPLE_DIR)
            sample_files.append(rel_path)

sample_choice = st.selectbox("Or choose a sample image:", [""] + sample_files)

# ---- Determine Source of Image ----
image = None
image_name = ""

if uploaded:
    image_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    image_name = uploaded.name
elif sample_choice:
    full_path = os.path.join(SAMPLE_DIR, sample_choice)
    image = load_image(full_path)
    image_name = sample_choice

# ---- Display Image + Predict Button ----
if image is not None:
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption=image_name,
        use_column_width=True,
    )

    if st.button("Predict Defect Type"):
        try:
            # Extract features
            feats = preprocess_image(image)

            # Predict probabilities
            probs = model.predict_proba(feats)[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoder[pred_idx]

            # Display prediction
            st.success(f"Predicted Class: **{pred_label}**")

            # Bar chart of class probabilities
            fig, ax = plt.subplots()
            bars = ax.bar(label_encoder, probs, color="skyblue")
            ax.set_title("Prediction Probabilities")
            ax.set_ylabel("Confidence")
            ax.set_xticklabels(label_encoder, rotation=45, ha="right")

            # Optional: annotate bars with exact values
            for bar, prob in zip(bars, probs):
                ax.annotate(
                    f"{prob:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, prob),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
