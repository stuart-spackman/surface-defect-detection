# app/app.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
from src.predict import classify_image


# define a directory for sample images
SAMPLE_DIR = "data/raw/NEU-DET/validation/images"  # path to sample images

# page configuration
st.set_page_config(
    page_title="Surface Defect Detector",
    layout="centered",
    initial_sidebar_state="expanded",
)

# title of the application
st.title("🔍 Surface Defect Detection App")

# sidebar for model selection
model_options = ["raw", "edge", "hog", "hog_edge", "hog_pca", "hog_edge_pca"]
selected_model = st.sidebar.selectbox("Choose a feature set + model", model_options)

# ------ image selection section ------
st.header("🖼️ Select or Upload Image")

# gather sample images
sample_files = []
for root, _, files in os.walk(SAMPLE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            rel_path = os.path.relpath(os.path.join(root, file), SAMPLE_DIR)
            sample_files.append(rel_path)

# sample image dropdown
sample_choice = st.selectbox("Or choose a sample image:", [""] + sample_files)

# file uploader
uploaded_file = st.file_uploader("Upload your own image", type=["jpg", "jpeg", "png"])

# determine which image to load
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("Image successfully loaded.")
elif sample_choice:
    image_path = os.path.join(SAMPLE_DIR, sample_choice)
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption=f"Sample Image: {sample_choice}", use_column_width=True)
    st.success("Sample image loaded.")
else:
    st.info("Please upload an image or choose a sample image.")

# ------ classification section ------
if image is not None:
    if st.button("🔎 Classify Image"):
        with st.spinner("Classifying..."):

            # placeholder for prediction pipeline (to be implemented in ../src/)
            from src.predict import classify_image

            # run classification
            predicted_label, class_probs = classify_image(image, selected_model)

            # display result
            st.subheader("🧠 Prediction")

            # first get confidence of the top prediction
            top_confidence = max(class_probs.values())

            # then choose a color based on confidence
            if top_confidence >= 0.80:
                label_color = "✅"
            elif top_confidence >= 0.50:
                label_color = "⚠️"
            else:
                label_color = "❌"

            st.markdown(
                f"{label_color} **Predicted Class:** `{predicted_label}` ({top_confidence: .2%} confidence)"
            )

            st.subheader("📊 Class Probabilities")
            st.bar_chart(class_probs)
