# src/config.py

import os

# image size used during training
IMAGE_SIZE = (128, 128)  # width, height

# root directory of the project (relative to this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# model directory
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# PCA models directory - actual PCA .pkl files are inside these subdirectories
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features", "NEU-DET")

# map combo name to the exact PCA model path
PCA_PATHS = {
    "hog_pca": os.path.join(FEATURE_DIR, "hog_pca", "pca_model.pkl"),
    "hog_edge_pca": os.path.join(FEATURE_DIR, "hog_edge_pca", "pca_model.pkl"),
}
