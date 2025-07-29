# src/features.py

import numpy as np
import joblib
import os
import cv2
from skimage.feature import hog
from src.config import PCA_PATHS

# HOG parameters
# can be adjusted if needed
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def compute_hog(img: np.ndarray) -> np.ndarray:
    """Extract HOG features from a 2D grayscale image."""
    return hog(img, **HOG_PARAMS)


def compute_edges(img: np.ndarray) -> np.ndarray:
    """Extract edge-based features using the Canny detector."""
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    return edges.flatten() / 255.0  # normalize


def apply_pca(features: np.ndarray, combo: str) -> np.ndarray:
    """Load the PCA transformer for the given combo and apply it."""
    pca_path = PCA_PATHS.get(combo)
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA file not found at: {pca_path}")
    pca = joblib.load(pca_path)
    return pca.transform(features.reshape(1, -1))


def extract_features(img: np.ndarray, combo: str) -> np.ndarray:
    """
    Extract features from preprocessed image using the specified combo.
    Args:
        img: Preprocessed 2D grayscale image
        combo: One of ['raw', 'hog', 'hog_edge', 'hog_pca', 'hog_edge_pca', 'edge']
    Returns:
        np.ndarray: Feature vector with shape (1, n_features)
    """
    if combo == "raw":
        return img.flatten().reshape(1, -1) / 255.0

    elif combo == "hog":
        return compute_hog(img).reshape(1, -1)

    elif combo == "edge":
        return compute_edges(img).reshape(1, -1)

    elif combo == "hog_edge":
        hog_feat = compute_hog(img)
        edge_feat = compute_edges(img)
        combined = np.concatenate([hog_feat, edge_feat])
        return combined.reshape(1, -1)

    elif combo == "hog_pca":
        hog_feat = compute_hog(img)
        return apply_pca(hog_feat, combo)

    elif combo == "hog_edge_pca":
        hog_feat = compute_hog(img)
        edge_feat = compute_edges(img)
        combined = np.concatenate([hog_feat, edge_feat])
        return apply_pca(combined, combo)

    else:
        raise ValueError(f"Unknown feature combo: {combo}")
