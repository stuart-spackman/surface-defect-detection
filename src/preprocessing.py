# src/preprocessing.py

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image, size=(128, 128)) -> np.ndarray:
    """
    Preprocess an image: convert to grayscale, resize, histogram equalize, and denoise.

    Args:
        image: PIL.Image object
        size: target (width, height) tuple for resizing

    Returns:
        np.ndarray: Preprocessed image (grayscale, 2D array)
    """

    # convert PIL Image to NumPy array
    img = np.array(image)

    if img is None:
        raise ValueError("Image not found: {img_path}")

    # convert to grayscale (if not already)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # resize to target dimensions
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # histogram equalization (contrast enhancement)
    equalized = cv2.equalizeHist(resized)

    # add Gaussian blur for extra noise reduction
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

    return blurred  # 2D array of shape (height, width)
