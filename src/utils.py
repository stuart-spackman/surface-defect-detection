# src/utils.py

import numpy as np
import cv2
import os
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Tuple, Optional


def load_annotation(xml_path: str) -> Optional[list[Tuple[int, int, int, int]]]:
    """
    Load one or more bounding boxes from Pascal VOC XML annotation.
    Returns a list of (xmin, ymin, xmax, ymax).
    DEBUGGING ATTEMPT: editing to handle multiple boxes and print debug output
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append((xmin, ymin, xmax, ymax))

        if boxes:
            print(f"✅ Loaded {len(boxes)} box(es) from {xml_path}")
        else:
            print(f"⚠️ No boxes found in {xml_path}")

        return boxes if boxes else None
    except Exception as e:
        print(f"⚠️ Error parsing {xml_path}: {e}")
        return None


def find_annotation_for_image(
    filename: str, annotation_dirs: list
) -> Optional[list[Tuple[int, int, int, int]]]:
    """
    Given an image filename, search annotation folders for a matching XML file.
    DEBUGGING ATTEMPT: Now only supporting subfolders in sample image paths.
    """
    # just get the actual filename (e.g., 'pitted_surface_289.jpg')
    base_name = os.path.basename(filename)
    xml_name = os.path.splitext(base_name)[0] + ".xml"

    for ann_dir in annotation_dirs:
        xml_path = os.path.join(ann_dir, xml_name)
        if os.path.exists(xml_path):
            return load_annotation(xml_path)

    print(f"❌ Annotation not found for: {filename} (looking for {xml_name})")
    return None


def draw_real_bounding_box(
    image: Image.Image,
    boxes: list[Tuple[int, int, int, int]],
    original_size=(200, 200),
    color=(255, 0, 0),
    thickness=1,
) -> Image.Image:
    """
    Draw a real bounding box (xmin, ymin, xmax, ymax) directly on the image.
    Assumes the image is already at its original resolution (e.g., 200x200).
    DEBUGGING ATTEMPT: adjusting to support multiple boxes
    """
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    scale_x = img_w / original_size[0]
    scale_y = img_h / original_size[1]

    for box in boxes:
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, thickness)

    return Image.fromarray(img_array)
