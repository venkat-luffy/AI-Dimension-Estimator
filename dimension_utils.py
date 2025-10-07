# dimension_utils.py

import cv2
import numpy as np

def get_dimensions_with_depth(mask, depth_map, pixels_per_cm):
    """
    Calculates dimensions using a segmentation mask and a depth map.
    """
    if mask is None or depth_map is None or pixels_per_cm is None:
        return None
    x, y, w, h = cv2.boundingRect(mask)
    width_cm = w / pixels_per_cm
    height_cm = h / pixels_per_cm
    object_depth_map = depth_map[mask > 0]
    avg_depth_value = np.percentile(object_depth_map, 50)
    depth_cm = width_cm * avg_depth_value
    volume_cm3 = width_cm * height_cm * depth_cm
    return {
        "width_cm": round(width_cm, 1),
        "height_cm": round(height_cm, 1),
        "depth_cm": round(depth_cm, 1),
        "volume_cm3": round(volume_cm3, 1)
    }

def draw_measurements(frame, mask, dimensions):
    """
    Draw contour and measurements on frame and return overlay image.
    frame: color BGR image
    mask: uint8 binary mask (0/255)
    dimensions: dict returned by get_dimensions_with_depth
    """
    overlay = frame.copy()
    if mask is None:
        return overlay
    masked_pixels = cv2.bitwise_and(frame, frame, mask=mask)
    mean_brightness = 0
    if mask.sum() > 0:
        mean_brightness = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2GRAY)[mask > 0].mean()
    text_color = (0, 0, 0) if mean_brightness > 127 else (255, 255, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, [max(contours, key=cv2.contourArea)], -1, (0, 255, 255), 2)
    if dimensions is None:
        return overlay
    if dimensions.get("width_cm") is not None:
        lines = [
            f"W: {dimensions['width_cm']} cm",
            f"H: {dimensions['height_cm']} cm",
            f"D: {dimensions['depth_cm']} cm",
            f"V: {dimensions['volume_cm3']} cm³"
        ]
    else:
        lines = [
            f"W: {dimensions.get('width_px',0)} px",
            f"H: {dimensions.get('height_px',0)} px",
            f"(no scale)"
        ]
    x, y, w, h = cv2.boundingRect(mask)
    tx, ty = x, max(20, y - 10)
    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (tx, ty + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return overlay
