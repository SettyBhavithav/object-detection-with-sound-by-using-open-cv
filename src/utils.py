# ==============================================================================
# Real-Time Object Detection & Audio Alert System - Utilities Module
# ==============================================================================

import cv2
import numpy as np
import time

# Distinct BGR color palette for COCO classes
CLASS_COLORS = {
    "person": (0, 255, 0),         # Green
    "bicycle": (255, 0, 0),        # Blue
    "car": (0, 0, 255),            # Red
    "motorbike": (255, 255, 0),    # Cyan
    "aeroplane": (255, 0, 255),    # Magenta
    "bus": (0, 255, 255),          # Yellow
    "train": (128, 128, 0),        # Olive
    "truck": (128, 0, 128),        # Purple
    "boat": (0, 128, 128),          # Teal
    "traffic light": (255, 165, 0), # Orange
    "stop sign": (255, 20, 147),   # Deep Pink
    "bench": (0, 255, 127),        # Spring Green
    "cat": (150, 75, 0),           # Brown
    "dog": (255, 99, 71),          # Tomato
    "chair": (255, 215, 0),        # Gold
    "bottle": (147, 112, 219),     # Medium Purple
    "laptop": (220, 20, 60),       # Crimson
    "cell phone": (0, 191, 255)    # Deep Sky Blue
}

class FPSCounter:
    """Calculates smoothed frames per second (FPS)."""
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        if delta > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / delta)
        return self.fps

def get_class_color(class_name):
    """Returns assigned BGR color for class name or defaults to bright white."""
    return CLASS_COLORS.get(str(class_name).lower(), (255, 255, 255))

def draw_detections(frame, detections, draw_centroid=True, roi_rect=None):
    """
    Draws bounding boxes, labels, confidence scores, centroids, and optional ROI zone.
    """
    height, width, _ = frame.shape

    # Draw ROI Danger Zone if provided
    if roi_rect is not None:
        rx, ry, rw, rh = roi_rect
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
        cv2.putText(frame, "DANGER ZONE (ROI)", (rx + 5, ry + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for det in detections:
        x, y, w, h = det['box']
        class_name = det['class_name']
        conf = det['confidence']
        track_id = det.get('object_id', None)
        color = get_class_color(class_name)

        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Construct label text
        if track_id is not None:
            label = f"ID #{track_id} | {class_name.capitalize()}: {conf:.2f}"
        else:
            label = f"{class_name.capitalize()}: {conf:.2f}"

        # Draw text background banner for readability
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, max(0, y - label_h - 10)), (x + label_w, max(label_h, y)), color, -1)
        cv2.putText(frame, label, (x, max(label_h + 2, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw centroid marker
        if draw_centroid:
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    return frame

def is_inside_roi(box, roi_rect=None):
    """Checks if bounding box centroid lies inside specified ROI rectangle (x, y, w, h)."""
    if roi_rect is None:
        return True
    
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    rx, ry, rw, rh = roi_rect
    
    return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)
