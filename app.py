# ==============================================================================
# Real-Time Object Detection & Audio Alert System - Flask Web Server (Render Cloud)
# ==============================================================================

import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.detector import YOLOObjectDetector
from src.utils import draw_detections

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize global detector instance
detector = YOLOObjectDetector(conf_threshold=0.35)

@app.route("/")
def index():
    """Serves the main Web UI Dashboard."""
    return render_template("index.html")

@app.route("/health")
def health():
    """Health check endpoint required for Render cloud deployment."""
    return jsonify({"status": "healthy", "service": "Object Detection Audio AI"})

@app.route("/sounds/<path:filename>")
def serve_sound(filename):
    """Serves sound alert audio assets."""
    sounds_dir = os.path.join(app.root_path, "sounds")
    return send_from_directory(sounds_dir, filename)

@app.route("/api/detect", methods=["POST"])
def detect_api():
    """
    JSON API endpoint accepting uploaded image binary.
    Returns array of detection bounding boxes and confidence scores.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image payload uploaded"}), 400

    file = request.files["image"]
    conf = float(request.form.get("conf", 0.35))
    roi_flag = request.form.get("roi", "false").lower() == "true"

    # Read image from memory buffer
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    height, width, _ = frame.shape
    roi_rect = (int(width * 0.25), int(height * 0.2), int(width * 0.5), int(height * 0.6)) if roi_flag else None

    # Update confidence threshold dynamically
    detector.conf_threshold = conf
    detections = detector.detect(frame, roi_rect=roi_rect)

    return jsonify({
        "status": "success",
        "detections": detections,
        "count": len(detections)
    })

@app.route("/api/detect_render_image", methods=["POST"])
def detect_render_image_api():
    """
    API endpoint accepting an uploaded image file, running detection,
    and returning Base64-encoded annotated JPEG for browser display.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image payload uploaded"}), 400

    file = request.files["image"]
    conf = float(request.form.get("conf", 0.35))
    roi_flag = request.form.get("roi", "false").lower() == "true"

    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    height, width, _ = frame.shape
    roi_rect = (int(width * 0.25), int(height * 0.2), int(width * 0.5), int(height * 0.6)) if roi_flag else None

    detector.conf_threshold = conf
    detections = detector.detect(frame, roi_rect=roi_rect)

    # Render bounding boxes onto frame
    output_frame = draw_detections(frame, detections, draw_centroid=True, roi_rect=roi_rect)

    # Encode JPEG to Base64
    _, buffer = cv2.imencode('.jpg', output_frame)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "status": "success",
        "detections": detections,
        "count": len(detections),
        "image_base64": img_b64
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# Rechecking OpenCV video stream initialization and pyttsx3 audio queue.
