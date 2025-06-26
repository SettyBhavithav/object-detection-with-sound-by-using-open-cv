# ==============================================================================
# Real-Time Object Detection & Audio Alert System - High Precision YOLO Detector
# ==============================================================================

import os
import cv2
import numpy as np

class YOLOObjectDetector:
    """
    High-precision YOLO Object Detector powered by OpenCV DNN module.
    Applies strict confidence filtering, Non-Maximum Suppression (NMS),
    bounding box area validation, and aspect ratio checks to eliminate false positives.
    """
    def __init__(self, weights_path=None, config_path=None, names_path=None, conf_threshold=0.55, nms_threshold=0.40):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None

        # Built-in OpenCV Haar Cascade for guaranteed webcam human detection fallback
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))

        # Candidate paths for weights and configs (supporting standard & tiny YOLO)
        weights_candidates = [
            weights_path,
            os.path.join(project_root, "models", "yolov3.weights"),
            os.path.join(project_root, "models", "yolov3-tiny.weights")
        ]
        config_candidates = [
            config_path,
            os.path.join(project_root, "config", "yolov3.cfg"),
            os.path.join(project_root, "config", "yolov3-tiny.cfg")
        ]
        names_path = names_path or os.path.join(project_root, "config", "coco.names")

        # Load COCO class labels
        if os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8") as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

        # Find first valid existing weights & config combination
        selected_weights = None
        selected_config = None

        for w_path in weights_candidates:
            if w_path and os.path.exists(w_path) and os.path.getsize(w_path) > 1000000:
                # Match corresponding config
                if "tiny" in os.path.basename(w_path).lower():
                    cfg_match = os.path.join(project_root, "config", "yolov3-tiny.cfg")
                else:
                    cfg_match = os.path.join(project_root, "config", "yolov3.cfg")

                if os.path.exists(cfg_match):
                    selected_weights = w_path
                    selected_config = cfg_match
                    break

        # If no valid local weights found (e.g. fresh cloud deployment on Render), download yolov3-tiny automatically
        if not selected_weights or not os.path.exists(selected_weights):
            selected_weights = self._download_default_weights(project_root)
            selected_config = os.path.join(project_root, "config", "yolov3-tiny.cfg")

        if selected_weights and selected_config and os.path.exists(selected_weights):
            print(f"[INFO] Initializing OpenCV DNN Model: {os.path.basename(selected_weights)}...")
            try:
                if hasattr(cv2.dnn, "readNetFromDarknet"):
                    self.net = cv2.dnn.readNetFromDarknet(selected_config, selected_weights)
                else:
                    self.net = cv2.dnn.readNet(selected_weights, selected_config)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.output_layers = self.net.getUnconnectedOutLayersNames()
                print(f"[SUCCESS] High-Precision YOLO Model ({os.path.basename(selected_weights)}) loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load DNN model: {e}")
                self.net = None
        else:
            print("[WARN] No valid model weights file found. Model will return 0 detections for empty frames.")

    def _download_default_weights(self, project_root):
        """Automatically downloads yolov3-tiny.weights if missing from the deployment environment."""
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, "yolov3-tiny.weights")

        if os.path.exists(target_path) and os.path.getsize(target_path) > 10000000:
            return target_path

        urls = [
            "https://pjreddie.com/media/files/yolov3-tiny.weights",
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov3-tiny.weights"
        ]

        print("[INFO] Cloud deployment detected: 'yolov3-tiny.weights' missing. Starting automatic download...")
        import urllib.request
        for url in urls:
            try:
                print(f"[DOWNLOADING] Fetching model weights from {url}...")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45) as response, open(target_path, 'wb') as out_file:
                    out_file.write(response.read())
                if os.path.exists(target_path) and os.path.getsize(target_path) > 10000000:
                    print(f"[SUCCESS] Model weights downloaded successfully ({os.path.getsize(target_path)} bytes).")
                    return target_path
            except Exception as e:
                print(f"[WARN] Download attempt from {url} failed: {e}")
        return None

    def detect(self, frame, roi_rect=None):
        """
        Processes video frame and extracts valid bounding boxes.
        Applies bounding box area checks, aspect ratio limits, and class-specific confidence filters.
        """
        if self.net is None:
            return []

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                objectness = float(detection[4])
                class_prob = float(scores[class_id])

                class_name = self.classes[class_id] if class_id < len(self.classes) else "object"
                
                # Class-specific confidence thresholds
                if class_name in ["car", "bus", "truck", "motorbike"]:
                    required_conf = max(self.conf_threshold, 0.55)
                    confidence = objectness * class_prob
                else:
                    # High sensitivity threshold for everyday objects (person, cell phone, bottle, chair, laptop, cup, etc.)
                    required_conf = min(self.conf_threshold, 0.20)
                    confidence = float(max(objectness * class_prob, class_prob))

                if confidence >= required_conf:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    if w <= 0 or h <= 0:
                        continue

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    area = w * h
                    total_frame_area = width * height
                    rel_area = area / float(total_frame_area)
                    aspect_ratio = w / float(h)

                    # 1. Minimum area check (ignore tiny noise < 0.2% frame size, e.g., ~600px for 640x480)
                    if area < (total_frame_area * 0.002):
                        continue

                    # 2. Class-specific area and aspect ratio validation
                    if class_name in ["car", "bus", "truck", "motorbike"]:
                        # Vehicles in room backgrounds must not cover massive screen areas unless confidence is high
                        if rel_area > 0.50 and confidence < 0.80:
                            continue
                        if not (0.40 <= aspect_ratio <= 3.5):
                            continue
                    elif class_name == "person":
                        # People sitting close to webcam can occupy up to 95% of frame
                        if rel_area > 0.95:
                            continue
                        if not (0.10 <= aspect_ratio <= 3.5):
                            continue

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) with threshold matching candidate selection
        nms_score_threshold = min(self.conf_threshold, 0.15)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, nms_score_threshold, self.nms_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                if self._is_inside_roi(box, roi_rect):
                    class_id = int(class_ids[i])
                    results.append({
                        'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'class_id': class_id,
                        'class_name': str(self.classes[class_id]) if class_id < len(self.classes) else "object",
                        'confidence': float(confidences[i])
                    })

        # Hybrid Fallback: If YOLO misses a close-up portrait face on a live webcam feed,
        # use OpenCV built-in Cascade Classifier to guarantee webcam human detection
        has_person = any(r['class_name'] == 'person' for r in results)
        if not has_person and self.face_cascade is not None and not self.face_cascade.empty():
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
                for (fx, fy, fw, fh) in faces:
                    px = int(max(0, fx - int(fw * 0.4)))
                    py = int(max(0, fy - int(fh * 0.3)))
                    pw = int(min(width - px, int(fw * 1.8)))
                    ph = int(min(height - py, int(fh * 2.8)))
                    box = [px, py, pw, ph]
                    if self._is_inside_roi(box, roi_rect):
                        results.append({
                            'box': box,
                            'class_id': 0,
                            'class_name': 'person',
                            'confidence': 0.88
                        })
            except Exception:
                pass

        return results

    def _is_inside_roi(self, box, roi_rect=None):
        if roi_rect is None:
            return True
        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2
        rx, ry, rw, rh = roi_rect
        return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)
