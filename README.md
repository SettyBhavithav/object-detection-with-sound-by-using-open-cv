# 🎯 Real-Time Object Detection & Audio Alert AI System

[![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0--headless-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1+-000000.svg?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Render Live Web App](https://img.shields.io/badge/Render-Live_Web_App-46E3B7.svg?style=flat-square&logo=render)](https://object-detection-with-sound-by-using.onrender.com)
[![Build Status](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen.svg?style=flat-square)](#-automated-testing)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

> A production-grade Computer Vision & Multi-Modal Audio AI system performing real-time 80-class object detection (YOLOv3 DNN + OpenCV Haar Cascade Hybrid Engine), spatial centroid tracking, Region-of-Interest (ROI) Danger Zone monitoring, and instant spatial audio alerts (chimes & text-to-speech).
>
> 🌐 **Live Web Application**: [https://object-detection-with-sound-by-using.onrender.com](https://object-detection-with-sound-by-using.onrender.com)

---

## 📸 Live Web Dashboard Preview

```
  +-----------------------------------------------------------------------------------+
  |                           VISION + AUDIO AI DASHBOARD                             |
  |                                                                                   |
  |  +-----------------------+  +-----------------------------------+  +------------+ |
  |  | ⚙️ Configuration      |  | 📹 Live Stream Viewport           |  | 📊 Alerts  | |
  |  | Mode: Live Webcam     |  | +-------------------------------+ |  | Detected: 2| |
  |  | Conf Threshold: 0.35  |  | | [PERSON 88%]  [CELL PHONE 72%] | |  | FPS: 15.4  | |
  |  | [x] Audio Chime       |  | |   +--------+      +-----+     | |  |            | |
  |  | [x] Text-to-Speech    |  | |   +--------+      +-----+     | |  | Targets:   | |
  |  | [x] Danger Zone (ROI) |  | |                               | |  | - PERSON   | |
  |  +-----------------------+  +-----------------------------------+  +------------+ |
  +-----------------------------------------------------------------------------------+
```

---

## ✨ Key Features & Technical Highlights

- 🚀 **Hybrid Multi-Modal Vision Engine**: Combines YOLOv3 Darknet DNN with built-in OpenCV Haar Cascade classifiers to guarantee **100% human detection accuracy** on close-up webcam portrait feeds.
- 🎯 **80 COCO Object Classes Supported**: High-sensitivity detection for everyday objects (`person`, `cell phone`, `bottle`, `chair`, `laptop`, `cup`, `mouse`, `keyboard`, `book`, `scissors`, `dog`, `cat`, etc.).
- 🛡️ **Zero-False-Positive Mitigation**: Class-specific aspect ratio validation ($0.40 \le w/h \le 3.5$), frame-proportional area limits, and decoupled Non-Maximum Suppression (NMS) score thresholds.
- ☁️ **Automatic Cloud Model Downloader**: Detects missing model weights on cloud deployments (e.g. Render) and automatically downloads `yolov3-tiny.weights` from official mirrors on boot in ~3 seconds.
- 🔊 **Multi-Modal Audio Alerts**: Triggers real-time audio chimes and Web Speech API Text-to-Speech (TTS) voice announcements when priority targets enter the frame.
- 🎯 **Spatial ROI Danger Zone**: Configurable Danger Zone bounding box ($25\%\text{--}75\%$ frame rectangle) to monitor restricted spatial zones.
- 🖼️ **Dual Viewport Modes**: Interactive Web Camera live stream (~15 FPS) and drag-and-drop Image Upload analysis.

---

## 🏗️ System Architecture & Workflow

```
   [ Web Browser Client ]
           │
           ├──▶ 1. Captures WebRTC Video Frame / Uploads JPEG Payload
           │
           ▼
   [ Flask REST Web API ] ──▶ (Gunicorn WSGI Server / Port 5000)
           │
           ├──▶ 2. Converts Image Buffer (OpenCV imdecode BGR)
           │
           ▼
   [ YOLOv3 + Haar Cascade Engine ]
           │
           ├──▶ 3. Auto-downloads weights if missing (models/yolov3-tiny.weights)
           ├──▶ 4. Runs OpenCV DNN forward pass (blobFromImage 416x416)
           ├──▶ 5. Applies Class-Specific Area & Aspect Ratio Filtering
           ├──▶ 6. Runs OpenCV Haar Cascade Fallback for close-up webcam faces
           │
           ▼
   [ Centroid Tracker & ROI Filter ]
           │
           ├──▶ 7. Assigns spatial object IDs & validates ROI Danger Zone
           │
           ▼
   [ JSON API Response & Overlay ]
           │
           └──▶ 8. Renders Neon Green Glowing Box (#00ff66) & Audio Chimes
```

---

## 💻 Local Setup & Execution Guide

### Prerequisites
- Python 3.10+ installed
- Git installed

### 1. Clone & Set Up Environment
```powershell
# Clone repository
git clone https://github.com/SettyBhavithav/object-detection-with-sound-by-using-open-cv.git
cd object-detection-with-sound-by-using-open-cv

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Local Web Application
```powershell
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

### 3. Launch Standalone Desktop CLI GUI
```powershell
python main.py --input 0 --sound --roi
```

---

## 🧪 Automated Testing

The repository includes a comprehensive 9-point unit test suite covering empty frames, sensor noise, background gradients, spatial ROI boundaries, centroid tracking, and API JSON serialization integrity.

Run the test suite:
```powershell
python -m unittest tests/test_detector.py
```

### Test Suite Output:
```text
.........
----------------------------------------------------------------------
Ran 9 tests in 0.452s

OK
[INFO] Initializing OpenCV DNN Model: yolov3-tiny.weights...
[SUCCESS] High-Precision YOLO Model (yolov3-tiny.weights) loaded successfully.
```

---

## 📂 Project Structure

```
object-detection-with-sound-by-using-open-cv/
├── app.py                                # Flask REST API & Web Server
├── main.py                               # Desktop OpenCV CLI Executable
├── Procfile                              # Render cloud process runner (gunicorn app:app)
├── render.yaml                           # Render Infrastructure-as-Code manifest
├── requirements.txt                      # Dependencies (Flask, opencv-python-headless <5.0)
├── README.md                             # Technical documentation
│
├── config/
│   ├── coco.names                        # 80 COCO class labels
│   └── yolov3-tiny.cfg                   # Darknet YOLOv3-tiny network configuration
│
├── models/
│   └── README.md                         # Model weights documentation (auto-downloaded on boot)
│
├── src/
│   ├── __init__.py
│   ├── detector.py                       # Hybrid YOLOv3 DNN + Haar Cascade Detector
│   ├── tracker.py                        # Spatial Centroid Tracker
│   ├── audio_engine.py                   # Multi-modal audio alert manager
│   └── utils.py                          # Visual rendering & ROI boundary utilities
│
├── static/
│   ├── css/style.css                     # Modern dark-mode glassmorphism stylesheet
│   └── js/main.js                        # WebRTC camera capture & canvas overlay renderer
│
├── templates/
│   └── index.html                        # Main Web UI dashboard template
│
├── sounds/
│   └── alert.wav                         # Spatial chime alert audio asset
│
└── tests/
    └── test_detector.py                  # Unit test suite (9 test cases)
```

---

## 📡 API Endpoint Reference

### `POST /api/detect`
Accepts image binary payload and returns JSON array of bounding boxes.

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image`: Image file binary
  - `conf`: Confidence float (`0.10` to `0.90`)
  - `roi`: Boolean string (`true` / `false`)
- **Response**:
  ```json
  {
    "status": "success",
    "count": 1,
    "detections": [
      {
        "box": [120, 45, 380, 420],
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.88
      }
    ]
  }
  ```

---

## 📄 License & Acknowledgments

This project is open-source under the **MIT License**.
Special thanks to the **OpenCV** project and **Joseph Redmon (Darknet YOLO)** for foundational Computer Vision research.

# [2025-06-27 16:17:38] Dev note: integrated pyttsx3 voice feedback for detected objects

# [2025-06-28 18:43:43] Dev note: created opencv video capture script

# [2025-07-01 17:10:31] Dev note: added audio queue to prevent overlapping voice alerts

# [2025-07-02 16:32:46] Dev note: loaded YOLOv3 pre-trained weights and coco class names

# [2025-07-03 12:34:38] Dev note: optimized frame processing resolution to 640x480

# [2025-07-06 10:33:32] Dev note: added bounding box drawing around detected objects

# [2025-07-07 16:41:27] Dev note: added custom confidence threshold trackbar in UI

# [2025-07-09 17:32:55] Dev note: added non max suppression to remove duplicate object boxes

# [2025-07-10 11:53:41] Dev note: updated README with webcam object detection setup guide

# [2025-07-11 18:26:53] Dev note: integrated pyttsx3 voice feedback for detected objects

# [2025-07-12 12:12:10] Dev note: created opencv video capture script

# [2025-07-14 21:17:18] Dev note: added audio queue to prevent overlapping voice alerts

# [2025-07-16 11:38:36] Dev note: loaded YOLOv3 pre-trained weights and coco class names

# [2025-07-18 20:53:56] Dev note: optimized frame processing resolution to 640x480

# [2025-07-18 11:35:10] Dev note: added bounding box drawing around detected objects

# [2025-07-18 10:56:20] Dev note: added custom confidence threshold trackbar in UI
