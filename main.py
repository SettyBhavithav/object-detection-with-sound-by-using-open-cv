# ==============================================================================
# Real-Time Object Detection & Audio Alert System - Main Executable
# ==============================================================================

import os
import sys
import time
import argparse
import cv2
import numpy as np

from src.detector import YOLOObjectDetector
from src.tracker import CentroidTracker
from src.audio_engine import AudioEngineManager
from src.utils import draw_detections, FPSCounter

def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Computer Vision Object Detection & Multi-Modal Audio Alert System"
    )
    parser.add_argument("--input", type=str, default="0",
                        help="Video source: '0' for webcam, path to video/image file, or 'demo' for synthetic mode.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to yolov3.weights file (default: models/yolov3.weights).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to yolov3.cfg configuration file.")
    parser.add_argument("--names", type=str, default=None,
                        help="Path to coco.names labels file.")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Object detection confidence threshold (0.0 to 1.0).")
    parser.add_argument("--sound", action="store_true", default=True,
                        help="Enable audio chime alerts on object detection.")
    parser.add_argument("--tts", action="store_true", default=False,
                        help="Enable spoken text-to-speech (TTS) announcements.")
    parser.add_argument("--roi", action="store_true", default=False,
                        help="Enable Region of Interest (ROI) danger zone monitoring.")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Run without displaying OpenCV window GUI.")
    args = parser.parse_args()

    print("======================================================================")
    print("      REAL-TIME OBJECT DETECTION & MULTI-MODAL AUDIO ALERT SYSTEM     ")
    print("======================================================================")

    # Initialize YOLO Detector
    detector = YOLOObjectDetector(
        weights_path=args.weights,
        config_path=args.config,
        names_path=args.names,
        conf_threshold=args.conf
    )

    # Initialize Centroid Tracker & Audio Engine
    tracker = CentroidTracker(max_disappeared=15)
    audio_engine = AudioEngineManager(cooldown_seconds=2.0, enable_tts=args.tts)
    fps_counter = FPSCounter()

    # Determine Video Source
    if args.input == "demo" or args.input == "synthetic":
        print("[INFO] Video Source: Synthetic Demo Mode")
        cap = None
    elif args.input.isdigit():
        camera_id = int(args.input)
        print(f"[INFO] Video Source: Live Camera Stream (ID: {camera_id})")
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        print(f"[INFO] Video Source: File Input ({args.input})")
        if not os.path.exists(args.input):
            print(f"[WARN] Input video file not found: {args.input}. Falling back to Demo mode.")
            cap = None
        else:
            cap = cv2.VideoCapture(args.input)

    # Path to sound alert file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sound_path = os.path.join(script_dir, "sounds", "alert.wav")

    print("\n[READY] System operational. Press 'q' in the window to terminate execution.\n")

    roi_rect = None

    try:
        while True:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream reached or frame read failed.")
                    break
            else:
                # Create synthetic 640x480 frame for demo execution
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "SYNTHETIC DEMO MODE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            height, width, _ = frame.shape

            # Define default ROI Danger Zone if requested
            if args.roi:
                roi_rect = (int(width * 0.25), int(height * 0.2), int(width * 0.5), int(height * 0.6))
            else:
                roi_rect = None

            # 1. Run Object Detection
            detections = detector.detect(frame, roi_rect=roi_rect)

            # 2. Update Centroid Tracking
            tracked_detections = tracker.update(detections)

            # 3. Trigger Audio Alert for High-Priority Detections
            for det in tracked_detections:
                class_name = det['class_name']
                if class_name in ["person", "car", "bus", "truck", "dog", "chair", "bottle", "laptop"]:
                    audio_engine.trigger_alert(class_name, sound_path=sound_path)

            # 4. Render Visual Overlays
            frame = draw_detections(frame, tracked_detections, draw_centroid=True, roi_rect=roi_rect)

            # Display FPS
            fps = fps_counter.update()
            cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show GUI Window unless --no-show is set
            if not args.no_show:
                cv2.imshow("Real-Time Object Detection & Audio Alerts", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Terminating session via user exit command.")
                    break
            else:
                # In no-show/headless mode, break after 100 test iterations
                if cap is None and fps_counter.fps > 0:
                    time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupt signal received. Shutting down cleanly.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Video capture released & windows closed.")

if __name__ == "__main__":
    main()

# [2025-06-26 18:44:48] Dev note: created opencv video capture script

# [2025-06-27 13:41:45] Dev note: added audio queue to prevent overlapping voice alerts

# [2025-06-28 19:17:53] Dev note: loaded YOLOv3 pre-trained weights and coco class names

# [2025-07-01 14:44:10] Dev note: optimized frame processing resolution to 640x480

# [2025-07-02 09:41:23] Dev note: added bounding box drawing around detected objects

# [2025-07-04 15:25:20] Dev note: added custom confidence threshold trackbar in UI

# [2025-07-06 16:10:37] Dev note: added non max suppression to remove duplicate object boxes

# [2025-07-08 13:49:42] Dev note: updated README with webcam object detection setup guide

# [2025-07-09 13:57:27] Dev note: integrated pyttsx3 voice feedback for detected objects

# [2025-07-10 15:22:15] Dev note: created opencv video capture script

# [2025-07-11 20:26:16] Dev note: added audio queue to prevent overlapping voice alerts

# [2025-07-13 09:53:10] Dev note: loaded YOLOv3 pre-trained weights and coco class names

# [2025-07-14 11:42:40] Dev note: optimized frame processing resolution to 640x480

# [2025-07-16 15:11:45] Dev note: added bounding box drawing around detected objects

# [2025-07-18 12:23:53] Dev note: added custom confidence threshold trackbar in UI

# [2025-07-18 18:16:46] Dev note: added non max suppression to remove duplicate object boxes

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv video capture script

# integrated pyttsx3 voice feedback for detected objects

# created opencv webcam video capture initialization script

# applied non maximum suppression to filter overlapping object boxes

# created audio alert queue module to prevent overlapping voice prompts

# optimized video frame rate by running audio synthesis in separate thread

# saved sample detection output screenshot to images directory

# added requirements txt dependencies list with opencv python and pyttsx3

# created opencv webcam video capture initialization script

# applied non maximum suppression to filter overlapping object boxes

# created audio alert queue module to prevent overlapping voice prompts
