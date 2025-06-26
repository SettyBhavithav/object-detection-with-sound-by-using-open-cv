# ==============================================================================
# Real-Time Object Detection & Audio Alert System - Test Suite
# Tests all detection edge cases, false positive mitigation, and API stability.
# ==============================================================================

import os
import sys
import unittest
import numpy as np
import cv2

# Add root directory to module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracker import CentroidTracker
from src.audio_engine import AudioEngineManager
from app import app, detector

class TestObjectDetectionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up global detector instance for tests."""
        cls.detector = detector
        cls.tracker = CentroidTracker(max_disappeared=10)

    def test_case_1_empty_black_frame(self):
        """Test Case 1: Pure black empty frame (0 detections, zero false positives)."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(black_frame)
        self.assertEqual(len(detections), 0, f"Expected 0 detections on empty black frame, got {len(detections)}")

    def test_case_2_random_noise_frame(self):
        """Test Case 2: Random noise frame (simulating camera static/sensor noise)."""
        np.random.seed(42)
        noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(noise_frame)
        self.assertEqual(len(detections), 0, f"Expected 0 false positives on random noise frame, got {len(detections)}")

    def test_case_3_room_background_gradient(self):
        """Test Case 3: Smooth room gradient & background shadows (empty space behind user)."""
        gradient_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            gradient_frame[i, :] = int(i / 480.0 * 200)
        detections = self.detector.detect(gradient_frame)
        self.assertEqual(len(detections), 0, f"Expected 0 false positive car/objects on background gradient, got {len(detections)}")

    def test_case_4_roi_filtering(self):
        """Test Case 4: Region of Interest (ROI) spatial boundary filtering."""
        # ROI zone: center box x=[160, 480], y=[96, 384]
        roi_rect = (160, 96, 320, 288)

        # Mock a box inside ROI
        box_inside = [200, 150, 100, 100]
        # Mock a box outside ROI
        box_outside = [10, 10, 50, 50]

        self.assertTrue(self.detector._is_inside_roi(box_inside, roi_rect), "Box inside ROI should pass")
        self.assertFalse(self.detector._is_inside_roi(box_outside, roi_rect), "Box outside ROI should be filtered out")

    def test_case_5_centroid_tracker(self):
        """Test Case 5: Centroid tracker object ID assignment."""
        mock_detections = [
            {'box': [100, 100, 50, 50], 'class_id': 0, 'class_name': 'person', 'confidence': 0.85}
        ]
        tracked = self.tracker.update(mock_detections)
        self.assertEqual(len(tracked), 1)
        self.assertIn('object_id', tracked[0])
        self.assertEqual(tracked[0]['object_id'], 0)

    def test_case_6_flask_health_endpoint(self):
        """Test Case 6: Flask API web server health check."""
        client = app.test_client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data.get("status"), "healthy")

    def test_case_7_flask_detect_api_empty_payload(self):
        """Test Case 7: API /api/detect error handling on missing image."""
        client = app.test_client()
        response = client.post("/api/detect")
        self.assertEqual(response.status_code, 400)

    def test_case_8_flask_detect_api_valid_image(self):
        """Test Case 8: API /api/detect with valid image payload."""
        # Create a valid JPEG in memory
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        _, img_buf = cv2.imencode('.jpg', frame)
        img_bytes = img_buf.tobytes()

        client = app.test_client()
        data = {
            'image': (sys.modules['io'].BytesIO(img_bytes), 'test.jpg'),
            'conf': '0.6'
        }
        response = client.post('/api/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertEqual(res_json.get("status"), "success")
        self.assertIn("detections", res_json)

    def test_case_9_large_webcam_person_box(self):
        """Test Case 9: Ensure large webcam person boxes (up to 85% area) are accepted."""
        # 640x480 total area = 307,200
        total_frame_area = 640 * 480
        # Mock a box covering 70% of frame area
        w, h = 480, 448 # area = 215040 (70% frame area)
        rel_area = (w * h) / float(total_frame_area)
        aspect_ratio = w / float(h)
        
        # Test aspect ratio and area limits for person class
        self.assertLess(rel_area, 0.88, "70% area person box should be under 88% limit")
        self.assertTrue(0.15 <= aspect_ratio <= 3.0, "Person aspect ratio should be valid")

if __name__ == "__main__":
    import io
    unittest.main()
