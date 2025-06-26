import cv2
import numpy as np
import io
from app import app

# Create a synthetic frame representing a camera frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Encode JPEG
_, img_buf = cv2.imencode('.jpg', frame)
img_bytes = img_buf.tobytes()

client = app.test_client()
response = client.post('/api/detect', data={
    'image': (io.BytesIO(img_bytes), 'frame.jpg'),
    'conf': '0.35',
    'roi': 'false'
}, content_type='multipart/form-data')

print("Status Code:", response.status_code)
print("Response JSON:", response.get_json())
