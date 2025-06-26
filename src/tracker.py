# ==============================================================================
# Real-Time Object Detection & Audio Alert System - Centroid Tracker Module
# ==============================================================================

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    """
    Tracks object centroids across frames using Euclidean distance matching.
    Maintains persistent Object IDs to suppress alert spamming for stationary targets.
    """
    def __init__(self, max_disappeared=15, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()       # ID -> (cx, cy)
        self.disappeared = OrderedDict() # ID -> consecutive missing frame count
        self.class_names = OrderedDict()  # ID -> class_name
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, class_name="object"):
        """Registers new detected object centroid with unique ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.class_names[self.next_object_id] = class_name
        self.next_object_id += 1

    def deregister(self, object_id):
        """Removes object tracking state once missing for max_disappeared frames."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.class_names:
            del self.class_names[object_id]

    def update(self, detections):
        """
        Updates object centroid associations with new frame detections.
        
        Args:
            detections (list of dict): List of detections containing 'box' and 'class_name'.
            
        Returns:
            list of dict: Detections list augmented with 'object_id'.
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, det in enumerate(detections):
            x, y, w, h = det['box']
            input_centroids[i] = (x + w // 2, y + h // 2)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], detections[i]['class_name'])
                detections[i]['object_id'] = self.next_object_id - 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute Euclidean distance matrix between existing centroids & input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.class_names[object_id] = detections[col]['class_name']
                detections[col]['object_id'] = object_id

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], detections[col]['class_name'])
                    detections[col]['object_id'] = self.next_object_id - 1

        return detections
