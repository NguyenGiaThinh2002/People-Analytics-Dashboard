import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Simple centroid-based tracker with line crossing detection.
        
        Args:
            max_disappeared: Frames before an ID is removed
            max_distance: Max pixel distance for matching centroids
        """
        self.next_id = 0
        self.objects = OrderedDict()      # {id: centroid}
        self.disappeared = OrderedDict()  # {id: frames_disappeared}
        self.previous_centroids = OrderedDict()  # {id: previous_centroid}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Counting
        self.count_in = 0
        self.count_out = 0
        self.crossed_ids = set()  # IDs that already crossed the line
        
        # Track recent crossings for statistics (cleared after being read)
        self.recent_crossings = []  # List of {"id": person_id, "direction": "in"/"out"}
    
    def reset_counts(self):
        """Reset all counters."""
        self.count_in = 0
        self.count_out = 0
        self.crossed_ids = set()
        self.recent_crossings = []
    
    def get_recent_crossings(self):
        """Get and clear recent crossings for statistics recording."""
        crossings = self.recent_crossings.copy()
        self.recent_crossings = []
        return crossings
    
    def register(self, centroid):
        """Register a new object with a unique ID."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.previous_centroids[self.next_id] = centroid
        self.next_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.previous_centroids:
            del self.previous_centroids[object_id]
        if object_id in self.crossed_ids:
            self.crossed_ids.discard(object_id)
    
    def update(self, bboxes, line_y):
        """
        Update tracker with new bounding boxes.
        
        Args:
            bboxes: List of (x1, y1, x2, y2) bounding boxes
            line_y: Y-coordinate of the counting line
        
        Returns:
            dict: {id: (centroid_x, centroid_y, x1, y1, x2, y2)}
        """
        # If no detections, mark all as disappeared
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._get_objects_with_boxes({})
        
        # Calculate centroids from bounding boxes
        input_centroids = []
        bbox_map = {}
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
            bbox_map[i] = (x1, y1, x2, y2)
        
        input_centroids = np.array(input_centroids)
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
            return self._get_objects_with_boxes(bbox_map)
        
        # Match existing objects to new detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        
        # Calculate distance matrix
        distances = np.zeros((len(object_centroids), len(input_centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, inp_centroid in enumerate(input_centroids):
                distances[i, j] = np.linalg.norm(obj_centroid - inp_centroid)
        
        # Greedy matching
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        matched = {}
        
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            
            object_id = object_ids[row]
            matched[object_id] = col
            used_rows.add(row)
            used_cols.add(col)
        
        # Update matched objects
        new_bbox_map = {}
        for object_id, col in matched.items():
            new_centroid = input_centroids[col]
            old_centroid = self.objects[object_id]
            
            # Check line crossing
            self._check_line_crossing(object_id, old_centroid, new_centroid, line_y)
            
            # Update
            self.previous_centroids[object_id] = old_centroid
            self.objects[object_id] = new_centroid
            self.disappeared[object_id] = 0
            new_bbox_map[object_id] = bbox_map[col]
        
        # Handle unmatched existing objects
        unmatched_rows = set(range(len(object_centroids))) - used_rows
        for row in unmatched_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
            else:
                # Keep last known bbox (approximate)
                new_bbox_map[object_id] = None
        
        # Register new objects
        unmatched_cols = set(range(len(input_centroids))) - used_cols
        for col in unmatched_cols:
            self.register(input_centroids[col])
            new_bbox_map[self.next_id - 1] = bbox_map[col]
        
        return self._get_objects_with_boxes(new_bbox_map)
    
    def _check_line_crossing(self, object_id, old_centroid, new_centroid, line_x):
        """Check if object crossed the vertical counting line."""
        if object_id in self.crossed_ids:
            return  # Already counted
        
        old_x = old_centroid[0] if isinstance(old_centroid, (list, tuple, np.ndarray)) else old_centroid
        new_x = new_centroid[0] if isinstance(new_centroid, (list, tuple, np.ndarray)) else new_centroid
        
        # Crossed left to right -> IN
        if old_x < line_x and new_x >= line_x:
            self.count_in += 1
            self.crossed_ids.add(object_id)
            self.recent_crossings.append({"id": object_id, "direction": "in"})
        # Crossed right to left -> OUT
        elif old_x > line_x and new_x <= line_x:
            self.count_out += 1
            self.crossed_ids.add(object_id)
            self.recent_crossings.append({"id": object_id, "direction": "out"})
    
    def _get_objects_with_boxes(self, bbox_map):
        """Return objects with their bounding boxes."""
        result = {}
        for object_id, centroid in self.objects.items():
            bbox = bbox_map.get(object_id)
            if bbox is not None:
                cx, cy = centroid[0], centroid[1]
                result[object_id] = (cx, cy, bbox[0], bbox[1], bbox[2], bbox[3])
        return result
    
    def get_counts(self):
        """Return (count_in, count_out, currently_inside)."""
        inside = self.count_in - self.count_out
        inside = max(0, inside)  # Ensure non-negative
        return self.count_in, self.count_out, inside

