import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_confidence=0.5):
        """
        Fast face detector using MediaPipe.
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # 1 = full-range model (better for varying distances)
            min_detection_confidence=min_confidence
        )
    
    def detect_faces_in_rois(self, frame, tracked_objects):
        """
        Detect faces only within person bounding boxes (ROIs).
        
        Args:
            frame: Full frame (BGR)
            tracked_objects: Dict from tracker {id: (cx, cy, x1, y1, x2, y2)}
        
        Returns:
            dict: {person_id: [(face_x1, face_y1, face_x2, face_y2), ...]}
                  Face coordinates are in FULL FRAME coordinates
        """
        faces_by_person = {}
        
        for person_id, data in tracked_objects.items():
            cx, cy, x1, y1, x2, y2 = data
            
            # Ensure valid ROI bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract ROI (upper portion of person bbox - where face usually is)
            # Focus on top 60% of bounding box for better face detection
            roi_height = y2 - y1
            face_region_y2 = y1 + int(roi_height * 0.6)
            
            roi = frame[y1:face_region_y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Convert to RGB for MediaPipe
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Detect faces in ROI
            results = self.detector.process(roi_rgb)
            
            faces = []
            if results.detections:
                roi_h, roi_w = roi.shape[:2]
                
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coords to ROI pixels
                    face_x1_roi = int(bbox.xmin * roi_w)
                    face_y1_roi = int(bbox.ymin * roi_h)
                    face_w = int(bbox.width * roi_w)
                    face_h = int(bbox.height * roi_h)
                    
                    # Add padding for better crop (30% on each side)
                    pad_x = int(face_w * 0.3)
                    pad_y = int(face_h * 0.3)
                    
                    # Convert to full frame coordinates with padding
                    face_x1 = x1 + face_x1_roi - pad_x
                    face_y1 = y1 + face_y1_roi - pad_y
                    face_x2 = x1 + face_x1_roi + face_w + pad_x
                    face_y2 = y1 + face_y1_roi + face_h + pad_y
                    
                    # Clamp to frame bounds
                    face_x1 = max(0, face_x1)
                    face_y1 = max(0, face_y1)
                    face_x2 = min(frame.shape[1], face_x2)
                    face_y2 = min(frame.shape[0], face_y2)
                    
                    # Only add if face is reasonably sized
                    face_width = face_x2 - face_x1
                    face_height = face_y2 - face_y1
                    if face_width >= 30 and face_height >= 30:
                        faces.append((face_x1, face_y1, face_x2, face_y2))
            
            faces_by_person[person_id] = faces
        
        return faces_by_person
    
    def get_face_crops(self, frame, faces_by_person):
        """
        Extract face crops from the frame with good quality.
        
        Args:
            frame: Full frame (BGR)
            faces_by_person: Dict from detect_faces_in_rois
        
        Returns:
            dict: {person_id: [face_crop_image, ...]}
        """
        crops_by_person = {}
        
        for person_id, faces in faces_by_person.items():
            crops = []
            for (fx1, fy1, fx2, fy2) in faces:
                if fx2 > fx1 and fy2 > fy1:
                    crop = frame[fy1:fy2, fx1:fx2].copy()
                    if crop.size > 0 and crop.shape[0] >= 30 and crop.shape[1] >= 30:
                        crops.append(crop)
            crops_by_person[person_id] = crops
        
        return crops_by_person
    
    def draw_faces(self, frame, faces_by_person):
        """
        Draw blue rectangles around detected faces.
        
        Args:
            frame: Frame to draw on (modified in place)
            faces_by_person: Dict from detect_faces_in_rois
        """
        for person_id, faces in faces_by_person.items():
            for (fx1, fy1, fx2, fy2) in faces:
                # Blue rectangle for face
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
    
    def close(self):
        """Release resources."""
        self.detector.close()
