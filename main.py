import cv2
import time
from ultralytics import YOLO
from config import (
    CAMERA_SOURCE, WINDOW_NAME, CONFIDENCE_THRESHOLD,
    MAX_DISAPPEARED, MAX_DISTANCE, LINE_POSITION,
    ENABLE_FACE_DETECTION, FACE_DETECTION_CONFIDENCE
)
from utils.tracker import CentroidTracker
from utils.face_detector import FaceDetector
from utils.gender_age import GenderAgeAnalyzer


def main():
    # Load YOLOv8 model
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully!")
    
    # Initialize tracker
    tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE)
    
    # Initialize face detector and gender/age analyzer
    face_detector = None
    gender_age_analyzer = None
    if ENABLE_FACE_DETECTION:
        print("Loading face detector...")
        face_detector = FaceDetector(min_confidence=FACE_DETECTION_CONFIDENCE)
        gender_age_analyzer = GenderAgeAnalyzer()
        print("Face detector and gender/age analyzer loaded!")
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get frame dimensions for line position
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_x = int(frame_width * LINE_POSITION)
    
    print(f"Camera opened: {frame_width}x{frame_height}")
    print(f"Counting line at X={line_x} (vertical)")
    print(f"Face detection: {'ENABLED' if ENABLE_FACE_DETECTION else 'DISABLED'}")
    print("Press 'q' to quit, 'r' to reset counts, 'f' to toggle face detection, 'c' to clear cache.")
    
    # FPS calculation
    prev_time = time.time()
    face_detection_enabled = ENABLE_FACE_DETECTION
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Run YOLO detection (class 0 = person only)
        results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Extract bounding boxes
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
        
        # Update tracker
        objects = tracker.update(bboxes, line_x)
        
        # Face detection and gender/age analysis
        faces_by_person = {}
        if face_detection_enabled and face_detector and len(objects) > 0:
            faces_by_person = face_detector.detect_faces_in_rois(frame, objects)
            face_detector.draw_faces(frame, faces_by_person)
            
            # Get face crops for gender/age analysis
            if gender_age_analyzer:
                # Get crops for persons without cached results
                crops_to_analyze = {}
                for person_id, faces in faces_by_person.items():
                    if not gender_age_analyzer.get_cached(person_id) and faces:
                        crops = face_detector.get_face_crops(frame, {person_id: faces})
                        if person_id in crops and crops[person_id]:
                            crops_to_analyze[person_id] = crops[person_id]
                
                # Analyze new faces
                if crops_to_analyze:
                    gender_age_analyzer.analyze_faces(crops_to_analyze)
                
                # Clean up old cached IDs
                gender_age_analyzer.clear_old_ids(objects.keys())
        
        # Draw vertical counting line (red)
        cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 3)
        
        # Draw tracked objects with gender/age info
        for object_id, data in objects.items():
            cx, cy, x1, y1, x2, y2 = data
            
            # Draw green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            
            # Build label with ID, gender, age
            label_parts = [f"{object_id}"]
            
            if gender_age_analyzer:
                cached = gender_age_analyzer.get_cached(object_id)
                if cached:
                    label_parts.append(cached["gender"][0])  # M or F
                    label_parts.append(cached["age_group"])
            
            label = " | ".join(label_parts)
            
            # Draw label background and text
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_bg_x2 = x1 + label_size[0] + 10
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (label_bg_x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Get counts
        count_in, count_out, inside = tracker.get_counts()
        
        # Draw count display (big text at top)
        count_text = f"IN: {count_in}   OUT: {count_out}   INSIDE: {inside}"
        
        # Draw background for count text
        text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, (5, 50), (text_size[0] + 20, 95), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Face detection status indicator
        face_status = "Face: ON" if face_detection_enabled else "Face: OFF"
        cv2.putText(frame, face_status, (frame_width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 0, 0) if face_detection_enabled else (128, 128, 128), 2)
        
        # Show frame
        cv2.imshow(WINDOW_NAME, frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_counts()
            print("Counts reset!")
        elif key == ord('f'):
            face_detection_enabled = not face_detection_enabled
            print(f"Face detection: {'ON' if face_detection_enabled else 'OFF'}")
        elif key == ord('c'):
            if gender_age_analyzer:
                gender_age_analyzer.cache.clear()
                print("Gender/age cache cleared! Will re-analyze all faces.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if face_detector:
        face_detector.close()
    
    # Final stats
    count_in, count_out, inside = tracker.get_counts()
    print(f"\n{'='*50}")
    print(f"FINAL STATISTICS")
    print(f"{'='*50}")
    print(f"People counts: IN={count_in}, OUT={count_out}, INSIDE={inside}")
    
    if gender_age_analyzer:
        stats = gender_age_analyzer.get_statistics()
        print(f"\nGender distribution:")
        print(f"  Male: {stats['gender']['Male']}")
        print(f"  Female: {stats['gender']['Female']}")
        print(f"\nAge distribution:")
        for group, count in stats['age_groups'].items():
            print(f"  {group}: {count}")
    
    print(f"{'='*50}")
    print("Camera closed.")


if __name__ == "__main__":
    main()
