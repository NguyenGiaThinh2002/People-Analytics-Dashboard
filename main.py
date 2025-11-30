import warnings
# Suppress protobuf deprecation warnings
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import cv2
import time
import os
from ultralytics import YOLO
from config import (
    CAMERA_SOURCE, WINDOW_NAME, CONFIDENCE_THRESHOLD,
    MAX_DISAPPEARED, MAX_DISTANCE, LINE_POSITION,
    ENABLE_FACE_DETECTION, FACE_DETECTION_CONFIDENCE,
    FACE_DETECTION_INTERVAL, GENDER_AGE_INTERVAL, USE_THREADED_ANALYSIS
)
from utils.tracker import CentroidTracker
from utils.face_detector import FaceDetector
from utils.gender_age import GenderAgeAnalyzer
from utils.statistics import DailyStats
from utils.visualizer import Visualizer


def main():
    # Load YOLOv8 model
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully!")
    
    # Initialize tracker
    tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE)
    
    # Initialize statistics tracker
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    daily_stats = DailyStats(report_dir=report_dir)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Initialize face detector and gender/age analyzer
    face_detector = None
    gender_age_analyzer = None
    if ENABLE_FACE_DETECTION:
        print("Loading face detector...")
        face_detector = FaceDetector(min_confidence=FACE_DETECTION_CONFIDENCE)
        gender_age_analyzer = GenderAgeAnalyzer(use_threading=USE_THREADED_ANALYSIS)
        print("Face detector and DeepFace analyzer loaded!")
        print(f"  - Face detection every {FACE_DETECTION_INTERVAL} frames")
        print(f"  - Gender/Age analysis every {GENDER_AGE_INTERVAL} frames")
        print(f"  - Threaded analysis: {USE_THREADED_ANALYSIS}")
    
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
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset counts")
    print("  'f' - Toggle face detection")
    print("  'c' - Clear gender/age cache")
    print("  's' - Save current report")
    
    # FPS calculation
    prev_time = time.time()
    face_detection_enabled = ENABLE_FACE_DETECTION
    frame_count = 0
    fps = 0
    
    # Cache for drawing (persists between detection frames)
    cached_faces_by_person = {}
    
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
        
        # Face detection and gender/age analysis (with frame skipping for performance)
        if face_detection_enabled and face_detector and len(objects) > 0:
            # Run face detection every N frames
            if frame_count % FACE_DETECTION_INTERVAL == 0:
                cached_faces_by_person = face_detector.detect_faces_in_rois(frame, objects)
            
            # Gender/Age analysis every N frames (more expensive)
            if gender_age_analyzer and frame_count % GENDER_AGE_INTERVAL == 0:
                # Get crops for persons without cached results
                crops_to_analyze = {}
                for person_id, faces in cached_faces_by_person.items():
                    if not gender_age_analyzer.get_cached(person_id) and faces:
                        crops = face_detector.get_face_crops(frame, {person_id: faces})
                        if person_id in crops and crops[person_id]:
                            crops_to_analyze[person_id] = crops[person_id]
                
                # Analyze new faces (non-blocking if threaded)
                if crops_to_analyze:
                    gender_age_analyzer.analyze_faces(crops_to_analyze)
                
                # Clean up old cached IDs
                gender_age_analyzer.clear_old_ids(objects.keys())
        
        # Check for new line crossings and record to statistics
        count_in, count_out, inside = tracker.get_counts()
        
        # Get recent crossings with specific person IDs
        recent_crossings = tracker.get_recent_crossings()
        
        for crossing in recent_crossings:
            person_id = crossing["id"]
            direction = crossing["direction"]
            
            # Get gender/age for this SPECIFIC person
            gender = None
            age_group = None
            
            if gender_age_analyzer:
                cached = gender_age_analyzer.get_cached(person_id)
                if cached:
                    gender = cached.get("gender")
                    age_group = cached.get("age_group")
                
                # If no cached result, try to force analyze NOW before they leave
                if (gender is None or age_group is None) and face_detector:
                    # Check if this person is still in frame
                    if person_id in objects:
                        # Get their face and analyze immediately
                        person_faces = cached_faces_by_person.get(person_id, [])
                        if person_faces:
                            crops = face_detector.get_face_crops(frame, {person_id: person_faces})
                            if person_id in crops and crops[person_id]:
                                # Force synchronous analysis
                                gender_age_analyzer._analyze_single_face(person_id, crops[person_id][0])
                                # Check cache again
                                cached = gender_age_analyzer.get_cached(person_id)
                                if cached:
                                    gender = cached.get("gender")
                                    age_group = cached.get("age_group")
            
            # Record to daily statistics
            daily_stats.record_person(
                person_id=person_id,
                direction=direction,
                gender=gender,
                age_group=age_group
            )
            
            status = f"{gender or 'Pending'} | {age_group or 'Pending'}"
            print(f"[Stats] Recorded: ID {person_id} -> {direction.upper()} | {status}")
        
        # Try to update any pending records with newly available gender/age data
        if gender_age_analyzer and hasattr(daily_stats, 'pending_updates') and daily_stats.pending_updates:
            for crossing_key in list(daily_stats.pending_updates.keys()):
                # Extract person_id from crossing_key (format: "id_direction")
                parts = crossing_key.rsplit('_', 1)
                if len(parts) == 2:
                    try:
                        pid = int(parts[0])
                        dir_val = parts[1]
                        cached = gender_age_analyzer.get_cached(pid)
                        if cached:
                            gender = cached.get("gender")
                            age_group = cached.get("age_group")
                            if gender or age_group:
                                daily_stats.update_pending(pid, dir_val, gender, age_group)
                                print(f"[Stats] Updated pending: ID {pid} -> {gender} | {age_group}")
                    except (ValueError, TypeError):
                        pass
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Prepare data for visualizer
        gender_age_data = {}
        if gender_age_analyzer:
            for person_id in objects.keys():
                cached = gender_age_analyzer.get_cached(person_id)
                if cached:
                    gender_age_data[person_id] = cached
        
        draw_data = {
            'objects': objects,
            'faces': cached_faces_by_person if face_detection_enabled else {},
            'gender_age': gender_age_data,
            'count_in': count_in,
            'count_out': count_out,
            'inside': inside,
            'line_x': line_x,
            'fps': fps,
            'face_detection_on': face_detection_enabled
        }
        
        # Draw everything using visualizer
        visualizer.draw(frame, draw_data)
        
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
        elif key == ord('s'):
            # Force save current report
            filepath = daily_stats.save_report()
            if filepath:
                print(f"Report saved to: {filepath}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if face_detector:
        face_detector.close()
    if gender_age_analyzer:
        gender_age_analyzer.stop()
    
    # Stop statistics thread
    daily_stats.stop()
    
    # Save final report
    print("\nSaving final report...")
    daily_stats.save_report()
    
    # Final stats
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
    
    # Show daily stats
    daily = daily_stats.get_current_stats()
    print(f"\nDaily Statistics ({daily['date']}):")
    print(f"  Total IN: {daily['total_in']}")
    print(f"  Total OUT: {daily['total_out']}")
    
    print(f"{'='*50}")
    print("Camera closed.")


if __name__ == "__main__":
    main()
