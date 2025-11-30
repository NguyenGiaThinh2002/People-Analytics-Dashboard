"""
Gender and Age analyzer using DeepFace with threading support.
"""

import cv2
import numpy as np
import threading
import queue
from deepface import DeepFace


class GenderAgeAnalyzer:
    def __init__(self, use_threading=True):
        """
        Gender and age analyzer using DeepFace.
        Results are cached per person ID to avoid repeated analysis.
        
        Args:
            use_threading: Run analysis in background thread for better FPS
        """
        self.cache = {}  # {person_id: {"gender": str, "age_group": str, "age_raw": int}}
        
        # Age group boundaries
        self.age_groups = [
            (0, 12, "<12"),
            (13, 25, "13-25"),
            (26, 45, "26-45"),
            (46, 60, "46-60"),
            (61, 200, ">60")
        ]
        
        # Minimum face size for analysis
        self.min_face_size = 48
        self.target_size = (224, 224)
        
        # Threading support
        self.use_threading = use_threading
        self._analysis_queue = queue.Queue(maxsize=10)
        self._stop_thread = False
        
        if use_threading:
            self._analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self._analysis_thread.start()
            print("[GenderAge] Background analysis thread started")
    
    def _age_to_group(self, age):
        """Convert raw age to age group string."""
        for min_age, max_age, group in self.age_groups:
            if min_age <= age <= max_age:
                return group
        return ">60"
    
    def _preprocess_face(self, face_crop):
        """Preprocess face crop for DeepFace analysis."""
        if face_crop is None or face_crop.size == 0:
            return None
        
        h, w = face_crop.shape[:2]
        
        if h < self.min_face_size or w < self.min_face_size:
            return None
        
        # Simple resize to target size (faster than padding)
        try:
            resized = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_LINEAR)
            return resized
        except:
            return None
    
    def _analyze_single_face(self, person_id, face_crop):
        """Analyze a single face (called from worker thread or directly)."""
        if person_id in self.cache:
            return  # Already cached
        
        processed_face = self._preprocess_face(face_crop)
        if processed_face is None:
            return
        
        try:
            analysis = DeepFace.analyze(
                processed_face,
                actions=['gender', 'age'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
            
            if isinstance(analysis, list):
                if len(analysis) == 0:
                    return
                analysis = analysis[0]
            
            # Extract gender
            gender_data = analysis.get('gender', {})
            if isinstance(gender_data, dict):
                man_prob = gender_data.get('Man', 0)
                woman_prob = gender_data.get('Woman', 0)
                gender = "Male" if man_prob > woman_prob else "Female"
            elif isinstance(gender_data, str):
                gender = "Male" if "man" in gender_data.lower() else "Female"
            else:
                return
            
            # Extract age
            age_raw = analysis.get('age', None)
            if age_raw is None:
                return
            
            age_raw = int(age_raw)
            age_group = self._age_to_group(age_raw)
            
            # Cache result
            result = {
                "gender": gender,
                "age_group": age_group,
                "age_raw": age_raw
            }
            self.cache[person_id] = result
            
            print(f"[GenderAge] ID {person_id}: {gender}, {age_raw}y ({age_group})")
            
        except Exception as e:
            pass
    
    def _analysis_worker(self):
        """Background worker thread for face analysis."""
        while not self._stop_thread:
            try:
                # Get item with timeout to allow checking stop flag
                item = self._analysis_queue.get(timeout=0.5)
                if item is None:
                    continue
                
                person_id, face_crop = item
                self._analyze_single_face(person_id, face_crop)
                self._analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                pass
    
    def analyze_faces(self, face_crops_by_person):
        """
        Analyze gender and age for face crops.
        
        Args:
            face_crops_by_person: Dict {person_id: [face_crop_image, ...]}
        
        Returns:
            dict: {person_id: {"gender": str, "age_group": str, "age_raw": int}}
        """
        results = {}
        
        for person_id, crops in face_crops_by_person.items():
            # Return cached result if exists
            if person_id in self.cache:
                results[person_id] = self.cache[person_id]
                continue
            
            if not crops or len(crops) == 0:
                continue
            
            # Get the best (largest) crop
            best_crop = max(crops, key=lambda c: c.shape[0] * c.shape[1] if c is not None and c.size > 0 else 0)
            
            if best_crop is None or best_crop.size == 0:
                continue
            
            if self.use_threading:
                # Queue for background analysis (non-blocking)
                try:
                    self._analysis_queue.put_nowait((person_id, best_crop.copy()))
                except queue.Full:
                    pass  # Queue full, skip this frame
            else:
                # Direct analysis (blocking)
                self._analyze_single_face(person_id, best_crop)
                if person_id in self.cache:
                    results[person_id] = self.cache[person_id]
        
        return results
    
    def get_cached(self, person_id):
        """Get cached result for a person ID."""
        return self.cache.get(person_id)
    
    def get_all_cached(self):
        """Get all cached results."""
        return self.cache.copy()
    
    def clear_old_ids(self, active_ids):
        """Remove cached entries for IDs that are no longer tracked."""
        active_set = set(active_ids)
        old_ids = [pid for pid in self.cache if pid not in active_set]
        for pid in old_ids:
            del self.cache[pid]
    
    def get_statistics(self):
        """Get statistics from all cached analyses."""
        stats = {
            "total": len(self.cache),
            "gender": {"Male": 0, "Female": 0},
            "age_groups": {"<12": 0, "13-25": 0, "26-45": 0, "46-60": 0, ">60": 0}
        }
        
        for data in self.cache.values():
            gender = data.get("gender", "")
            age_group = data.get("age_group", "")
            
            if gender in stats["gender"]:
                stats["gender"][gender] += 1
            if age_group in stats["age_groups"]:
                stats["age_groups"][age_group] += 1
        
        return stats
    
    def stop(self):
        """Stop the background analysis thread."""
        self._stop_thread = True
        if self.use_threading and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=2)
