import cv2
import numpy as np
from deepface import DeepFace


class GenderAgeAnalyzer:
    def __init__(self):
        """
        Gender and age analyzer using DeepFace.
        Results are cached per person ID to avoid repeated analysis.
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
        self.target_size = (224, 224)  # DeepFace expected input size
    
    def _age_to_group(self, age):
        """Convert raw age to age group string."""
        for min_age, max_age, group in self.age_groups:
            if min_age <= age <= max_age:
                return group
        return ">60"  # Default for very old
    
    def _preprocess_face(self, face_crop):
        """
        Preprocess face crop for better DeepFace analysis.
        
        Args:
            face_crop: BGR face image
            
        Returns:
            Preprocessed BGR image or None if too small
        """
        if face_crop is None or face_crop.size == 0:
            return None
        
        h, w = face_crop.shape[:2]
        
        # Skip if too small
        if h < self.min_face_size or w < self.min_face_size:
            return None
        
        # Add padding around face (helps DeepFace detect better)
        pad_ratio = 0.2
        pad_h = int(h * pad_ratio)
        pad_w = int(w * pad_ratio)
        
        # Create padded image with border replication
        padded = cv2.copyMakeBorder(
            face_crop, 
            pad_h, pad_h, pad_w, pad_w, 
            cv2.BORDER_REPLICATE
        )
        
        # Resize to target size while maintaining aspect ratio
        new_h, new_w = padded.shape[:2]
        scale = min(self.target_size[0] / new_h, self.target_size[1] / new_w)
        
        resized_w = int(new_w * scale)
        resized_h = int(new_h * scale)
        
        resized = cv2.resize(padded, (resized_w, resized_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to exact target size
        delta_w = self.target_size[1] - resized_w
        delta_h = self.target_size[0] - resized_h
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left
        
        final = cv2.copyMakeBorder(
            resized, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=[128, 128, 128]
        )
        
        return final
    
    def analyze_faces(self, face_crops_by_person):
        """
        Analyze gender and age for face crops (only for uncached persons).
        
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
            
            # Skip if no face crops
            if not crops or len(crops) == 0:
                continue
            
            # Try each crop until we get a successful analysis
            analysis_success = False
            
            for face_crop in crops:
                # Preprocess the face
                processed_face = self._preprocess_face(face_crop)
                
                if processed_face is None:
                    continue
                
                try:
                    # Run DeepFace analysis with detection enabled for better accuracy
                    # Using opencv backend which is faster
                    analysis = DeepFace.analyze(
                        processed_face,
                        actions=['gender', 'age'],
                        enforce_detection=False,  # We already have face crops
                        detector_backend='skip',  # Skip detection since we have face crops
                        silent=True
                    )
                    
                    # Handle both single result and list
                    if isinstance(analysis, list):
                        if len(analysis) == 0:
                            continue
                        analysis = analysis[0]
                    
                    # Extract gender
                    gender_data = analysis.get('gender', {})
                    if isinstance(gender_data, dict):
                        # Get probabilities
                        man_prob = gender_data.get('Man', 0)
                        woman_prob = gender_data.get('Woman', 0)
                        gender = "Male" if man_prob > woman_prob else "Female"
                    elif isinstance(gender_data, str):
                        gender = "Male" if "man" in gender_data.lower() else "Female"
                    else:
                        continue
                    
                    # Extract age
                    age_raw = analysis.get('age', None)
                    if age_raw is None:
                        continue
                    
                    age_raw = int(age_raw)
                    age_group = self._age_to_group(age_raw)
                    
                    # Cache and return
                    result = {
                        "gender": gender,
                        "age_group": age_group,
                        "age_raw": age_raw
                    }
                    self.cache[person_id] = result
                    results[person_id] = result
                    analysis_success = True
                    
                    print(f"[GenderAge] ID {person_id}: {gender}, {age_raw}y ({age_group})")
                    break  # Success, no need to try other crops
                    
                except Exception as e:
                    # Try next crop
                    continue
            
            if not analysis_success and crops:
                # If all crops failed, try one more time with the largest crop
                largest_crop = max(crops, key=lambda c: c.shape[0] * c.shape[1] if c is not None and c.size > 0 else 0)
                if largest_crop is not None and largest_crop.size > 0:
                    try:
                        # Try with retinaface backend for better detection
                        analysis = DeepFace.analyze(
                            largest_crop,
                            actions=['gender', 'age'],
                            enforce_detection=False,
                            detector_backend='opencv',
                            silent=True
                        )
                        
                        if isinstance(analysis, list) and len(analysis) > 0:
                            analysis = analysis[0]
                        
                        gender_data = analysis.get('gender', {})
                        if isinstance(gender_data, dict):
                            man_prob = gender_data.get('Man', 0)
                            woman_prob = gender_data.get('Woman', 0)
                            gender = "Male" if man_prob > woman_prob else "Female"
                        else:
                            gender = "Male" if "man" in str(gender_data).lower() else "Female"
                        
                        age_raw = int(analysis.get('age', 25))
                        age_group = self._age_to_group(age_raw)
                        
                        result = {
                            "gender": gender,
                            "age_group": age_group,
                            "age_raw": age_raw
                        }
                        self.cache[person_id] = result
                        results[person_id] = result
                        
                        print(f"[GenderAge] ID {person_id}: {gender}, {age_raw}y ({age_group}) [retry]")
                        
                    except Exception as e:
                        pass
        
        return results
    
    def get_cached(self, person_id):
        """Get cached result for a person ID."""
        return self.cache.get(person_id)
    
    def get_all_cached(self):
        """Get all cached results."""
        return self.cache.copy()
    
    def clear_old_ids(self, active_ids):
        """
        Remove cached entries for IDs that are no longer tracked.
        
        Args:
            active_ids: Set or list of currently active person IDs
        """
        active_set = set(active_ids)
        old_ids = [pid for pid in self.cache if pid not in active_set]
        for pid in old_ids:
            del self.cache[pid]
    
    def get_statistics(self):
        """
        Get statistics from all cached analyses.
        
        Returns:
            dict: {
                "total": int,
                "gender": {"Male": int, "Female": int},
                "age_groups": {"<12": int, "13-25": int, ...}
            }
        """
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
