"""
Beautiful visualization for PeopleAnalytics Live.
"""

import cv2
import numpy as np
from datetime import datetime


class Visualizer:
    def __init__(self):
        """Initialize visualizer with color schemes and fonts."""
        # Colors (BGR format)
        self.colors = {
            'male': (255, 150, 50),       # Blue-ish
            'female': (180, 105, 255),    # Pink
            'unknown': (0, 255, 0),       # Green
            'line': (0, 0, 255),          # Red
            'centroid': (0, 255, 255),    # Yellow
            'face': (255, 100, 100),      # Light blue
            'panel_bg': (40, 40, 40),     # Dark gray
            'panel_border': (80, 80, 80), # Gray
            'text_white': (255, 255, 255),
            'text_green': (100, 255, 100),
            'text_yellow': (100, 255, 255),
            'arrow_in': (100, 255, 100),  # Green for IN
            'arrow_out': (100, 100, 255), # Red-ish for OUT
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_small = 0.5
        self.font_scale_medium = 0.7
        self.font_scale_large = 1.0
        self.font_scale_xlarge = 1.5
        self.thickness_thin = 1
        self.thickness_medium = 2
        self.thickness_bold = 3
    
    def _get_gender_color(self, gender):
        """Get color based on gender."""
        if gender == "Male":
            return self.colors['male']
        elif gender == "Female":
            return self.colors['female']
        else:
            return self.colors['unknown']
    
    def _draw_rounded_rect(self, frame, pt1, pt2, color, thickness=1, radius=10):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw the rectangle with rounded corners approximation
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def _draw_filled_rounded_rect(self, frame, pt1, pt2, color, radius=10):
        """Draw a filled rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw filled rectangles
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Draw filled corners
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    
    def draw_person_box(self, frame, x1, y1, x2, y2, person_id, gender=None, age_group=None):
        """Draw a person bounding box with label."""
        color = self._get_gender_color(gender)
        
        # Draw box with thicker border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw corner accents
        corner_len = 15
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
        
        # Build label
        label_parts = [f"ID:{person_id}"]
        if gender:
            label_parts.append(gender[0])  # M or F
        if age_group:
            label_parts.append(age_group)
        label = " | ".join(label_parts)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, self.font, self.font_scale_small, self.thickness_medium)
        label_w, label_h = label_size
        
        # Label above the box
        label_x = x1
        label_y = y1 - 5
        
        # Background with slight transparency effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x - 2, label_y - label_h - 8), 
                     (label_x + label_w + 6, label_y + 2), color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Text
        cv2.putText(frame, label, (label_x + 2, label_y - 3), 
                   self.font, self.font_scale_small, (0, 0, 0), self.thickness_medium)
    
    def draw_centroid(self, frame, cx, cy):
        """Draw centroid marker."""
        cv2.circle(frame, (cx, cy), 6, self.colors['centroid'], -1)
        cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 1)
    
    def draw_face_box(self, frame, x1, y1, x2, y2):
        """Draw face detection box."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['face'], 2)
    
    def draw_counting_line(self, frame, line_x, frame_height):
        """Draw the counting line with direction arrows."""
        # Main line (dashed effect)
        dash_length = 20
        gap_length = 10
        y = 0
        while y < frame_height:
            cv2.line(frame, (line_x, y), (line_x, min(y + dash_length, frame_height)), 
                    self.colors['line'], 3)
            y += dash_length + gap_length
        
        # Draw direction arrows and labels
        arrow_y_in = frame_height // 3
        arrow_y_out = 2 * frame_height // 3
        arrow_len = 40
        
        # IN arrow (pointing right) →
        cv2.arrowedLine(frame, (line_x - arrow_len, arrow_y_in), 
                       (line_x + arrow_len, arrow_y_in), 
                       self.colors['arrow_in'], 3, tipLength=0.3)
        cv2.putText(frame, "IN", (line_x + arrow_len + 10, arrow_y_in + 5), 
                   self.font, self.font_scale_medium, self.colors['arrow_in'], self.thickness_medium)
        
        # OUT arrow (pointing left) ←
        cv2.arrowedLine(frame, (line_x + arrow_len, arrow_y_out), 
                       (line_x - arrow_len, arrow_y_out), 
                       self.colors['arrow_out'], 3, tipLength=0.3)
        cv2.putText(frame, "OUT", (line_x - arrow_len - 50, arrow_y_out + 5), 
                   self.font, self.font_scale_medium, self.colors['arrow_out'], self.thickness_medium)
    
    def draw_counter_panel(self, frame, count_in, count_out, inside, frame_width):
        """Draw the main counter panel at top."""
        panel_height = 80
        panel_margin = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_margin, panel_margin), 
                     (frame_width - panel_margin, panel_height), 
                     self.colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_margin, panel_margin), 
                     (frame_width - panel_margin, panel_height), 
                     self.colors['panel_border'], 2)
        
        # Calculate positions for 3 counters
        section_width = (frame_width - 2 * panel_margin) // 3
        
        # IN counter
        self._draw_counter_item(frame, panel_margin + section_width // 2, 
                               panel_height // 2, "IN", count_in, self.colors['arrow_in'])
        
        # OUT counter
        self._draw_counter_item(frame, panel_margin + section_width + section_width // 2, 
                               panel_height // 2, "OUT", count_out, self.colors['arrow_out'])
        
        # INSIDE counter
        self._draw_counter_item(frame, panel_margin + 2 * section_width + section_width // 2, 
                               panel_height // 2, "INSIDE", inside, self.colors['text_yellow'])
        
        # Divider lines
        cv2.line(frame, (panel_margin + section_width, panel_margin + 10), 
                (panel_margin + section_width, panel_height - 10), 
                self.colors['panel_border'], 1)
        cv2.line(frame, (panel_margin + 2 * section_width, panel_margin + 10), 
                (panel_margin + 2 * section_width, panel_height - 10), 
                self.colors['panel_border'], 1)
    
    def _draw_counter_item(self, frame, center_x, center_y, label, value, color):
        """Draw a single counter item."""
        # Label
        label_size, _ = cv2.getTextSize(label, self.font, self.font_scale_small, self.thickness_thin)
        cv2.putText(frame, label, (center_x - label_size[0] // 2, center_y - 15), 
                   self.font, self.font_scale_small, (180, 180, 180), self.thickness_thin)
        
        # Value
        value_str = str(value)
        value_size, _ = cv2.getTextSize(value_str, self.font, self.font_scale_xlarge, self.thickness_bold)
        cv2.putText(frame, value_str, (center_x - value_size[0] // 2, center_y + 25), 
                   self.font, self.font_scale_xlarge, color, self.thickness_bold)
    
    def draw_info_panel(self, frame, fps, face_detection_on, frame_width, frame_height):
        """Draw FPS and date/time info panel at bottom."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        panel_height = 35
        panel_y = frame_height - panel_height - 5
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, panel_y), (frame_width - 5, frame_height - 5), 
                     self.colors['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        fps_color = self.colors['text_green'] if fps >= 20 else self.colors['text_yellow']
        cv2.putText(frame, fps_text, (15, panel_y + 23), 
                   self.font, self.font_scale_medium, fps_color, self.thickness_medium)
        
        # Face detection status
        face_status = "Face: ON" if face_detection_on else "Face: OFF"
        face_color = self.colors['text_green'] if face_detection_on else (128, 128, 128)
        cv2.putText(frame, face_status, (130, panel_y + 23), 
                   self.font, self.font_scale_medium, face_color, self.thickness_medium)
        
        # Date and time (right side)
        datetime_text = f"{date_str}  {time_str}"
        dt_size, _ = cv2.getTextSize(datetime_text, self.font, self.font_scale_medium, self.thickness_medium)
        cv2.putText(frame, datetime_text, (frame_width - dt_size[0] - 15, panel_y + 23), 
                   self.font, self.font_scale_medium, self.colors['text_white'], self.thickness_medium)
    
    def draw(self, frame, data):
        """
        Main draw function - renders everything.
        
        Args:
            frame: The video frame to draw on
            data: Dictionary containing:
                - objects: {id: (cx, cy, x1, y1, x2, y2)}
                - faces: {person_id: [(x1, y1, x2, y2), ...]}
                - gender_age: {person_id: {"gender": str, "age_group": str}}
                - count_in: int
                - count_out: int
                - inside: int
                - line_x: int
                - fps: float
                - face_detection_on: bool
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Draw counting line first (behind everything)
        line_x = data.get('line_x', frame_width // 2)
        self.draw_counting_line(frame, line_x, frame_height)
        
        # Draw face boxes
        faces = data.get('faces', {})
        for person_id, face_list in faces.items():
            for (fx1, fy1, fx2, fy2) in face_list:
                self.draw_face_box(frame, fx1, fy1, fx2, fy2)
        
        # Draw person boxes and centroids
        objects = data.get('objects', {})
        gender_age = data.get('gender_age', {})
        
        for person_id, obj_data in objects.items():
            cx, cy, x1, y1, x2, y2 = obj_data
            
            # Get gender and age info
            ga_info = gender_age.get(person_id, {})
            gender = ga_info.get('gender')
            age_group = ga_info.get('age_group')
            
            # Draw person box
            self.draw_person_box(frame, x1, y1, x2, y2, person_id, gender, age_group)
            
            # Draw centroid
            self.draw_centroid(frame, cx, cy)
        
        # Draw counter panel (on top)
        count_in = data.get('count_in', 0)
        count_out = data.get('count_out', 0)
        inside = data.get('inside', 0)
        self.draw_counter_panel(frame, count_in, count_out, inside, frame_width)
        
        # Draw info panel at bottom
        fps = data.get('fps', 0)
        face_detection_on = data.get('face_detection_on', True)
        self.draw_info_panel(frame, fps, face_detection_on, frame_width, frame_height)
        
        return frame

