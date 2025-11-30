"""
Streamlit Dashboard for PeopleAnalytics Live.
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import cv2
import time
import os
import sys
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from config import (
    CAMERA_SOURCE, CONFIDENCE_THRESHOLD,
    MAX_DISAPPEARED, MAX_DISTANCE, LINE_POSITION,
    ENABLE_FACE_DETECTION, FACE_DETECTION_CONFIDENCE,
    FACE_DETECTION_INTERVAL, GENDER_AGE_INTERVAL, USE_THREADED_ANALYSIS
)
from utils.tracker import CentroidTracker
from utils.face_detector import FaceDetector
from utils.gender_age import GenderAgeAnalyzer
from utils.visualizer import Visualizer

# Page config
st.set_page_config(
    page_title="PeopleAnalytics Live Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-counter {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .counter-in {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        color: #4ade80;
    }
    .counter-out {
        background: linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 100%);
        color: #f87171;
    }
    .counter-inside {
        background: linear-gradient(135deg, #1a3a4a 0%, #2d4a5a 100%);
        color: #60a5fa;
    }
    .counter-label {
        font-size: 16px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Shared state between video thread and Streamlit."""
    def __init__(self):
        self.frame = None
        self.count_in = 0
        self.count_out = 0
        self.inside = 0
        self.fps = 0
        self.gender_counts = {"Male": 0, "Female": 0}
        self.age_counts = {"<12": 0, "13-25": 0, "26-45": 0, "46-60": 0, ">60": 0}
        self.hourly_counts = defaultdict(lambda: {"in": 0, "out": 0})
        self.is_running = False
        self.stop_requested = False
        self.lock = threading.Lock()
        self.counted_persons = set()


@st.cache_resource
def get_shared_state():
    """Get or create shared state (persists across reruns)."""
    return DashboardState()


def run_pipeline(state):
    """Run the full detection pipeline in a separate thread."""
    
    print("[Pipeline] Starting...")
    
    # Load models
    model = YOLO("yolov8n.pt")
    tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE)
    visualizer = Visualizer()
    
    face_detector = None
    gender_age_analyzer = None
    if ENABLE_FACE_DETECTION:
        face_detector = FaceDetector(min_confidence=FACE_DETECTION_CONFIDENCE)
        gender_age_analyzer = GenderAgeAnalyzer(use_threading=USE_THREADED_ANALYSIS)
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("[Pipeline] Error: Could not open camera")
        state.is_running = False
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_x = int(frame_width * LINE_POSITION)
    
    print(f"[Pipeline] Camera opened: {frame_width}x{frame_height}")
    
    state.is_running = True
    state.stop_requested = False
    prev_time = time.time()
    frame_count = 0
    cached_faces_by_person = {}
    
    while not state.stop_requested:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # YOLO detection
        results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
        
        # Update tracker
        objects = tracker.update(bboxes, line_x)
        
        # Face detection
        if face_detector and len(objects) > 0:
            if frame_count % FACE_DETECTION_INTERVAL == 0:
                cached_faces_by_person = face_detector.detect_faces_in_rois(frame, objects)
            
            if gender_age_analyzer and frame_count % GENDER_AGE_INTERVAL == 0:
                crops_to_analyze = {}
                for person_id, faces in cached_faces_by_person.items():
                    if not gender_age_analyzer.get_cached(person_id) and faces:
                        crops = face_detector.get_face_crops(frame, {person_id: faces})
                        if person_id in crops and crops[person_id]:
                            crops_to_analyze[person_id] = crops[person_id]
                
                if crops_to_analyze:
                    gender_age_analyzer.analyze_faces(crops_to_analyze)
                
                gender_age_analyzer.clear_old_ids(objects.keys())
        
        # Get counts
        count_in, count_out, inside = tracker.get_counts()
        
        # Process crossings for statistics
        recent_crossings = tracker.get_recent_crossings()
        current_hour = datetime.now().hour
        
        with state.lock:
            # Initialize pending_gender_age if not exists
            if not hasattr(state, 'pending_gender_age'):
                state.pending_gender_age = {}
            
            for crossing in recent_crossings:
                person_id = crossing["id"]
                direction = crossing["direction"]
                crossing_key = f"{person_id}_{direction}"
                
                if crossing_key not in state.counted_persons:
                    state.counted_persons.add(crossing_key)
                    
                    # Update hourly counts (always)
                    state.hourly_counts[current_hour][direction] += 1
                    
                    # Try to get gender/age
                    gender = None
                    age_group = None
                    
                    if gender_age_analyzer:
                        cached = gender_age_analyzer.get_cached(person_id)
                        if cached:
                            gender = cached.get("gender")
                            age_group = cached.get("age_group")
                        
                        # Force analysis NOW if not cached
                        if (gender is None or age_group is None) and face_detector:
                            if person_id in objects:
                                person_faces = cached_faces_by_person.get(person_id, [])
                                if person_faces:
                                    crops = face_detector.get_face_crops(frame, {person_id: person_faces})
                                    if person_id in crops and crops[person_id]:
                                        gender_age_analyzer._analyze_single_face(person_id, crops[person_id][0])
                                        cached = gender_age_analyzer.get_cached(person_id)
                                        if cached:
                                            gender = cached.get("gender")
                                            age_group = cached.get("age_group")
                    
                    # Only update if we have valid data
                    if gender in ["Male", "Female"]:
                        state.gender_counts[gender] += 1
                    else:
                        # Add to pending
                        state.pending_gender_age[crossing_key] = person_id
                    
                    if age_group in ["<12", "13-25", "26-45", "46-60", ">60"]:
                        state.age_counts[age_group] += 1
            
            # Try to update pending records
            if gender_age_analyzer and state.pending_gender_age:
                for crossing_key, pid in list(state.pending_gender_age.items()):
                    cached = gender_age_analyzer.get_cached(pid)
                    if cached:
                        gender = cached.get("gender")
                        age_group = cached.get("age_group")
                        
                        if gender in ["Male", "Female"]:
                            state.gender_counts[gender] += 1
                            del state.pending_gender_age[crossing_key]
                        
                        if age_group in ["<12", "13-25", "26-45", "46-60", ">60"]:
                            state.age_counts[age_group] += 1
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Prepare visualization data
        gender_age_data = {}
        if gender_age_analyzer:
            for person_id in objects.keys():
                cached = gender_age_analyzer.get_cached(person_id)
                if cached:
                    gender_age_data[person_id] = cached
        
        draw_data = {
            'objects': objects,
            'faces': cached_faces_by_person,
            'gender_age': gender_age_data,
            'count_in': count_in,
            'count_out': count_out,
            'inside': inside,
            'line_x': line_x,
            'fps': fps,
            'face_detection_on': True
        }
        
        # Draw on frame
        visualizer.draw(frame, draw_data)
        
        # Update shared state
        with state.lock:
            state.frame = frame.copy()
            state.count_in = count_in
            state.count_out = count_out
            state.inside = inside
            state.fps = fps
    
    # Cleanup
    print("[Pipeline] Stopping...")
    cap.release()
    if face_detector:
        face_detector.close()
    if gender_age_analyzer:
        gender_age_analyzer.stop()
    state.is_running = False
    print("[Pipeline] Stopped")


def get_latest_report():
    """Find the latest Excel report file."""
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    if not os.path.exists(report_dir):
        return None
    
    files = glob.glob(os.path.join(report_dir, "report_*.xlsx"))
    if not files:
        return None
    
    return max(files, key=os.path.getctime)


def main():
    st.title("👥 PeopleAnalytics Live Dashboard")
    
    # Get shared state
    state = get_shared_state()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Show current status
        if state.is_running:
            st.success("🟢 Pipeline Running")
        else:
            st.warning("🔴 Pipeline Stopped")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True, disabled=state.is_running):
                state.stop_requested = False
                thread = threading.Thread(target=run_pipeline, args=(state,), daemon=True)
                thread.start()
                time.sleep(1)  # Give thread time to start
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not state.is_running):
                state.stop_requested = True
                time.sleep(1)
                st.rerun()
        
        st.divider()
        
        # Download report
        st.header("📥 Reports")
        latest_report = get_latest_report()
        if latest_report and os.path.exists(latest_report):
            with open(latest_report, "rb") as f:
                st.download_button(
                    label="📊 Download Latest Report",
                    data=f.read(),
                    file_name=os.path.basename(latest_report),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.info("No reports available yet")
        
        st.divider()
        
        # Status info
        st.header("📊 Info")
        st.write(f"FPS: {state.fps:.1f}")
        st.write(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Refresh button
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Main content
    # Row 1: Counters
    st.subheader("📈 Real-Time Counters")
    col1, col2, col3 = st.columns(3)
    
    with state.lock:
        count_in = state.count_in
        count_out = state.count_out
        inside = state.inside
        gender_counts = dict(state.gender_counts)
        age_counts = dict(state.age_counts)
        hourly_counts = dict(state.hourly_counts)
        frame = state.frame.copy() if state.frame is not None else None
    
    with col1:
        st.markdown(f"""
        <div class="big-counter counter-in">
            <div class="counter-label">IN</div>
            <div>{count_in}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="big-counter counter-out">
            <div class="counter-label">OUT</div>
            <div>{count_out}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="big-counter counter-inside">
            <div class="counter-label">INSIDE</div>
            <div>{inside}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: Video feed and charts
    col_video, col_charts = st.columns([2, 1])
    
    with col_video:
        st.subheader("📹 Live Feed")
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", width="stretch")
        else:
            st.info("📷 Waiting for video feed... Click 'Start' to begin.")
    
    with col_charts:
        st.subheader("📊 Demographics")
        
        # Gender pie chart
        gender_data = [v for v in gender_counts.values() if v > 0]
        gender_labels = [k for k, v in gender_counts.items() if v > 0]
        
        if gender_data:
            fig_gender = px.pie(
                values=gender_data,
                names=gender_labels,
                title="Gender Distribution",
                color=gender_labels,
                color_discrete_map={"Male": "#3b82f6", "Female": "#ec4899"},
                hole=0.4
            )
            fig_gender.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=250,
                margin=dict(t=30, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_gender, key="gender_pie")
        else:
            st.info("No gender data yet")
        
        # Age pie chart
        age_data = [v for v in age_counts.values() if v > 0]
        age_labels = [k for k, v in age_counts.items() if v > 0]
        
        if age_data:
            fig_age = px.pie(
                values=age_data,
                names=age_labels,
                title="Age Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_age.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=250,
                margin=dict(t=30, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_age, key="age_pie")
        else:
            st.info("No age data yet")
    
    # Row 3: Hourly chart
    st.subheader("📈 Visitors Per Hour")
    
    hours = list(range(24))
    hourly_in = [hourly_counts.get(h, {}).get("in", 0) for h in hours]
    hourly_out = [hourly_counts.get(h, {}).get("out", 0) for h in hours]
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in hours],
        y=hourly_in,
        name="IN",
        line=dict(color="#4ade80", width=3),
        fill='tozeroy',
        fillcolor='rgba(74, 222, 128, 0.2)'
    ))
    fig_hourly.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in hours],
        y=hourly_out,
        name="OUT",
        line=dict(color="#f87171", width=3),
        fill='tozeroy',
        fillcolor='rgba(248, 113, 113, 0.2)'
    ))
    fig_hourly.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig_hourly, key="hourly_line")
    
    # Auto-refresh when running
    if state.is_running:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
