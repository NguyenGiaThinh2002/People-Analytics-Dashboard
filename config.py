# Camera configuration
CAMERA_SOURCE = 0  # 0 = default webcam, or use RTSP URL like "rtsp://user:pass@ip:port/stream"

# Window settings
WINDOW_NAME = "PeopleAnalytics Live"

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection

# Tracking settings
MAX_DISAPPEARED = 30  # Frames before ID is removed
MAX_DISTANCE = 50     # Max pixel distance for centroid matching

# Counting line settings (VERTICAL LINE)
LINE_POSITION = 0.5   # Line position as fraction of frame WIDTH (0.5 = middle)

# Face detection settings
ENABLE_FACE_DETECTION = True  # Set to False to disable face detection
FACE_DETECTION_CONFIDENCE = 0.4  # Minimum confidence for face detection (lowered for better child detection)

# Gender/Age analysis settings
RETRY_ANALYSIS_FRAMES = 30  # Re-analyze after this many frames if confidence was low

# Performance settings
FACE_DETECTION_INTERVAL = 2    # Run face detection every N frames (1 = every frame)
GENDER_AGE_INTERVAL = 3        # Run gender/age analysis every N frames
USE_THREADED_ANALYSIS = True   # Run DeepFace in background thread

