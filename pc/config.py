"""
Configuration constants for the hand gesture control system.
All configurable parameters are centralized here for easy tuning.
"""

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAMERA_INDEX = 0          # Webcam device index (0 = default camera)
CAMERA_WIDTH = 640        # Frame width in pixels
CAMERA_HEIGHT = 480       # Frame height in pixels
CAMERA_FPS = 30           # Target frames per second

# =============================================================================
# MEDIAPIPE SETTINGS
# =============================================================================
MIN_DETECTION_CONFIDENCE = 0.7   # Minimum confidence for hand detection
MIN_TRACKING_CONFIDENCE = 0.5    # Minimum confidence for hand tracking
MAX_NUM_HANDS = 1                # Maximum number of hands to detect

# =============================================================================
# DEBOUNCER SETTINGS
# =============================================================================
DEBOUNCE_FRAMES = 5              # Frames required to confirm state change
HYSTERESIS_THRESHOLD = 0.6       # Confidence threshold for state change
NO_HAND_TIMEOUT_FRAMES = 30      # Frames without hand before failsafe (OPEN)

# =============================================================================
# SERIAL SETTINGS
# =============================================================================
SERIAL_PORT = "COM3"             # Serial port (Windows: COM3, Linux: /dev/ttyUSB0)
BAUD_RATE = 115200               # Must match Arduino sketch
SERIAL_TIMEOUT = 1.0             # Read timeout in seconds

# =============================================================================
# GESTURE THRESHOLDS
# =============================================================================
# Finger curl detection threshold
# Lower value = finger must be more extended to count as "open"
FINGER_CURL_THRESHOLD = 0.5

# Number of curled fingers to classify as OPEN vs CLOSE
OPEN_MAX_CURLED = 1              # 0-1 curled fingers = OPEN
CLOSE_MIN_CURLED = 4             # 4-5 curled fingers = CLOSE

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
SHOW_DISPLAY = True              # Show webcam feed with overlay
DISPLAY_WINDOW_NAME = "Hand Gesture Control"
OVERLAY_FONT_SCALE = 0.7
OVERLAY_THICKNESS = 2

# =============================================================================
# PER-FINGER CONTROL SETTINGS
# =============================================================================
# Debouncing for individual finger state changes
FINGER_DEBOUNCE_FRAMES = 5       # Frames required to confirm finger state change
FINGER_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for finger state

# Servo pulse calibration (matches Arduino defaults)
# Pulse range: 150-600 (12-bit PWM on PCA9685)
SERVO_OPEN_PULSE = 150           # Fully open position
SERVO_CLOSE_PULSE = 600          # Fully closed position

# Per-finger calibration (can be tuned individually if needed)
# Keys: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
FINGER_OPEN_PULSE = {
    0: 150,  # Thumb
    1: 150,  # Index
    2: 150,  # Middle
    3: 150,  # Ring
    4: 150   # Pinky
}

FINGER_CLOSE_PULSE = {
    0: 600,  # Thumb
    1: 600,  # Index
    2: 600,  # Middle
    3: 600,  # Ring
    4: 600   # Pinky
}

# Hardware channel mapping for PCA9685 servo driver
# Maps finger index to hardware channel
FINGER_CHANNELS = {
    0: 0,    # Thumb  -> Channel 0
    1: 4,    # Index  -> Channel 4
    2: 5,    # Middle -> Channel 5
    3: 6,    # Ring   -> Channel 6
    4: 7     # Pinky  -> Channel 7
}

# Command transmission settings
INTER_COMMAND_DELAY = 0.005      # Delay between SET commands (5ms)
MAX_COMMANDS_PER_FRAME = 5       # Maximum commands to send per frame
