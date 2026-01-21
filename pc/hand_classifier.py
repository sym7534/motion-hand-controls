"""
Hand gesture classification module using MediaPipe Hands.
Detects hands and classifies gestures as OPEN, CLOSE, or NO_HAND.
"""

import cv2
import logging
from enum import Enum, auto
from typing import Tuple, Optional, List
import numpy as np
import mediapipe as mp

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    OPEN_MAX_CURLED,
    CLOSE_MIN_CURLED,
    FINGER_CHANNELS,
    FINGER_OPEN_PULSE,
    FINGER_CLOSE_PULSE
)

logger = logging.getLogger(__name__)


class GestureState(Enum):
    """Possible hand gesture states."""
    NO_HAND = auto()
    OPEN = auto()
    CLOSE = auto()


class FingerState:
    """
    Represents the state of all five fingers.

    Each finger can be independently open (True) or closed (False).
    Provides methods to map finger states to hardware channels and PWM pulse values.
    """

    # Finger indices
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4

    # Finger names for display
    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def __init__(self):
        """Initialize with all fingers open."""
        # Binary state for each finger (True=open, False=closed)
        self.fingers = [True, True, True, True, True]

    def get_pulse(self, finger_index: int) -> int:
        """
        Get PWM pulse value for a finger based on its state.

        Args:
            finger_index: Finger index (0-4)

        Returns:
            PWM pulse value (150=open, 600=closed)
        """
        if finger_index < 0 or finger_index >= 5:
            logger.warning(f"Invalid finger index: {finger_index}")
            return FINGER_OPEN_PULSE[0]

        is_open = self.fingers[finger_index]
        return FINGER_OPEN_PULSE[finger_index] if is_open else FINGER_CLOSE_PULSE[finger_index]

    def get_channel(self, finger_index: int) -> int:
        """
        Get hardware channel for a finger.

        Args:
            finger_index: Finger index (0-4)

        Returns:
            PCA9685 channel number
        """
        return FINGER_CHANNELS[finger_index]

    def get_channel_pulse(self, finger_index: int) -> Tuple[int, int]:
        """
        Get (channel, pulse) tuple for SET command.

        Args:
            finger_index: Finger index (0-4)

        Returns:
            Tuple of (channel, pulse)
        """
        return (self.get_channel(finger_index), self.get_pulse(finger_index))

    def __str__(self) -> str:
        """String representation showing finger states."""
        states = ["OPEN" if f else "CLOSED" for f in self.fingers]
        return f"FingerState({', '.join(f'{self.FINGER_NAMES[i]}:{states[i]}' for i in range(5))})"

    def __repr__(self) -> str:
        return self.__str__()


class HandClassifier:
    """
    MediaPipe-based hand gesture classifier.

    Detects hands in video frames and classifies gestures based on
    finger curl analysis.
    """

    # MediaPipe landmark indices
    WRIST = 0
    THUMB_TIP = 4
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_MCP = 9
    RING_TIP = 16
    RING_MCP = 13
    PINKY_TIP = 20
    PINKY_MCP = 17

    # Finger landmark pairs: (tip_index, mcp_index)
    FINGER_LANDMARKS = [
        (8, 5),    # Index
        (12, 9),   # Middle
        (16, 13),  # Ring
        (20, 17),  # Pinky
    ]

    def __init__(
        self,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        max_num_hands: int = MAX_NUM_HANDS
    ):
        """
        Initialize the hand classifier.

        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to detect
        """
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self._last_landmarks = None
        logger.info("HandClassifier initialized")

    def classify(self, frame: np.ndarray) -> Tuple[GestureState, float]:
        """
        Classify the hand gesture in the given frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            Tuple of (GestureState, confidence)
            - confidence is 0.0-1.0, higher means more certain
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        try:
            results = self._hands.process(rgb_frame)
        except Exception as e:
            logger.warning(f"MediaPipe processing error: {e}")
            return GestureState.NO_HAND, 0.0

        # Check if any hands detected
        if not results.multi_hand_landmarks:
            self._last_landmarks = None
            return GestureState.NO_HAND, 0.0

        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        self._last_landmarks = hand_landmarks

        # Determine handedness for thumb analysis
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        # Count curled fingers
        curled_count = self._count_curled_fingers(hand_landmarks, handedness)

        # Classify based on curled finger count
        state, confidence = self._classify_from_curl_count(curled_count)

        logger.debug(
            f"Curled fingers: {curled_count}, State: {state.name}, "
            f"Confidence: {confidence:.2f}"
        )

        return state, confidence

    def _count_curled_fingers(
        self,
        landmarks,
        handedness: str
    ) -> int:
        """
        Count how many fingers are curled (closed).

        Args:
            landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"

        Returns:
            Number of curled fingers (0-5)
        """
        curled = 0

        # Check thumb (uses X-axis comparison, direction depends on hand)
        thumb_tip = landmarks.landmark[self.THUMB_TIP]
        thumb_mcp = landmarks.landmark[self.THUMB_MCP]

        if handedness == "Right":
            # Right hand: thumb curled if tip is to the right of MCP
            if thumb_tip.x > thumb_mcp.x:
                curled += 1
        else:
            # Left hand: thumb curled if tip is to the left of MCP
            if thumb_tip.x < thumb_mcp.x:
                curled += 1

        # Check other fingers (use Y-axis comparison)
        # Finger is curled if tip Y > MCP Y (tip is below MCP in image coords)
        for tip_idx, mcp_idx in self.FINGER_LANDMARKS:
            tip = landmarks.landmark[tip_idx]
            mcp = landmarks.landmark[mcp_idx]

            if tip.y > mcp.y:
                curled += 1

        return curled

    def _classify_from_curl_count(
        self,
        curled_count: int
    ) -> Tuple[GestureState, float]:
        """
        Classify gesture based on number of curled fingers.

        Args:
            curled_count: Number of curled fingers (0-5)

        Returns:
            Tuple of (GestureState, confidence)
        """
        if curled_count <= OPEN_MAX_CURLED:
            # Open hand (0-1 fingers curled)
            # Higher confidence when fewer fingers curled
            confidence = 1.0 - (curled_count * 0.15)
            return GestureState.OPEN, confidence

        elif curled_count >= CLOSE_MIN_CURLED:
            # Closed fist (4-5 fingers curled)
            # Higher confidence when more fingers curled
            confidence = 0.85 + ((curled_count - CLOSE_MIN_CURLED) * 0.15)
            return GestureState.CLOSE, confidence

        else:
            # Ambiguous (2-3 fingers curled)
            # Return the closer state with low confidence
            if curled_count == 2:
                return GestureState.OPEN, 0.4
            else:  # curled_count == 3
                return GestureState.CLOSE, 0.4

    def classify_fingers(self, frame: np.ndarray) -> Tuple[FingerState, List[float]]:
        """
        Classify individual finger states (open/closed) from the frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            Tuple of (FingerState, confidences)
            - FingerState: Binary state for each of 5 fingers
            - confidences: List of 5 confidence values (0.0-1.0)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        try:
            results = self._hands.process(rgb_frame)
        except Exception as e:
            logger.warning(f"MediaPipe processing error: {e}")
            # Return default state (all open) with zero confidence
            finger_state = FingerState()
            return finger_state, [0.0] * 5

        # Check if any hands detected
        if not results.multi_hand_landmarks:
            self._last_landmarks = None
            # No hand detected - default to all open with zero confidence
            finger_state = FingerState()
            return finger_state, [0.0] * 5

        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        self._last_landmarks = hand_landmarks

        # Determine handedness for thumb analysis
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        # Analyze each finger individually
        finger_curled, confidences = self._analyze_finger_states(hand_landmarks, handedness)

        # Create FingerState (invert curled to get open state)
        finger_state = FingerState()
        finger_state.fingers = [not curled for curled in finger_curled]

        logger.debug(
            f"Finger states: {finger_state}, "
            f"Confidences: {[f'{c:.2f}' for c in confidences]}"
        )

        return finger_state, confidences

    def _analyze_finger_states(
        self,
        landmarks,
        handedness: str
    ) -> Tuple[List[bool], List[float]]:
        """
        Analyze curl state for each individual finger.

        Args:
            landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"

        Returns:
            Tuple of (curled_list, confidence_list)
            - curled_list: List of 5 bools (True if finger is curled/closed)
            - confidence_list: List of 5 confidence values (0.0-1.0)
        """
        curled = [False] * 5
        confidences = [0.0] * 5

        # Analyze Thumb (special case - X-axis comparison)
        thumb_tip = landmarks.landmark[self.THUMB_TIP]
        thumb_mcp = landmarks.landmark[self.THUMB_MCP]

        if handedness == "Right":
            # Right hand: thumb curled if tip is to the right of MCP
            thumb_curled = thumb_tip.x > thumb_mcp.x
        else:
            # Left hand: thumb curled if tip is to the left of MCP
            thumb_curled = thumb_tip.x < thumb_mcp.x

        # Calculate confidence based on X-axis distance
        thumb_dist = abs(thumb_tip.x - thumb_mcp.x)
        curled[FingerState.THUMB] = thumb_curled
        confidences[FingerState.THUMB] = min(thumb_dist * 5.0, 1.0)  # Scale to 0-1

        # Analyze other fingers (Y-axis comparison)
        finger_landmarks_map = [
            (self.INDEX_TIP, self.INDEX_MCP, FingerState.INDEX),
            (self.MIDDLE_TIP, self.MIDDLE_MCP, FingerState.MIDDLE),
            (self.RING_TIP, self.RING_MCP, FingerState.RING),
            (self.PINKY_TIP, self.PINKY_MCP, FingerState.PINKY)
        ]

        for tip_idx, mcp_idx, finger_idx in finger_landmarks_map:
            tip = landmarks.landmark[tip_idx]
            mcp = landmarks.landmark[mcp_idx]

            # Finger is curled if tip.y > mcp.y (tip below knuckle in image coords)
            finger_curled = tip.y > mcp.y

            # Calculate confidence based on Y-axis distance
            y_dist = abs(tip.y - mcp.y)

            curled[finger_idx] = finger_curled
            confidences[finger_idx] = min(y_dist * 3.0, 1.0)  # Scale to 0-1

        return curled, confidences

    def draw_landmarks(
        self,
        frame: np.ndarray,
        state: GestureState,
        confidence: float
    ) -> np.ndarray:
        """
        Draw hand landmarks and state info on the frame.

        Args:
            frame: BGR image to draw on
            state: Current gesture state
            confidence: Confidence value

        Returns:
            Frame with drawings
        """
        output = frame.copy()

        # Draw hand landmarks if available
        if self._last_landmarks is not None:
            self._mp_drawing.draw_landmarks(
                output,
                self._last_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style()
            )

        # Draw state text
        state_color = {
            GestureState.NO_HAND: (128, 128, 128),  # Gray
            GestureState.OPEN: (0, 255, 0),          # Green
            GestureState.CLOSE: (0, 0, 255),         # Red
        }

        text = f"{state.name} ({confidence:.0%})"
        cv2.putText(
            output,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            state_color[state],
            2,
            cv2.LINE_AA
        )

        return output

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            logger.info("HandClassifier closed")

    def __enter__(self) -> 'HandClassifier':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Simple test with webcam
    logging.basicConfig(level=logging.DEBUG)

    from camera import Camera

    print("Testing hand classifier...")
    print("Press 'q' to quit")

    with Camera() as cam, HandClassifier() as classifier:
        while True:
            success, frame = cam.read()
            if not success:
                continue

            state, confidence = classifier.classify(frame)
            frame = classifier.draw_landmarks(frame, state, confidence)

            cv2.imshow("Hand Classifier Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Test complete")
