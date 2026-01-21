"""
Gesture debouncing module with state machine and hysteresis.
Prevents servo chatter by requiring consistent state detection before changing.
"""

import logging
from typing import Optional

from hand_classifier import GestureState
from config import (
    DEBOUNCE_FRAMES,
    HYSTERESIS_THRESHOLD,
    NO_HAND_TIMEOUT_FRAMES,
    FINGER_DEBOUNCE_FRAMES,
    CURL_CHANGE_THRESHOLD
)

logger = logging.getLogger(__name__)


class GestureDebouncer:
    """
    State machine that debounces gesture classifications.

    Prevents rapid state changes by requiring:
    1. Consistent state detection for N consecutive frames
    2. Confidence above hysteresis threshold
    3. NO_HAND timeout failsafe to return to OPEN state
    """

    def __init__(
        self,
        debounce_frames: int = DEBOUNCE_FRAMES,
        hysteresis_threshold: float = HYSTERESIS_THRESHOLD,
        no_hand_timeout: int = NO_HAND_TIMEOUT_FRAMES
    ):
        """
        Initialize the debouncer.

        Args:
            debounce_frames: Frames required to confirm state change
            hysteresis_threshold: Minimum confidence to accept state change
            no_hand_timeout: Frames without hand before failsafe to OPEN
        """
        self._debounce_frames = debounce_frames
        self._hysteresis_threshold = hysteresis_threshold
        self._no_hand_timeout = no_hand_timeout

        # Current stable state
        self._stable_state: GestureState = GestureState.NO_HAND

        # Pending state tracking
        self._pending_state: Optional[GestureState] = None
        self._pending_count: int = 0

        # NO_HAND timeout counter
        self._no_hand_count: int = 0

        # State change flag
        self._state_changed: bool = False

        logger.info(
            f"GestureDebouncer initialized: debounce={debounce_frames}, "
            f"hysteresis={hysteresis_threshold}, timeout={no_hand_timeout}"
        )

    def update(
        self,
        raw_state: GestureState,
        confidence: float
    ) -> GestureState:
        """
        Process a raw classification and return the stable state.

        Args:
            raw_state: Raw gesture state from classifier
            confidence: Confidence value (0.0-1.0)

        Returns:
            Current stable (debounced) state
        """
        self._state_changed = False

        # Handle NO_HAND timeout
        if raw_state == GestureState.NO_HAND:
            self._no_hand_count += 1

            # Failsafe: if no hand detected for too long and we're in CLOSE,
            # transition to OPEN for safety
            if (self._stable_state == GestureState.CLOSE and
                    self._no_hand_count >= self._no_hand_timeout):
                logger.info("NO_HAND timeout - failsafe to OPEN")
                self._stable_state = GestureState.OPEN
                self._state_changed = True
                self._reset_pending()
                return self._stable_state
        else:
            self._no_hand_count = 0

        # If raw state matches stable state, reset pending and return
        if raw_state == self._stable_state:
            self._reset_pending()
            return self._stable_state

        # Check if confidence meets threshold
        if confidence < self._hysteresis_threshold:
            # Low confidence - don't update pending state
            logger.debug(
                f"Low confidence {confidence:.2f} < {self._hysteresis_threshold}"
            )
            return self._stable_state

        # Handle pending state transitions
        if raw_state == self._pending_state:
            # Same pending state - increment counter
            self._pending_count += 1
            logger.debug(
                f"Pending {raw_state.name}: {self._pending_count}/{self._debounce_frames}"
            )

            # Check if we've reached debounce threshold
            if self._pending_count >= self._debounce_frames:
                # Transition to new stable state
                old_state = self._stable_state
                self._stable_state = raw_state
                self._state_changed = True
                self._reset_pending()

                logger.info(f"State change: {old_state.name} -> {self._stable_state.name}")

        else:
            # Different pending state - start new pending
            self._pending_state = raw_state
            self._pending_count = 1
            logger.debug(f"New pending state: {raw_state.name}")

        return self._stable_state

    def _reset_pending(self) -> None:
        """Reset pending state tracking."""
        self._pending_state = None
        self._pending_count = 0

    @property
    def stable_state(self) -> GestureState:
        """Get current stable state without updating."""
        return self._stable_state

    @property
    def state_changed(self) -> bool:
        """Check if last update caused a state change."""
        return self._state_changed

    @property
    def pending_info(self) -> str:
        """Get debug info about pending state."""
        if self._pending_state is None:
            return "None"
        return f"{self._pending_state.name} ({self._pending_count}/{self._debounce_frames})"

    def reset(self) -> None:
        """Reset debouncer to initial state."""
        self._stable_state = GestureState.NO_HAND
        self._reset_pending()
        self._no_hand_count = 0
        self._state_changed = False
        logger.info("Debouncer reset")


class PerFingerDebouncer:
    """
    Debounces individual finger curl values to prevent servo chatter.

    Uses threshold-based detection: only triggers commands when curl amount
    changes by more than the configured threshold (e.g., 5%).
    Each finger is tracked independently.
    """

    def __init__(self, change_threshold: float = CURL_CHANGE_THRESHOLD):
        """
        Initialize the per-finger debouncer.

        Args:
            change_threshold: Minimum curl change (0.0-1.0) to trigger update (default 0.05 = 5%)
        """
        from config import BASE_CURL_POSITION

        self._change_threshold = change_threshold

        # Per-finger state tracking (5 fingers)
        # Initialize to base position from config
        self._stable_curl = [BASE_CURL_POSITION[i] for i in range(5)]
        self._state_changed = [False] * 5

        logger.info(f"PerFingerDebouncer initialized: threshold={change_threshold*100:.0f}% curl change")

    def update(self, raw_curl_amounts: list) -> list:
        """
        Update debouncer with raw finger curl amounts from classifier.

        Uses threshold-based detection: only updates stable value when the
        curl amount changes by more than the configured threshold.

        Args:
            raw_curl_amounts: List of 5 floats (0.0-1.0, where 0.0=extended, 1.0=curled)

        Returns:
            List of 5 stable (debounced) curl amounts
        """
        if len(raw_curl_amounts) != 5:
            logger.warning(f"Invalid curl amounts length: {len(raw_curl_amounts)}, expected 5")
            return self._stable_curl

        # Process each finger independently
        for i in range(5):
            self._state_changed[i] = False
            raw_curl = raw_curl_amounts[i]

            # Clamp to valid range
            raw_curl = max(0.0, min(1.0, raw_curl))

            # Calculate change from stable value
            curl_delta = abs(raw_curl - self._stable_curl[i])

            # Check if change exceeds threshold
            if curl_delta > self._change_threshold:
                # Significant change detected - update stable value
                old_curl = self._stable_curl[i]
                self._stable_curl[i] = raw_curl
                self._state_changed[i] = True

                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                logger.info(
                    f"Finger {i} ({finger_names[i]}) curl change: "
                    f"{old_curl*100:.0f}% -> {raw_curl*100:.0f}% "
                    f"(Δ={curl_delta*100:.0f}%)"
                )

        return self._stable_curl

    def get_changed_fingers(self) -> list:
        """
        Get indices of fingers that changed in the last update.

        Returns:
            List of finger indices (0-4) that just changed state
        """
        return [i for i in range(5) if self._state_changed[i]]

    @property
    def stable_curls(self) -> list:
        """Get current stable curl amounts without updating."""
        return self._stable_curl

    def reset(self) -> None:
        """Reset debouncer to base position."""
        from config import BASE_CURL_POSITION

        self._stable_curl = [BASE_CURL_POSITION[i] for i in range(5)]
        self._state_changed = [False] * 5
        logger.info("PerFingerDebouncer reset to base position")


if __name__ == "__main__":
    # Simple unit test
    logging.basicConfig(level=logging.DEBUG)

    print("Testing GestureDebouncer...")

    debouncer = GestureDebouncer(debounce_frames=3)

    # Test sequence
    test_inputs = [
        (GestureState.NO_HAND, 0.0),
        (GestureState.OPEN, 0.9),
        (GestureState.OPEN, 0.85),
        (GestureState.OPEN, 0.9),   # Should trigger change to OPEN
        (GestureState.CLOSE, 0.8),
        (GestureState.CLOSE, 0.85),
        (GestureState.OPEN, 0.9),   # Interrupt - reset pending
        (GestureState.CLOSE, 0.9),
        (GestureState.CLOSE, 0.9),
        (GestureState.CLOSE, 0.9),  # Should trigger change to CLOSE
    ]

    for i, (state, conf) in enumerate(test_inputs):
        result = debouncer.update(state, conf)
        print(
            f"Input {i}: {state.name:8} conf={conf:.1f} -> "
            f"Stable: {result.name:8} Changed: {debouncer.state_changed}"
        )

    print("\nTest complete")
