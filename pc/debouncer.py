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
    FINGER_DEBOUNCE_FRAMES
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
    Debounces individual finger states to prevent servo chatter.

    Each finger is tracked independently with its own debounce counter,
    allowing some fingers to change state while others remain stable.
    """

    def __init__(self, debounce_frames: int = FINGER_DEBOUNCE_FRAMES):
        """
        Initialize the per-finger debouncer.

        Args:
            debounce_frames: Frames required to confirm finger state change
        """
        self._debounce_frames = debounce_frames

        # Per-finger state tracking (5 fingers)
        self._stable_state = [True] * 5  # All fingers start open
        self._pending_state = [None] * 5
        self._pending_count = [0] * 5
        self._state_changed = [False] * 5

        logger.info(f"PerFingerDebouncer initialized: debounce={debounce_frames} frames")

    def update(self, raw_finger_states: list) -> list:
        """
        Update debouncer with raw finger states from classifier.

        Args:
            raw_finger_states: List of 5 bools (True=open, False=closed)

        Returns:
            List of 5 stable (debounced) finger states
        """
        if len(raw_finger_states) != 5:
            logger.warning(f"Invalid finger states length: {len(raw_finger_states)}, expected 5")
            return self._stable_state

        # Process each finger independently
        for i in range(5):
            self._state_changed[i] = False
            raw = raw_finger_states[i]

            # Already stable - reset pending
            if raw == self._stable_state[i]:
                self._pending_state[i] = None
                self._pending_count[i] = 0
                continue

            # Same pending state - increment counter
            if raw == self._pending_state[i]:
                self._pending_count[i] += 1

                logger.debug(
                    f"Finger {i} pending: {self._pending_count[i]}/{self._debounce_frames}"
                )

                # Reached threshold - transition to new stable state
                if self._pending_count[i] >= self._debounce_frames:
                    old_state = "OPEN" if self._stable_state[i] else "CLOSED"
                    new_state = "OPEN" if raw else "CLOSED"

                    self._stable_state[i] = raw
                    self._state_changed[i] = True
                    self._pending_state[i] = None
                    self._pending_count[i] = 0

                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    logger.info(
                        f"Finger {i} ({finger_names[i]}) state change: {old_state} -> {new_state}"
                    )
            else:
                # New pending state - start counter
                self._pending_state[i] = raw
                self._pending_count[i] = 1

        return self._stable_state

    def get_changed_fingers(self) -> list:
        """
        Get indices of fingers that changed in the last update.

        Returns:
            List of finger indices (0-4) that just changed state
        """
        return [i for i in range(5) if self._state_changed[i]]

    @property
    def stable_states(self) -> list:
        """Get current stable states without updating."""
        return self._stable_state

    def reset(self) -> None:
        """Reset debouncer to initial state (all fingers open)."""
        self._stable_state = [True] * 5
        self._pending_state = [None] * 5
        self._pending_count = [0] * 5
        self._state_changed = [False] * 5
        logger.info("PerFingerDebouncer reset")


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
