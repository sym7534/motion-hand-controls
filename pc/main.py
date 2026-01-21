"""
Main application for hand gesture control system.
Orchestrates camera capture, gesture classification, debouncing, and serial communication.
"""

import argparse
import logging
import sys
import time
import cv2

from camera import Camera, CameraError
from hand_classifier import HandClassifier, GestureState, FingerState
from debouncer import GestureDebouncer, PerFingerDebouncer
from serial_link import SerialLink
from config import (
    SERIAL_PORT,
    SHOW_DISPLAY,
    DISPLAY_WINDOW_NAME,
    OVERLAY_FONT_SCALE,
    OVERLAY_THICKNESS
)

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Configure logging format and level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Hand Gesture Control System'
    )
    parser.add_argument(
        '--port', '-p',
        default=SERIAL_PORT,
        help=f'Serial port for Arduino (default: {SERIAL_PORT})'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display'
    )
    parser.add_argument(
        '--no-serial',
        action='store_true',
        help='Disable serial output (vision only mode)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--list-ports',
        action='store_true',
        help='List available serial ports and exit'
    )

    return parser.parse_args()


def draw_overlay(
    frame,
    raw_state: GestureState,
    stable_state: GestureState,
    confidence: float,
    fps: float,
    serial_connected: bool
) -> None:
    """Draw status overlay on frame."""
    height, width = frame.shape[:2]

    # Colors
    colors = {
        GestureState.NO_HAND: (128, 128, 128),
        GestureState.OPEN: (0, 255, 0),
        GestureState.CLOSE: (0, 0, 255),
    }

    # Raw state (top-left)
    cv2.putText(
        frame,
        f"Raw: {raw_state.name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        OVERLAY_FONT_SCALE,
        colors[raw_state],
        OVERLAY_THICKNESS
    )

    # Stable state (top-left, below raw)
    cv2.putText(
        frame,
        f"Stable: {stable_state.name}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        OVERLAY_FONT_SCALE,
        colors[stable_state],
        OVERLAY_THICKNESS
    )

    # Confidence bar (top-left, below stable)
    bar_x = 10
    bar_y = 80
    bar_width = 200
    bar_height = 20
    fill_width = int(bar_width * confidence)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                  colors[raw_state], -1)
    cv2.putText(
        frame,
        f"{confidence:.0%}",
        (bar_x + bar_width + 10, bar_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # FPS (top-right)
    fps_text = f"FPS: {fps:.1f}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(
        frame,
        fps_text,
        (width - text_size[0] - 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Serial status (top-right, below FPS)
    serial_text = "Serial: OK" if serial_connected else "Serial: OFF"
    serial_color = (0, 255, 0) if serial_connected else (0, 0, 255)
    cv2.putText(
        frame,
        serial_text,
        (width - 100, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        serial_color,
        1
    )

    # Instructions (bottom)
    cv2.putText(
        frame,
        "Press 'q' to quit | 'r' to reset",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )


def draw_finger_overlay(
    frame,
    raw_states: list,
    stable_states: list,
    confidences: list,
    fps: float,
    serial_connected: bool
) -> None:
    """
    Draw per-finger status overlay on frame.

    Args:
        frame: Video frame to draw on
        raw_states: List of 5 raw finger states (True=open, False=closed)
        stable_states: List of 5 stable finger states (True=open, False=closed)
        confidences: List of 5 confidence values (0.0-1.0)
        fps: Current frames per second
        serial_connected: Whether serial is connected
    """
    height, width = frame.shape[:2]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    # Colors: Green for open, Red for closed
    open_color = (0, 255, 0)
    closed_color = (0, 0, 255)

    # Draw finger status grid (left side)
    y_offset = 30
    line_height = 25

    for i in range(5):
        raw_color = open_color if raw_states[i] else closed_color
        stable_color = open_color if stable_states[i] else closed_color

        # Finger name
        cv2.putText(
            frame,
            f"{finger_names[i]}:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Raw state indicator (small circle)
        cv2.circle(frame, (100, y_offset - 5), 5, raw_color, -1)

        # Stable state indicator (larger circle)
        cv2.circle(frame, (120, y_offset - 5), 7, stable_color, -1)

        # Confidence bar
        bar_x = 140
        bar_width = 80
        bar_height = 10
        fill = int(bar_width * confidences[i])

        cv2.rectangle(
            frame,
            (bar_x, y_offset - 10),
            (bar_x + bar_width, y_offset),
            (100, 100, 100),
            1
        )
        cv2.rectangle(
            frame,
            (bar_x, y_offset - 10),
            (bar_x + fill, y_offset),
            raw_color,
            -1
        )

        # Confidence percentage
        cv2.putText(
            frame,
            f"{confidences[i]:.0%}",
            (bar_x + bar_width + 5, y_offset - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        y_offset += line_height

    # Legend (below finger status)
    legend_y = y_offset + 10
    cv2.putText(frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(frame, (70, legend_y - 5), 5, (255, 255, 255), 1)
    cv2.putText(frame, "Raw", (80, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(frame, (120, legend_y - 5), 7, (255, 255, 255), 1)
    cv2.putText(frame, "Stable", (132, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # FPS (top-right)
    fps_text = f"FPS: {fps:.1f}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(
        frame,
        fps_text,
        (width - text_size[0] - 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Serial status (top-right, below FPS)
    serial_text = "Serial: OK" if serial_connected else "Serial: OFF"
    serial_color = (0, 255, 0) if serial_connected else (0, 0, 255)
    cv2.putText(
        frame,
        serial_text,
        (width - 100, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        serial_color,
        1
    )

    # Instructions (bottom)
    cv2.putText(
        frame,
        "Press 'q' to quit | 'r' to reset",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )


def run_capture_loop(
    camera: Camera,
    classifier: HandClassifier,
    debouncer: GestureDebouncer,
    serial: SerialLink,
    show_display: bool = True,
    use_serial: bool = True
) -> None:
    """
    Main processing loop.

    Args:
        camera: Camera instance
        classifier: HandClassifier instance
        debouncer: GestureDebouncer instance
        serial: SerialLink instance
        show_display: Whether to show video display
        use_serial: Whether to send serial commands
    """
    last_sent_state: GestureState = None
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    logger.info("Starting capture loop (press 'q' to quit)")

    while True:
        # Capture frame
        success, frame = camera.read()
        if not success:
            logger.warning("Frame capture failed")
            continue

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # Classify gesture
        raw_state, confidence = classifier.classify(frame)

        # Debounce
        stable_state = debouncer.update(raw_state, confidence)

        # Send command on state change
        if debouncer.state_changed and stable_state != last_sent_state:
            if stable_state != GestureState.NO_HAND and use_serial:
                if serial.send_gesture(stable_state):
                    logger.info(f"Sent command: {stable_state.name}")
                else:
                    logger.warning(f"Failed to send: {stable_state.name}")
            last_sent_state = stable_state

        # Display
        if show_display:
            # Draw hand landmarks
            frame = classifier.draw_landmarks(frame, raw_state, confidence)

            # Draw overlay
            draw_overlay(
                frame,
                raw_state,
                stable_state,
                confidence,
                fps,
                serial.is_connected if use_serial else False
            )

            cv2.imshow(DISPLAY_WINDOW_NAME, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord('r'):
                debouncer.reset()
                last_sent_state = None
                logger.info("Debouncer reset")

    if show_display:
        cv2.destroyAllWindows()


def run_finger_control_loop(
    camera: Camera,
    classifier: HandClassifier,
    debouncer: PerFingerDebouncer,
    serial: SerialLink,
    show_display: bool = True,
    use_serial: bool = True
) -> None:
    """
    Main processing loop for individual finger control.

    Args:
        camera: Camera instance
        classifier: HandClassifier instance
        debouncer: PerFingerDebouncer instance
        serial: SerialLink instance
        show_display: Whether to show video display
        use_serial: Whether to send serial commands
    """
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    logger.info("Starting finger control loop (press 'q' to quit)")

    while True:
        # Capture frame
        success, frame = camera.read()
        if not success:
            logger.warning("Frame capture failed")
            continue

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # Classify individual fingers
        finger_state, confidences = classifier.classify_fingers(frame)

        # Debounce per finger
        stable_states = debouncer.update(finger_state.fingers)
        changed_fingers = debouncer.get_changed_fingers()

        # Send commands only for changed fingers
        if changed_fingers and use_serial:
            # Update FingerState with stable states
            finger_state.fingers = stable_states

            # Send only changed fingers
            sent = serial.send_finger_state(finger_state, changed_fingers)

            if sent > 0:
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                changed_names = [finger_names[i] for i in changed_fingers]
                states = ["OPEN" if stable_states[i] else "CLOSED" for i in changed_fingers]
                logger.info(f"Sent {sent} commands: {', '.join(f'{name}={state}' for name, state in zip(changed_names, states))}")

        # Display
        if show_display:
            # Draw hand landmarks first (if available)
            if classifier._last_landmarks is not None:
                # Create a dummy state for drawing landmarks
                dummy_state = GestureState.OPEN
                frame = classifier.draw_landmarks(frame, dummy_state, max(confidences) if confidences else 0.0)

            # Draw per-finger overlay
            draw_finger_overlay(
                frame,
                finger_state.fingers,  # Raw states
                stable_states,         # Stable states
                confidences,           # Confidences
                fps,
                serial.is_connected if use_serial else False
            )

            cv2.imshow(DISPLAY_WINDOW_NAME, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord('r'):
                debouncer.reset()
                logger.info("PerFingerDebouncer reset")

    if show_display:
        cv2.destroyAllWindows()


def main() -> int:
    """
    Main application entry point.

    Returns:
        Exit code (0=success, 1=camera error, 2=serial error)
    """
    args = parse_args()
    setup_logging(args.debug)

    # List ports mode
    if args.list_ports:
        print("Available serial ports:")
        for port, desc in SerialLink.list_available_ports():
            print(f"  {port}: {desc}")
        return 0

    logger.info("Hand Gesture Control System starting...")

    # Initialize components
    camera = Camera()
    classifier = HandClassifier()
    debouncer = PerFingerDebouncer()  # Use per-finger debouncer for individual finger control
    serial = SerialLink(port=args.port)

    try:
        # Start camera
        try:
            camera.start()
        except CameraError as e:
            logger.error(f"Camera error: {e}")
            return 1

        # Connect serial (optional)
        use_serial = not args.no_serial
        if use_serial:
            if serial.connect():
                logger.info("Serial connected")
            else:
                logger.warning("Serial connection failed - continuing without serial")
                use_serial = False

        # Run main loop with individual finger control
        run_finger_control_loop(
            camera=camera,
            classifier=classifier,
            debouncer=debouncer,
            serial=serial,
            show_display=not args.no_display and SHOW_DISPLAY,
            use_serial=use_serial
        )

        logger.info("Shutting down...")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    finally:
        # Cleanup
        classifier.close()
        camera.stop()
        serial.disconnect()


if __name__ == "__main__":
    sys.exit(main())
