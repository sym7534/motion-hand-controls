# main app for hand gesture control
# camera capture, gesture classification, debouncing, serial comms

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
    # configure logging format + level
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    # parse cli args
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
    # draw status overlay on frame
    height, width = frame.shape[:2]

    # colors
    colors = {
        GestureState.NO_HAND: (128, 128, 128),
        GestureState.OPEN: (0, 255, 0),
        GestureState.CLOSE: (0, 0, 255),
    }

    # raw state (top-left)
    cv2.putText(
        frame,
        f"Raw: {raw_state.name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        OVERLAY_FONT_SCALE,
        colors[raw_state],
        OVERLAY_THICKNESS
    )

    # stable state (below raw)
    cv2.putText(
        frame,
        f"Stable: {stable_state.name}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        OVERLAY_FONT_SCALE,
        colors[stable_state],
        OVERLAY_THICKNESS
    )

    # confidence bar
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

    # fps (top-right)
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

    # serial status
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

    # instructions
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
    raw_curls: list,
    stable_curls: list,
    confidences: list,
    fps: float,
    serial_connected: bool
) -> None:
    # draw per-finger status overlay on frame
    height, width = frame.shape[:2]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def get_gradient_color(curl: float) -> tuple:
        # gradient color based on curl: 0%=green, 50%=yellow, 100%=red
        if curl <= 0.5:
            # green to yellow
            ratio = curl * 2  # 0.0-0.5 → 0.0-1.0
            b = int(255 * ratio)
            return (0, 255, b)
        else:
            # yellow to red
            ratio = (curl - 0.5) * 2  # 0.5-1.0 → 0.0-1.0
            g = int(255 * (1 - ratio))
            return (0, g, 255)

    # finger status grid
    y_offset = 30
    line_height = 30

    for i in range(5):
        raw_color = get_gradient_color(raw_curls[i])
        stable_color = get_gradient_color(stable_curls[i])

        # finger name
        cv2.putText(
            frame,
            f"{finger_names[i]}:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # raw curl indicator
        cv2.circle(frame, (100, y_offset - 5), 5, raw_color, -1)

        # stable curl indicator
        cv2.circle(frame, (120, y_offset - 5), 7, stable_color, -1)

        # curl percentage
        curl_pct_text = f"{stable_curls[i]*100:.0f}%"
        cv2.putText(
            frame,
            curl_pct_text,
            (140, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            stable_color,
            1
        )

        # pulse value
        pulse = int(150 + stable_curls[i] * 450)
        pulse_text = f"→ {pulse}"
        cv2.putText(
            frame,
            pulse_text,
            (190, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )

        # confidence bar
        bar_x = 250
        bar_width = 50
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
            stable_color,
            -1
        )

        y_offset += line_height

    # legend
    legend_y = y_offset + 10
    cv2.putText(frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(frame, (70, legend_y - 5), 5, (255, 255, 255), 1)
    cv2.putText(frame, "Raw", (80, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(frame, (120, legend_y - 5), 7, (255, 255, 255), 1)
    cv2.putText(frame, "Stable", (132, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # fps
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

    # serial status
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

    # instructions
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
    # main processing loop
    last_sent_state: GestureState = None
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    logger.info("Starting capture loop (press 'q' to quit)")

    while True:
        # capture frame
        success, frame = camera.read()
        if not success:
            logger.warning("Frame capture failed")
            continue

        # calculate fps
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # classify gesture
        raw_state, confidence = classifier.classify(frame)

        # debounce
        stable_state = debouncer.update(raw_state, confidence)

        # send command on state change
        if debouncer.state_changed and stable_state != last_sent_state:
            if stable_state != GestureState.NO_HAND and use_serial:
                if serial.send_gesture(stable_state):
                    logger.info(f"Sent command: {stable_state.name}")
                else:
                    logger.warning(f"Failed to send: {stable_state.name}")
            last_sent_state = stable_state

        # display
        if show_display:
            # draw hand landmarks
            frame = classifier.draw_landmarks(frame, raw_state, confidence)

            # draw overlay
            draw_overlay(
                frame,
                raw_state,
                stable_state,
                confidence,
                fps,
                serial.is_connected if use_serial else False
            )

            cv2.imshow(DISPLAY_WINDOW_NAME, frame)

            # handle keyboard input
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
    # main processing loop for individual finger control
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    logger.info("Starting finger control loop (press 'q' to quit)")

    while True:
        # capture frame
        success, frame = camera.read()
        if not success:
            logger.warning("Frame capture failed")
            continue

        # calculate fps
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # classify individual fingers
        finger_state, confidences = classifier.classify_fingers(frame)

        # debounce per finger
        stable_curls = debouncer.update(finger_state.curl_amounts)
        changed_fingers = debouncer.get_changed_fingers()

        # send commands for changed fingers
        if changed_fingers and use_serial:
            # create FingerState with stable curls
            from hand_classifier import FingerState
            stable_finger_state = FingerState(curl_amounts=stable_curls)

            # send only changed fingers
            sent = serial.send_finger_state(stable_finger_state, changed_fingers)

            if sent > 0:
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                changed_names = [finger_names[i] for i in changed_fingers]
                curls = [f"{stable_curls[i]*100:.0f}%" for i in changed_fingers]
                pulses = [int(150 + stable_curls[i] * 450) for i in changed_fingers]
                logger.info(f"Sent {sent} commands: {', '.join(f'{name}={curl} ({pulse})' for name, curl, pulse in zip(changed_names, curls, pulses))}")

        # display
        if show_display:
            # draw hand landmarks (if available)
            if classifier._last_landmarks is not None:
                # dummy state for drawing
                dummy_state = GestureState.OPEN
                frame = classifier.draw_landmarks(frame, dummy_state, max(confidences) if confidences else 0.0)

                # draw wrist-to-pinky-knuckle distance
                h, w = frame.shape[:2]
                wrist = classifier._last_landmarks.landmark[HandClassifier.WRIST]
                pinky_mcp = classifier._last_landmarks.landmark[HandClassifier.PINKY_MCP]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                px, py = int(pinky_mcp.x * w), int(pinky_mcp.y * h)
                dist = ((px - wx) ** 2 + (py - wy) ** 2) ** 0.5
                cv2.line(frame, (wx, wy), (px, py), (255, 255, 0), 2)
                mx, my = (wx + px) // 2, (wy + py) // 2
                cv2.putText(frame, f"{dist:.0f} px", (mx + 5, my - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # draw per-finger overlay
            draw_finger_overlay(
                frame,
                finger_state.curl_amounts,  # raw curls
                stable_curls,               # stable curls
                confidences,                # confidences
                fps,
                serial.is_connected if use_serial else False
            )

            cv2.imshow(DISPLAY_WINDOW_NAME, frame)

            # handle keyboard input
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
    # main entry point - returns 0=success, 1=camera error, 2=serial error
    args = parse_args()
    setup_logging(args.debug)

    # list ports mode
    if args.list_ports:
        print("Available serial ports:")
        for port, desc in SerialLink.list_available_ports():
            print(f"  {port}: {desc}")
        return 0

    logger.info("Hand Gesture Control System starting...")

    # init components
    camera = Camera()
    classifier = HandClassifier()
    debouncer = PerFingerDebouncer()  # per-finger debouncer for individual finger control
    serial = SerialLink(port=args.port)

    try:
        # start camera
        try:
            camera.start()
        except CameraError as e:
            logger.error(f"Camera error: {e}")
            return 1

        # connect serial (optional)
        use_serial = not args.no_serial
        if use_serial:
            if serial.connect():
                logger.info("Serial connected")
            else:
                logger.warning("Serial connection failed - continuing without serial")
                use_serial = False

        # run main loop
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
        # cleanup
        classifier.close()
        camera.stop()
        serial.disconnect()


if __name__ == "__main__":
    sys.exit(main())
