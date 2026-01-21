# Hand Gesture Control System - Code Outline and Usage Instructions

## Table of Contents

1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Dependencies](#software-dependencies)
5. [Installation](#installation)
6. [Usage Instructions](#usage-instructions)
7. [Code Outline](#code-outline)
8. [Command Protocol](#command-protocol)
9. [Calibration Guide](#calibration-guide)
10. [Troubleshooting](#troubleshooting)

---

## System Overview

This system uses computer vision to detect hand gestures and control a robotic hand. The architecture consists of:

```
+----------------+     USB Serial     +---------------+     I2C      +---------+
|   PC (Python)  | ----------------> | Arduino Nano  | -----------> | PCA9685 |
|   - OpenCV     |    OPEN/CLOSE     |   - Parser    |              | PWM     |
|   - MediaPipe  |                   |   - Motion    |              +---------+
+----------------+                   +---------------+                   |
       ^                                                                 |
       |                                                            PWM signals
   Webcam                                                               |
                                                                        v
                                                              +------------------+
                                                              | 5x MG90S Servos  |
                                                              | (channels 0,4,5,6,7)
                                                              +------------------+
```

**Data Flow:**
1. Webcam captures video frames
2. MediaPipe detects hand landmarks
3. Classifier determines gesture (OPEN/CLOSE/NO_HAND)
4. Debouncer filters noise and confirms state changes
5. Serial link sends commands to Arduino
6. Arduino drives servos via PCA9685

---

## Project Structure

```
motion-hand-controls/
├── pc/                              # Python PC application
│   ├── config.py                    # Configuration constants
│   ├── camera.py                    # Webcam capture module
│   ├── hand_classifier.py           # MediaPipe gesture classification
│   ├── debouncer.py                 # State machine with debouncing
│   ├── serial_link.py               # USB serial communication
│   ├── main.py                      # Application entry point
│   └── requirements.txt             # Python dependencies
├── arduino/                         # Arduino firmware
│   └── hand_control/
│       └── hand_control.ino         # Main Arduino sketch
├── docs/
│   └── CODE_OUTLINE.md              # This file
├── CLAUDE.md                        # Original specification
└── README.md                        # Project readme
```

---

## Hardware Requirements

### Components
- **Arduino Nano** (or compatible board)
- **PCA9685** 16-channel PWM driver
- **5x MG90S** micro servos
- **5V Power Supply** (2A+ recommended for servos)
- **USB cable** for Arduino
- **Webcam** (built-in or USB)

### Wiring

| Arduino Nano | PCA9685      |
|--------------|--------------|
| A4 (SDA)     | SDA          |
| A5 (SCL)     | SCL          |
| 5V           | VCC          |
| GND          | GND          |

| PCA9685      | MG90S Servos           |
|--------------|------------------------|
| Channel 0    | Thumb signal           |
| Channel 4    | Index finger signal    |
| Channel 5    | Middle finger signal   |
| Channel 6    | Ring finger signal     |
| Channel 7    | Pinky finger signal    |
| V+           | Servo power (external) |
| GND          | Servo ground           |

**Important:** Use external 5V power for servos. Do NOT power multiple servos from Arduino 5V pin.

---

## Software Dependencies

### PC Side
- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- PySerial 3.5+
- NumPy 1.24+

### Arduino Side
- Arduino IDE 2.0+
- Adafruit PWM Servo Driver Library

---

## Installation

### PC Setup

1. Navigate to the PC directory:
   ```bash
   cd motion-hand-controls/pc
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Arduino Setup

1. Open Arduino IDE
2. Install the Adafruit PWM Servo Driver library:
   - Go to **Sketch > Include Library > Manage Libraries**
   - Search for "Adafruit PWM Servo Driver"
   - Install the library
3. Open `arduino/hand_control/hand_control.ino`
4. Select your board: **Tools > Board > Arduino Nano**
5. Select the correct port: **Tools > Port > COMx** (Windows) or `/dev/ttyUSBx` (Linux)
6. Upload the sketch

---

## Usage Instructions

### Quick Start

1. Connect Arduino to PC via USB
2. Power on the servo power supply
3. Run the application:
   ```bash
   cd motion-hand-controls/pc
   python main.py --port COM3
   ```
4. Hold your hand in front of the webcam
5. Open hand = robot hand opens
6. Close fist = robot hand closes
7. Press 'q' to quit

### Command Line Options

```
python main.py [OPTIONS]

Options:
  -p, --port PORT    Serial port for Arduino (default: COM3)
  --no-display       Disable video display (headless mode)
  --no-serial        Disable serial output (vision-only mode)
  -d, --debug        Enable debug logging
  --list-ports       List available serial ports and exit
```

### Examples

```bash
# Run with default settings
python main.py

# Specify a different serial port
python main.py --port COM5

# Run in vision-only mode (no Arduino)
python main.py --no-serial

# List available serial ports
python main.py --list-ports

# Run with debug output
python main.py --debug
```

### Display Overlay

When running with display enabled, you'll see:
- **Raw state**: Instantaneous classification result
- **Stable state**: Debounced state (what gets sent to Arduino)
- **Confidence bar**: Visual confidence indicator
- **FPS counter**: Processing speed
- **Serial status**: Connection indicator

### Keyboard Controls
- **q**: Quit the application
- **r**: Reset the debouncer

---

## Code Outline

### PC Modules

#### config.py
Configuration constants for the entire system.

```python
# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Debouncer settings
DEBOUNCE_FRAMES = 5
HYSTERESIS_THRESHOLD = 0.6
NO_HAND_TIMEOUT_FRAMES = 30

# Serial settings
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
```

#### camera.py
Webcam capture with context manager support.

```python
class Camera:
    def __init__(index, width, height, fps, max_retries)
    def start() -> bool
    def read() -> Tuple[bool, np.ndarray]
    def stop() -> None
    def is_opened -> bool
    def frame_size -> Tuple[int, int]
```

#### hand_classifier.py
MediaPipe-based gesture classification.

```python
class GestureState(Enum):
    NO_HAND
    OPEN
    CLOSE

class HandClassifier:
    def __init__(min_detection_confidence, min_tracking_confidence)
    def classify(frame) -> Tuple[GestureState, float]
    def draw_landmarks(frame, state, confidence) -> np.ndarray
    def close() -> None
```

**Finger Detection Logic:**
- Compares fingertip Y-position to MCP joint Y-position
- Thumb uses X-position (accounts for thumb orientation)
- 0-1 curled fingers = OPEN
- 4-5 curled fingers = CLOSE
- 2-3 curled fingers = ambiguous (lower confidence)

#### debouncer.py
State machine with debouncing and hysteresis.

```python
class GestureDebouncer:
    def __init__(debounce_frames, hysteresis_threshold, no_hand_timeout)
    def update(raw_state, confidence) -> GestureState
    def stable_state -> GestureState
    def state_changed -> bool
    def reset() -> None
```

**Algorithm:**
1. If raw matches stable: reset pending, return stable
2. If confidence < threshold: ignore, return stable
3. If raw matches pending: increment counter
4. If counter >= debounce_frames: transition to new state
5. If raw differs from pending: start new pending
6. NO_HAND timeout: failsafe to OPEN if closed too long

#### serial_link.py
USB serial communication to Arduino.

```python
class SerialLink:
    def __init__(port, baud_rate, timeout, auto_reconnect)
    def connect() -> bool
    def disconnect() -> None
    def send_command(command) -> bool
    def send_gesture(state) -> bool
    def read_response(timeout) -> Optional[str]
    def is_connected -> bool
    @staticmethod
    def list_available_ports() -> list
```

#### main.py
Application orchestrator.

```python
def setup_logging(debug) -> None
def parse_args() -> argparse.Namespace
def draw_overlay(frame, raw_state, stable_state, ...) -> None
def run_capture_loop(camera, classifier, debouncer, serial, ...) -> None
def main() -> int
```

### Arduino Sketch

#### hand_control.ino
Main Arduino firmware with the following sections:

```cpp
// Configuration
#define PCA9685_ADDR 0x40
#define PWM_FREQ 50
#define SERVOMIN 150
#define SERVOMAX 600
const uint8_t FINGER_CHANNELS[] = {0, 4, 5, 6, 7};

// Global state
Adafruit_PWMServoDriver pwm;
uint16_t currentPulse[NUM_FINGERS];
uint16_t targetPulse[NUM_FINGERS];
uint16_t fingerOpen[NUM_FINGERS];
uint16_t fingerClose[NUM_FINGERS];

// Functions
void setup()
void loop()
void processCommand(char* cmd)
void parseSetCommand(char* args)
void parseCalCommand(char* args)
void executeOpen()
void executeClose()
void executeWave()
void executeStop()
void updateSmoothMotion()
void moveFinger(uint8_t index, uint16_t pulse)
void printStatus()
void printHelp()
```

---

## Command Protocol

Commands are ASCII text, newline-terminated, case-insensitive.

| Command | Format | Description | Response |
|---------|--------|-------------|----------|
| OPEN | `OPEN\n` | Open all fingers | `ACK:OPEN` |
| CLOSE | `CLOSE\n` | Close all fingers | `ACK:CLOSE` |
| WAVE | `WAVE\n` | Wave animation | `ACK:WAVE` |
| SET | `SET <ch> <pulse>\n` | Set channel pulse | `ACK:SET <ch> <pulse>` |
| CAL | `CAL <f> <open> <close>\n` | Calibrate finger | `ACK:CAL <f> ...` |
| STOP | `STOP\n` | Hold position | `ACK:STOP` |
| STATUS | `STATUS\n` | Print status | Status dump |
| HELP | `HELP\n` | Print help | Help text |

### Error Responses
- `ERR:UNKNOWN_CMD:<cmd>` - Unrecognized command
- `ERR:INVALID_CHANNEL:<ch>` - Invalid servo channel
- `ERR:INVALID_FINGER:<f>` - Invalid finger index
- `ERR:SET_SYNTAX` - SET command format error
- `ERR:CAL_SYNTAX` - CAL command format error
- `ERR:BUFFER_OVERFLOW` - Command too long

---

## Calibration Guide

### Initial Setup

1. Open Arduino Serial Monitor (115200 baud)
2. Send `STATUS` to see current calibration
3. Use `SET` to find optimal pulse values

### Finding Pulse Values

For each finger:
1. Find OPEN position:
   ```
   SET 0 150   # Start low
   SET 0 200   # Increase until finger fully opens
   SET 0 180   # Fine-tune to avoid buzzing
   ```

2. Find CLOSE position:
   ```
   SET 0 600   # Start high
   SET 0 500   # Decrease until finger fully closes
   SET 0 550   # Fine-tune to avoid strain
   ```

3. Save calibration:
   ```
   CAL 0 180 550   # Finger 0 (Thumb): open=180, close=550
   CAL 1 150 600   # Finger 1 (Index): open=150, close=600
   ...
   ```

### Finger Index Mapping

| Index | Finger | Channel |
|-------|--------|---------|
| 0 | Thumb | 0 |
| 1 | Index | 4 |
| 2 | Middle | 5 |
| 3 | Ring | 6 |
| 4 | Pinky | 7 |

### Tips
- Start with conservative values to avoid mechanical stress
- Listen for buzzing - indicates overdriven position
- Each finger may need different calibration
- Calibration is lost on Arduino reset (consider saving to EEPROM)

---

## Troubleshooting

### Camera Issues

**Problem: "Could not open camera"**
- Check if another application is using the camera
- Try a different camera index: edit `CAMERA_INDEX` in config.py
- On Windows, check Camera Privacy settings

**Problem: Low FPS**
- Reduce resolution in config.py
- Ensure good lighting (MediaPipe works faster with clear hands)
- Close other applications

### Serial Issues

**Problem: "Failed to connect to COM3"**
- Run `python main.py --list-ports` to find correct port
- Check Arduino is connected and powered
- On Windows, check Device Manager for COM port
- On Linux, ensure user has dialout group access

**Problem: "No ACK received"**
- Check baud rate matches (115200)
- Open Serial Monitor to verify Arduino is responding
- Check USB cable (some are charge-only)

### Detection Issues

**Problem: Unstable detection / flickering**
- Increase `DEBOUNCE_FRAMES` in config.py (default: 5)
- Ensure good, even lighting
- Avoid busy backgrounds
- Keep hand clearly visible in frame

**Problem: Wrong gesture detected**
- Adjust `MIN_DETECTION_CONFIDENCE` (default: 0.7)
- Try different hand positions
- Ensure fingers are clearly separated for OPEN

### Servo Issues

**Problem: Servos buzzing**
- Reduce pulse values (they may be hitting mechanical limits)
- Use `CAL` command to fine-tune per-finger limits
- Check power supply (should be 5V, 2A+)

**Problem: Servos not moving**
- Verify I2C connection (SDA=A4, SCL=A5)
- Check PCA9685 address (default: 0x40)
- Send `STATUS` command to verify Arduino sees commands
- Check external servo power supply

**Problem: Arduino resets when servos move**
- Power supply insufficient - use external 5V supply
- Ensure common ground between all components

---

## Performance Notes

### Latency
- Camera capture: ~33ms (30 FPS)
- MediaPipe processing: ~20-50ms
- Debounce delay: ~166ms (5 frames at 30 FPS)
- Serial transmission: ~1ms
- **Total gesture-to-motion: ~200-250ms**

### Optimizations
- Smooth motion interpolation reduces servo jitter
- Debouncing prevents command spam
- Rate limiting (only send on state change) reduces serial traffic

---

## Extending the System

### Adding New Gestures
1. Define new state in `GestureState` enum
2. Add detection logic in `HandClassifier._classify_from_curl_count()`
3. Add command mapping in `serial_link.GESTURE_COMMANDS`
4. Add Arduino handler in `processCommand()`

### Adding Per-Finger Control
The system already supports individual finger control via the `SET` command.
To add gesture-based per-finger control:
1. Track individual finger curl states in classifier
2. Send `SET` commands for specific fingers instead of `OPEN`/`CLOSE`

### Saving Calibration to EEPROM
Add to Arduino sketch:
```cpp
#include <EEPROM.h>

void saveCalibration() {
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        EEPROM.put(i * 4, fingerOpen[i]);
        EEPROM.put(i * 4 + 2, fingerClose[i]);
    }
}

void loadCalibration() {
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        EEPROM.get(i * 4, fingerOpen[i]);
        EEPROM.get(i * 4 + 2, fingerClose[i]);
    }
}
```
