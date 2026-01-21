## OpenCV → Arduino Nano → PCA9685 → MG90S Robot Hand

---

## 0) What runs where

### PC / Laptop (Python)

* Captures video (webcam)
* Runs hand detection + classification (open vs closed)
* Applies smoothing/debouncing to prevent flicker
* Sends **high-level commands** over USB serial to Arduino

### Arduino Nano (C++)

* Receives serial commands (text or bytes)
* Validates/parses commands
* Translates commands into target servo positions
* Sends PWM commands over I2C to PCA9685

### PCA9685 (hardware PWM driver)

* Generates stable 50 Hz servo pulses per channel
* Drives MG90S signal pins on channels **0, 4, 5, 6, 7**

---

## 1) System data flow

1. **Camera frame** acquired on PC
2. Frame processed → **hand landmarks/features** extracted
3. Classifier decides:

   * `OPEN` (hand open)
   * `CLOSE` (fist)
4. Decision passed through **state machine** (debounce + hysteresis)
5. When state changes, PC sends command over **USB serial**
6. Arduino reads command → updates target positions
7. Arduino commands PCA9685 via **I2C** (`Wire`)
8. PCA9685 outputs PWM to servos → fingers move

---

## 2) Hardware interface outline

### USB Serial (PC ↔ Arduino)

* Single-direction commands (PC → Arduino) is enough
* Optional Arduino → PC feedback (“ACK”, telemetry)

### I2C (Arduino ↔ PCA9685)

* Arduino Nano uses:

  * SDA = A4
  * SCL = A5
* PCA9685 address typically `0x40` (depends on jumpers)

### Servo power (critical)

* Servos powered from external 5V supply (recommended)
* **Common ground**: servo PSU GND ↔ PCA9685 GND ↔ Arduino GND
* Avoid powering multiple MG90S from Nano 5V (resets/jitter)

---

## 3) Command protocol design

### Minimal command set (recommended)

* `OPEN` → all fingers to open position
* `CLOSE` → all fingers to closed position

### Optional extensions (nice to have)

* `WAVE` → sequential finger motion (your test sequence)
* `SET <channel> <pulse>` → direct per-servo control for calibration
* `CAL <finger> <openPulse> <closePulse>` → per-finger tuning
* `STOP` → hold current position / disable outputs (if you implement)

### Formatting choices

* Human-readable ASCII lines (easy debugging)
* Newline terminated for easy parsing
* Make commands **idempotent** (sending `OPEN` twice is safe)

---

## 4) PC-side vision pipeline outline (OpenCV)

### 4.1 Capture stage

* Choose camera index
* Configure resolution and FPS
* Preprocess frame:

  * optional resize for speed
  * optional color conversion (BGR→RGB if using a model)

### 4.2 Hand detection / tracking stage

Two common approaches:

* **Landmark-based** (easiest robust option)

  * Detect hand + keypoints each frame
  * Gives fingertip positions and joint angles
* **Contour/skin-threshold based**

  * Faster + simpler, but less reliable in varied lighting

### 4.3 Feature extraction

Examples of “hand closed” indicators:

* Distances between fingertips and palm center / MCP joints
* Finger curl estimates (angle proxies from landmarks)
* Hand silhouette convexity defects (if contour-based)

### 4.4 Classification logic

* Convert raw features → one of:

  * `OPEN`
  * `CLOSE`
  * (optional) `UNKNOWN` if confidence low

### 4.5 Stabilization / debouncing (important)

Goal: prevent servo chatter from noisy detection.

* Maintain a short history buffer (e.g., last N frames)
* Only change state if:

  * new state persists for N frames, OR
  * confidence exceeds threshold consistently
* Add hysteresis:

  * harder to switch states than to stay in the same state
* Rate-limit commands:

  * only send on **state change**
  * enforce a minimum interval between sends

### 4.6 Serial send stage

* Open serial port
* On state change:

  * send command line
* Optional: listen for Arduino responses (`ACK`, errors)

---

## 5) PC-side control state machine outline

### States

* `NO_HAND` (no detection)
* `HAND_OPEN`
* `HAND_CLOSED`

### Transitions

* `NO_HAND → HAND_OPEN` when confident open detected for N frames
* `NO_HAND → HAND_CLOSED` when confident closed detected for N frames
* `HAND_OPEN ↔ HAND_CLOSED` only after debounce/hysteresis checks
* Optional “failsafe”:

  * if no hand for T seconds → command `OPEN` (safer pose)

### Output actions

* Enter `HAND_OPEN`: send `OPEN`
* Enter `HAND_CLOSED`: send `CLOSE`
* Otherwise: send nothing

---

## 6) Arduino-side program outline

### 6.1 Initialization (setup)

* Start Serial (baud rate matches PC)
* Init I2C (`Wire`)
* Init PCA9685 driver:

  * set oscillator frequency (if you use that calibration)
  * set PWM frequency = 50 Hz
* Define channel mapping:

  * `fingerChannels = {0,4,5,6,7}`
* Define safe pulse bounds:

  * `SERVOMIN`, `SERVOMAX`
  * (optional) per-finger min/max arrays
* Move hand to a safe default pose (usually `OPEN`)

### 6.2 Main loop responsibilities

* Read incoming serial bytes
* Assemble into complete command lines
* Parse command
* Validate (known command? valid range?)
* Execute motion primitive

### 6.3 Command parsing layer

* Robust to:

  * `\r\n` vs `\n`
  * extra spaces
  * unknown commands (ignore or reply error)
* Optional: send back `ACK:<cmd>` to PC for debugging

### 6.4 Motion primitives

* `moveAllFingers(pulse)`
* `moveFinger(channel, pulse)`
* `waveSequence()`

### 6.5 Motion strategy

Two options:

* **Instant step** (simple)

  * Set target pulse immediately
* **Smooth ramp** (recommended)

  * Interpolate from current pulse → target pulse over time
  * Reduces jerk + buzzing + mechanical stress

### 6.6 Safety handling

* Clamp pulses to safe range
* Optional per-finger clamp (because finger mechanics differ)
* Optional “timeout”:

  * if no command received for T seconds → open hand / hold
* Optional “disable outputs” mode if you add it

---

## 7) Servo mapping + calibration outline

### 7.1 Pulse representation

* PCA9685 typically uses a 12-bit counter (0–4095)
* You’re already using the “pulse length” style constants:

  * `SERVOMIN` for open
  * `SERVOMAX` for closed

### 7.2 Calibration workflow

1. Start with conservative min/max to prevent binding
2. Test each finger individually:

   * find minimum that fully opens without buzzing
   * find maximum that fully closes without strain
3. Store per-finger values:

   * `openPulse[finger]`
   * `closePulse[finger]`
4. Update `moveAllFingers()` to use per-finger values if needed

### 7.3 Mechanical considerations

* MG90S can buzz if:

  * commanded beyond mechanical limit
  * load is too high
  * power supply sags
* Smooth ramping + good power supply typically helps most.

---

## 8) Debugging + validation plan

### Stage 1: Hardware-only (Arduino)

* Confirm PCA9685 detected
* Move each channel individually
* Verify correct finger mapping (0,4,5,6,7)
* Confirm no resets when moving all fingers (power issue check)

### Stage 2: Serial-only

* Send `OPEN`/`CLOSE` from Serial Monitor
* Verify Arduino parsing is stable and consistent

### Stage 3: Vision-only (PC)

* Display classification on-screen (“OPEN/CLOSE/NO_HAND”)
* Confirm stable state changes under normal lighting

### Stage 4: Full integration

* Send only on state changes
* Verify no command spam
* Add ramping if motion is harsh

---

## 9) Recommended “minimal viable” structure

### PC program modules

* `camera.py` (capture + frame loop)
* `hand_classifier.py` (features → OPEN/CLOSE/NO_HAND)
* `debouncer.py` (state stability logic)
* `serial_link.py` (send commands, optional read ACK)
* `main.py` (ties it together)

### Arduino sketch sections

* Config constants (channels, min/max)
* Init: Serial + PCA9685
* Serial read buffer + parser
* Motion functions (move all, move finger, wave)
* Optional smoothing + safety clamp

