/**
 * Hand Control Arduino Sketch
 *
 * Receives serial commands from PC and controls MG90S servos via PCA9685.
 *
 * Hardware:
 *   - Arduino Nano
 *   - PCA9685 PWM driver (I2C: SDA=A4, SCL=A5, Address=0x40)
 *   - 5x MG90S servos on channels 0, 4, 5, 6, 7
 *   - External 5V power supply for servos
 *
 * Commands:
 *   OPEN              - All fingers to open position
 *   CLOSE             - All fingers to closed position
 *   WAVE              - Sequential finger wave animation
 *   SET <ch> <pulse>  - Set specific channel to pulse value
 *   CAL <f> <o> <c>   - Calibrate finger (index, open pulse, close pulse)
 *   STOP              - Hold current position
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

// PCA9685 Settings
#define PCA9685_ADDR 0x40
#define PWM_FREQ 50  // 50 Hz for servos

// Servo Pulse Limits (12-bit PWM range, typical values 150-600)
#define SERVOMIN 150   // Minimum pulse length (full open)
#define SERVOMAX 600   // Maximum pulse length (full close)
#define SERVO_CENTER ((SERVOMIN + SERVOMAX) / 2)

// Finger Channel Mapping
const uint8_t FINGER_CHANNELS[] = {0, 4, 5, 6, 7};
const uint8_t NUM_FINGERS = 5;

// Finger names for debugging
const char* FINGER_NAMES[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};

// Per-Finger Calibration (adjustable via CAL command)
uint16_t fingerOpen[NUM_FINGERS]  = {150, 150, 150, 150, 150};
uint16_t fingerClose[NUM_FINGERS] = {600, 600, 600, 600, 600};

// Serial Settings
#define SERIAL_BAUD 115200
#define CMD_BUFFER_SIZE 64

// Smoothing Settings
#define SMOOTH_ENABLED true
#define SMOOTH_STEP 15       // Pulse increment per update cycle
#define SMOOTH_DELAY_MS 20   // Milliseconds between smooth updates

// Safety Settings
#define COMMAND_TIMEOUT_MS 0  // Set >0 to enable failsafe (0 = disabled)

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

// PCA9685 driver instance
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

// Command parsing buffer
char cmdBuffer[CMD_BUFFER_SIZE];
uint8_t cmdIndex = 0;

// Current and target servo positions (for smoothing)
uint16_t currentPulse[NUM_FINGERS];
uint16_t targetPulse[NUM_FINGERS];

// Timing
unsigned long lastCommandTime = 0;
unsigned long lastSmoothUpdate = 0;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
    // Initialize serial
    Serial.begin(SERIAL_BAUD);
    while (!Serial) {
        delay(10);  // Wait for USB serial (Leonardo/Micro)
    }

    // Initialize I2C and PCA9685
    Wire.begin();
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);  // Internal oscillator calibration
    pwm.setPWMFreq(PWM_FREQ);

    delay(10);  // Give PCA9685 time to settle

    // Initialize servo positions to OPEN
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        currentPulse[i] = fingerOpen[i];
        targetPulse[i] = fingerOpen[i];
        pwm.setPWM(FINGER_CHANNELS[i], 0, currentPulse[i]);
    }

    lastCommandTime = millis();

    Serial.println("READY");
    Serial.println("Hand Control v1.0");
    Serial.print("Fingers: ");
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        Serial.print(FINGER_CHANNELS[i]);
        if (i < NUM_FINGERS - 1) Serial.print(", ");
    }
    Serial.println();
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop() {
    // Read and process serial commands
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (cmdIndex > 0) {
                cmdBuffer[cmdIndex] = '\0';
                processCommand(cmdBuffer);
                cmdIndex = 0;
            }
        } else if (cmdIndex < CMD_BUFFER_SIZE - 1) {
            cmdBuffer[cmdIndex++] = c;
        } else {
            // Buffer overflow - reset
            cmdIndex = 0;
            Serial.println("ERR:BUFFER_OVERFLOW");
        }
    }

    // Update smooth motion
    if (SMOOTH_ENABLED) {
        updateSmoothMotion();
    }

    // Safety timeout (optional failsafe)
    #if COMMAND_TIMEOUT_MS > 0
    if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
        executeOpen();
        lastCommandTime = millis();
        Serial.println("WARN:TIMEOUT_FAILSAFE");
    }
    #endif
}

// =============================================================================
// COMMAND PARSER
// =============================================================================

void processCommand(char* cmd) {
    // Trim leading whitespace
    while (*cmd == ' ') cmd++;

    // Trim trailing whitespace
    char* end = cmd + strlen(cmd) - 1;
    while (end > cmd && *end == ' ') *end-- = '\0';

    // Skip empty commands
    if (strlen(cmd) == 0) return;

    // Convert to uppercase for case-insensitive matching
    for (char* p = cmd; *p; p++) {
        *p = toupper(*p);
    }

    lastCommandTime = millis();

    // Parse command
    if (strcmp(cmd, "OPEN") == 0) {
        executeOpen();
        Serial.println("ACK:OPEN");
    }
    else if (strcmp(cmd, "CLOSE") == 0) {
        executeClose();
        Serial.println("ACK:CLOSE");
    }
    else if (strcmp(cmd, "WAVE") == 0) {
        executeWave();
        Serial.println("ACK:WAVE");
    }
    else if (strncmp(cmd, "SET ", 4) == 0) {
        parseSetCommand(cmd + 4);
    }
    else if (strncmp(cmd, "CAL ", 4) == 0) {
        parseCalCommand(cmd + 4);
    }
    else if (strcmp(cmd, "STOP") == 0) {
        executeStop();
        Serial.println("ACK:STOP");
    }
    else if (strcmp(cmd, "STATUS") == 0) {
        printStatus();
    }
    else if (strcmp(cmd, "HELP") == 0) {
        printHelp();
    }
    else {
        Serial.print("ERR:UNKNOWN_CMD:");
        Serial.println(cmd);
    }
}

// =============================================================================
// SET COMMAND PARSER
// =============================================================================

void parseSetCommand(char* args) {
    // Format: SET <channel> <pulse>
    int channel, pulse;

    if (sscanf(args, "%d %d", &channel, &pulse) == 2) {
        // Validate channel
        int fingerIndex = getFingerIndex(channel);

        if (fingerIndex >= 0) {
            // Clamp pulse to safe range
            pulse = constrain(pulse, SERVOMIN, SERVOMAX);

            // Set target (smooth) or apply directly
            targetPulse[fingerIndex] = pulse;
            if (!SMOOTH_ENABLED) {
                currentPulse[fingerIndex] = pulse;
                pwm.setPWM(channel, 0, pulse);
            }

            Serial.print("ACK:SET ");
            Serial.print(channel);
            Serial.print(" ");
            Serial.println(pulse);
        } else {
            Serial.print("ERR:INVALID_CHANNEL:");
            Serial.println(channel);
        }
    } else {
        Serial.println("ERR:SET_SYNTAX (SET <channel> <pulse>)");
    }
}

// =============================================================================
// CAL COMMAND PARSER
// =============================================================================

void parseCalCommand(char* args) {
    // Format: CAL <finger_index> <openPulse> <closePulse>
    int finger, openPulse, closePulse;

    if (sscanf(args, "%d %d %d", &finger, &openPulse, &closePulse) == 3) {
        if (finger >= 0 && finger < NUM_FINGERS) {
            // Clamp to safe ranges
            fingerOpen[finger] = constrain(openPulse, SERVOMIN, SERVOMAX);
            fingerClose[finger] = constrain(closePulse, SERVOMIN, SERVOMAX);

            Serial.print("ACK:CAL ");
            Serial.print(finger);
            Serial.print(" (");
            Serial.print(FINGER_NAMES[finger]);
            Serial.print(") open=");
            Serial.print(fingerOpen[finger]);
            Serial.print(" close=");
            Serial.println(fingerClose[finger]);
        } else {
            Serial.print("ERR:INVALID_FINGER:");
            Serial.print(finger);
            Serial.print(" (valid: 0-");
            Serial.print(NUM_FINGERS - 1);
            Serial.println(")");
        }
    } else {
        Serial.println("ERR:CAL_SYNTAX (CAL <finger> <open> <close>)");
    }
}

// =============================================================================
// MOTION PRIMITIVES
// =============================================================================

void executeOpen() {
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        targetPulse[i] = fingerOpen[i];
    }
    if (!SMOOTH_ENABLED) {
        applyAllTargets();
    }
}

void executeClose() {
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        targetPulse[i] = fingerClose[i];
    }
    if (!SMOOTH_ENABLED) {
        applyAllTargets();
    }
}

void executeStop() {
    // Hold current positions (stop any smooth motion)
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        targetPulse[i] = currentPulse[i];
    }
}

void applyAllTargets() {
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        currentPulse[i] = targetPulse[i];
        pwm.setPWM(FINGER_CHANNELS[i], 0, currentPulse[i]);
    }
}

// =============================================================================
// WAVE SEQUENCE
// =============================================================================

void executeWave() {
    const uint16_t waveDelay = 120;  // Milliseconds between fingers

    // Wave close (thumb to pinky)
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        moveFinger(i, fingerClose[i]);
        delay(waveDelay);
    }

    // Brief hold
    delay(100);

    // Wave open (pinky to thumb)
    for (int8_t i = NUM_FINGERS - 1; i >= 0; i--) {
        moveFinger(i, fingerOpen[i]);
        delay(waveDelay);
    }
}

void moveFinger(uint8_t fingerIndex, uint16_t pulse) {
    if (fingerIndex >= NUM_FINGERS) return;

    pulse = constrain(pulse, SERVOMIN, SERVOMAX);
    currentPulse[fingerIndex] = pulse;
    targetPulse[fingerIndex] = pulse;
    pwm.setPWM(FINGER_CHANNELS[fingerIndex], 0, pulse);
}

// =============================================================================
// SMOOTH MOTION
// =============================================================================

void updateSmoothMotion() {
    unsigned long now = millis();

    if (now - lastSmoothUpdate < SMOOTH_DELAY_MS) {
        return;  // Not time to update yet
    }
    lastSmoothUpdate = now;

    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        if (currentPulse[i] != targetPulse[i]) {
            if (currentPulse[i] < targetPulse[i]) {
                currentPulse[i] = min((uint16_t)(currentPulse[i] + SMOOTH_STEP),
                                      targetPulse[i]);
            } else {
                currentPulse[i] = max((uint16_t)(currentPulse[i] - SMOOTH_STEP),
                                      targetPulse[i]);
            }
            pwm.setPWM(FINGER_CHANNELS[i], 0, currentPulse[i]);
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

int getFingerIndex(uint8_t channel) {
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        if (FINGER_CHANNELS[i] == channel) {
            return i;
        }
    }
    return -1;  // Not found
}

void printStatus() {
    Serial.println("=== STATUS ===");
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        Serial.print(FINGER_NAMES[i]);
        Serial.print(" (ch ");
        Serial.print(FINGER_CHANNELS[i]);
        Serial.print("): current=");
        Serial.print(currentPulse[i]);
        Serial.print(" target=");
        Serial.print(targetPulse[i]);
        Serial.print(" range=[");
        Serial.print(fingerOpen[i]);
        Serial.print("-");
        Serial.print(fingerClose[i]);
        Serial.println("]");
    }
    Serial.println("==============");
}

void printHelp() {
    Serial.println("=== COMMANDS ===");
    Serial.println("OPEN           - Open all fingers");
    Serial.println("CLOSE          - Close all fingers");
    Serial.println("WAVE           - Wave animation");
    Serial.println("SET <ch> <pls> - Set channel pulse (150-600)");
    Serial.println("CAL <f> <o> <c>- Calibrate finger");
    Serial.println("STOP           - Hold current position");
    Serial.println("STATUS         - Show current state");
    Serial.println("HELP           - Show this help");
    Serial.println("================");
}
