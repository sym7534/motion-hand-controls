/**
 * Hand Control - Minimal
 *
 * Simple serial-controlled servo driver for PCA9685.
 *
 * Commands:
 *   OPEN              - Set all fingers to open pulse
 *   CLOSE             - Set all fingers to close pulse
 *   SET <ch> <pulse>  - Set specific channel to pulse (150-600)
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define PCA9685_ADDR 0x40
#define PWM_FREQ 50

#define SERVO_MIN 150
#define SERVO_MAX 600

#define SERVO_OPEN_PULSE 150
#define SERVO_CLOSE_PULSE 600

#define SERIAL_BAUD 115200
#define CMD_BUFFER_SIZE 64

// Finger Channel Mapping (Thumb, Index, Middle, Ring, Pinky)
const uint8_t FINGER_CHANNELS[] = {8, 7, 6, 5, 4};
const uint8_t NUM_FINGERS = 5;

// PCA9685 driver instance
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

// Command parsing buffer
char cmdBuffer[CMD_BUFFER_SIZE];
uint8_t cmdIndex = 0;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) {
        delay(10);
    }

    Wire.begin();
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(PWM_FREQ);

    delay(10);

    // Start in OPEN position
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        pwm.setPWM(FINGER_CHANNELS[i], 0, SERVO_OPEN_PULSE);
    }

    Serial.println("READY");
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop() {
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
            cmdIndex = 0;
            Serial.println("ERR:BUFFER_OVERFLOW");
        }
    }
}

// =============================================================================
// COMMANDS
// =============================================================================

void processCommand(char* cmd) {
    // Trim leading whitespace
    while (*cmd == ' ') cmd++;

    // Trim trailing whitespace
    char* end = cmd + strlen(cmd) - 1;
    while (end > cmd && *end == ' ') *end-- = '\0';

    if (strlen(cmd) == 0) return;

    // Uppercase for case-insensitive matching
    for (char* p = cmd; *p; p++) {
        *p = toupper(*p);
    }

    if (strcmp(cmd, "OPEN") == 0) {
        setAll(SERVO_OPEN_PULSE);
        Serial.println("ACK:OPEN");
    } else if (strcmp(cmd, "CLOSE") == 0) {
        setAll(SERVO_CLOSE_PULSE);
        Serial.println("ACK:CLOSE");
    } else if (strncmp(cmd, "SET ", 4) == 0) {
        parseSet(cmd + 4);
    } else {
        Serial.print("ERR:UNKNOWN_CMD:");
        Serial.println(cmd);
    }
}

void parseSet(char* args) {
    int channel, pulse;

    if (sscanf(args, "%d %d", &channel, &pulse) == 2) {
        if (channel >= 0 && channel <= 15) {
            pulse = constrain(pulse, SERVO_MIN, SERVO_MAX);
            pwm.setPWM(channel, 0, pulse);
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

void setAll(uint16_t pulse) {
    pulse = constrain(pulse, SERVO_MIN, SERVO_MAX);
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        pwm.setPWM(FINGER_CHANNELS[i], 0, pulse);
    }
}
