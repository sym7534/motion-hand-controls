/**
 * Hand Control Unlock Sketch
 *
 * Disables PWM output on all servo channels so the servos are no longer
 * holding position ("unlocked") and can be moved by hand.
 *
 * Hardware:
 *   - Arduino Nano
 *   - PCA9685 PWM driver (I2C: SDA=A4, SCL=A5, Address=0x40)
 *   - 5x MG90S servos on channels 8, 7, 6, 5, 4
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define PCA9685_ADDR 0x40
#define PWM_FREQ 50

// Finger Channel Mapping (Thumb, Index, Middle, Ring, Pinky)
const uint8_t FINGER_CHANNELS[] = {8, 7, 6, 5, 4};
const uint8_t NUM_FINGERS = 5;

// PCA9685 driver instance
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

// =============================================================================
// SETUP
// =============================================================================

void setup() {
    Wire.begin();
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(PWM_FREQ);

    delay(10);

    // Disable PWM output so servos are released/unpowered
    for (uint8_t i = 0; i < NUM_FINGERS; i++) {
        // 4096 = "full off" bit for PCA9685
        pwm.setPWM(FINGER_CHANNELS[i], 0, 4096);
    }
}

// =============================================================================
// LOOP
// =============================================================================

void loop() {
    // Nothing to do: servos are released.
}
