# motion-hand-controls


TODO:

make closed % be a relative distance, not absolute distance - accounts for distance to camera

limit servo movement ? or track servo movement to get a min/max

remake hand_classifier.py to return distance based on fingertip <-> MCP joint and then rotate servos based on that
- nonlinear? idk like depends on how your fingers bend

define base position for servos -> reset to base position every time it starts

+----------------------+   USB Serial   +------------------+   ESP‑NOW / Wi‑Fi   +------------------+   I2C   +---------+
|   Laptop (Python)    |  ------------> |  ESP32 #1 (PC)    |  ----------------> |  ESP32 #2 (Hand) | ------> | PCA9685 |
|  OpenCV + MediaPipe  |   OPEN/CLOSE   |  Serial→Wireless  |   command packets  |  Wireless→I2C    |        |  PWM    |
+----------------------+                +------------------+                     +------------------+        +---------+
                                                                                                         PWM signals |
                                                                                                                     v
                                                                                                           +------------------+
                                                                                                           | 5x MG90S Servos |
                                                                                                           +------------------+

+----------------------+   USB Serial   +------------------+   ESP‑NOW (2.4GHz)   +------------------+   UART   +---------------+   I2C   +---------+
|   Laptop (Python)    |  ------------> |  ESP32 #1 (PC)    |  ~~~~~~~~~~~~~~~~~> |  ESP32 #2 (Hand) | ------> | Arduino Nano  | ------> | PCA9685 |
|  OpenCV + MediaPipe  |   OPEN/CLOSE   |  Serial→ESP‑NOW   |   command packets   |  ESP‑NOW→UART    |        | Cmd Parser    |        |  PWM    |
+----------------------+                +------------------+                      +------------------+        +---------------+        +---------+
                                                                                                                           PWM signals |
                                                                                                                                      v
                                                                                                                        +------------------+
                                                                                                                        | 5x MG90S Servos |
                                                                                                                        +------------------+

Power:
- Servos + PCA9685: external regulated 5V supply (common ground with Nano/ESP32 #2)
- ESP32s: 5V USB or 3.3V regulated
