# motion-hand-controls

**Setup Instructions**

Flash arduino file to arduino nano -> servo shield (this project uses pca9685)

Run python main.py --port COM# (the port your arduino is connected to) in /pc

Power:
- MG90S Servos + PCA9685: external regulated 5V supply (common ground with Arduinon Nano/ESP32 #2)
- ESP32s: 5V USB or 3.3V regulated
