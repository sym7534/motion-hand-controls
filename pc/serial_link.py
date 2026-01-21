"""
Serial communication module for USB connection to Arduino.
Handles command sending and optional response reading.
"""

import logging
import time
from typing import Optional

import serial
from serial.tools import list_ports

from hand_classifier import GestureState, FingerState
from config import (
    SERIAL_PORT,
    BAUD_RATE,
    SERIAL_TIMEOUT,
    INTER_COMMAND_DELAY
)

logger = logging.getLogger(__name__)


class SerialError(Exception):
    """Exception raised for serial communication errors."""
    pass


# Gesture state to command mapping
GESTURE_COMMANDS = {
    GestureState.OPEN: "OPEN",
    GestureState.CLOSE: "CLOSE",
    # NO_HAND intentionally not mapped - we don't send commands for it
}


class SerialLink:
    """
    USB serial communication manager for Arduino connection.

    Handles connection, command sending, and optional response reading.
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud_rate: int = BAUD_RATE,
        timeout: float = SERIAL_TIMEOUT,
        auto_reconnect: bool = True
    ):
        """
        Initialize serial link configuration.

        Args:
            port: Serial port name (e.g., "COM3" or "/dev/ttyUSB0")
            baud_rate: Communication speed (must match Arduino)
            timeout: Read timeout in seconds
            auto_reconnect: Attempt reconnection on failure
        """
        self._port = port
        self._baud_rate = baud_rate
        self._timeout = timeout
        self._auto_reconnect = auto_reconnect
        self._serial: Optional[serial.Serial] = None
        self._last_send_time: float = 0

    def connect(self) -> bool:
        """
        Open serial connection to Arduino.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud_rate,
                timeout=self._timeout
            )

            # Wait for Arduino to reset (common after serial connection)
            time.sleep(2.0)

            # Clear any startup messages
            self._serial.reset_input_buffer()

            logger.info(f"Serial connected: {self._port} @ {self._baud_rate} baud")
            return True

        except serial.SerialException as e:
            logger.error(f"Failed to connect to {self._port}: {e}")
            self._serial = None
            return False

    def disconnect(self) -> None:
        """Close serial connection."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as e:
                logger.warning(f"Error closing serial: {e}")
            finally:
                self._serial = None
                logger.info("Serial disconnected")

    def send_command(self, command: str) -> bool:
        """
        Send a command string to Arduino.

        Args:
            command: Command to send (newline will be added)

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_connected:
            if self._auto_reconnect:
                logger.info("Attempting auto-reconnect...")
                if not self.connect():
                    return False
            else:
                logger.warning("Not connected, cannot send command")
                return False

        try:
            # Add newline terminator
            full_command = command.strip() + "\n"
            self._serial.write(full_command.encode('ascii'))
            self._serial.flush()
            self._last_send_time = time.time()

            logger.debug(f"Sent: {command}")
            return True

        except serial.SerialException as e:
            logger.error(f"Send failed: {e}")
            self._serial = None  # Mark as disconnected
            return False

    def send_gesture(self, state: GestureState) -> bool:
        """
        Send command for a gesture state.

        Args:
            state: Gesture state to send command for

        Returns:
            True if sent successfully, False if state has no command or send failed.
        """
        command = GESTURE_COMMANDS.get(state)

        if command is None:
            logger.debug(f"No command for state: {state.name}")
            return False

        return self.send_command(command)

    def send_finger_set(self, channel: int, pulse: int) -> bool:
        """
        Send SET command for individual servo control.

        Args:
            channel: PCA9685 channel (0, 4, 5, 6, 7)
            pulse: PWM pulse value (150-600)

        Returns:
            True if sent successfully
        """
        command = f"SET {channel} {pulse}"
        return self.send_command(command)

    def send_finger_state(
        self,
        finger_state: FingerState,
        changed_indices: Optional[list] = None
    ) -> int:
        """
        Send SET commands for finger states.

        Only sends commands for fingers that changed (delta-based transmission).
        Includes small delay between commands to prevent Arduino buffer overflow.

        Args:
            finger_state: FingerState object with current finger states
            changed_indices: Optional list of finger indices to update (None = all 5)

        Returns:
            Number of commands successfully sent
        """
        if changed_indices is None:
            # Send all fingers
            changed_indices = range(5)
        elif len(changed_indices) == 0:
            # No changes - return early
            return 0

        sent_count = 0
        for finger_idx in changed_indices:
            channel, pulse = finger_state.get_channel_pulse(finger_idx)

            if self.send_finger_set(channel, pulse):
                sent_count += 1
            else:
                logger.warning(f"Failed to send SET {channel} {pulse} for finger {finger_idx}")

            # Small delay between commands to avoid buffer overflow
            if finger_idx < changed_indices[-1]:  # Don't delay after last command
                time.sleep(INTER_COMMAND_DELAY)

        logger.debug(f"Sent {sent_count}/{len(changed_indices)} finger commands")
        return sent_count

    def read_response(self, timeout: float = 0.1) -> Optional[str]:
        """
        Read a response line from Arduino.

        Args:
            timeout: Read timeout in seconds

        Returns:
            Response string or None if no response/error.
        """
        if not self.is_connected:
            return None

        try:
            old_timeout = self._serial.timeout
            self._serial.timeout = timeout

            line = self._serial.readline()
            self._serial.timeout = old_timeout

            if line:
                response = line.decode('ascii', errors='ignore').strip()
                logger.debug(f"Received: {response}")
                return response

            return None

        except serial.SerialException as e:
            logger.warning(f"Read failed: {e}")
            return None

    def send_and_wait_ack(
        self,
        command: str,
        timeout: float = 0.5
    ) -> Optional[str]:
        """
        Send command and wait for ACK response.

        Args:
            command: Command to send
            timeout: Time to wait for response

        Returns:
            ACK response string or None if no ACK received.
        """
        if not self.send_command(command):
            return None

        # Wait for ACK
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.read_response(timeout=0.05)
            if response and response.startswith("ACK:"):
                return response
            elif response and response.startswith("ERR:"):
                logger.warning(f"Error response: {response}")
                return response

        logger.warning(f"No ACK received for: {command}")
        return None

    @property
    def is_connected(self) -> bool:
        """Check if serial port is open and ready."""
        return self._serial is not None and self._serial.is_open

    @property
    def port(self) -> str:
        """Get configured port name."""
        return self._port

    @staticmethod
    def list_available_ports() -> list:
        """
        List available serial ports on the system.

        Returns:
            List of (port_name, description) tuples.
        """
        ports = []
        for port in list_ports.comports():
            ports.append((port.device, port.description))
        return ports

    def __enter__(self) -> 'SerialLink':
        """Context manager entry - connect."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disconnect."""
        self.disconnect()


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)

    print("Available serial ports:")
    for port, desc in SerialLink.list_available_ports():
        print(f"  {port}: {desc}")

    print("\nTesting SerialLink...")
    print("(Will fail if no Arduino connected)")

    link = SerialLink()

    if link.connect():
        print("Connected!")

        # Test commands
        for cmd in ["OPEN", "CLOSE", "WAVE", "OPEN"]:
            print(f"\nSending: {cmd}")
            response = link.send_and_wait_ack(cmd, timeout=1.0)
            print(f"Response: {response}")
            time.sleep(1.0)

        link.disconnect()
    else:
        print("Connection failed")

    print("\nTest complete")
