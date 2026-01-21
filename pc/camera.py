"""
Camera capture module for webcam video acquisition.
Provides a clean interface for frame capture with context manager support.
"""

import cv2
import logging
from typing import Optional, Tuple
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS
)

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Exception raised when camera operations fail."""
    pass


class Camera:
    """
    Webcam capture wrapper with context manager support.

    Usage:
        with Camera() as cam:
            while True:
                success, frame = cam.read()
                if success:
                    # process frame
    """

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        max_retries: int = 3
    ):
        """
        Initialize camera configuration.

        Args:
            index: Camera device index (0 = default camera)
            width: Desired frame width in pixels
            height: Desired frame height in pixels
            fps: Desired frames per second
            max_retries: Number of connection attempts before failing
        """
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._max_retries = max_retries
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """
        Initialize and open the camera.

        Returns:
            True if camera opened successfully, False otherwise.

        Raises:
            CameraError: If camera fails to open after max retries.
        """
        for attempt in range(self._max_retries):
            logger.debug(f"Camera connection attempt {attempt + 1}/{self._max_retries}")

            self._cap = cv2.VideoCapture(self._index)

            if self._cap.isOpened():
                # Configure camera properties
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)

                # Verify actual settings
                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))

                if actual_width != self._width or actual_height != self._height:
                    logger.warning(
                        f"Requested {self._width}x{self._height} but got "
                        f"{actual_width}x{actual_height}"
                    )
                if actual_fps != self._fps:
                    logger.warning(
                        f"Requested {self._fps} FPS but got {actual_fps} FPS"
                    )

                logger.info(
                    f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS"
                )
                return True

            logger.warning(f"Failed to open camera on attempt {attempt + 1}")

        raise CameraError(
            f"Could not open camera index {self._index} after {self._max_retries} attempts"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success: bool, frame: np.ndarray or None)
        """
        if self._cap is None or not self._cap.isOpened():
            logger.warning("Attempted to read from unopened camera")
            return False, None

        success, frame = self._cap.read()

        if not success:
            logger.debug("Frame capture failed")

        return success, frame

    def stop(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    @property
    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get current frame dimensions (width, height)."""
        if self._cap is None:
            return 0, 0
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def __enter__(self) -> 'Camera':
        """Context manager entry - start camera."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop camera."""
        self.stop()


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)

    print("Testing camera capture...")
    with Camera() as cam:
        print(f"Camera opened: {cam.is_opened}")
        print(f"Frame size: {cam.frame_size}")

        for i in range(10):
            success, frame = cam.read()
            if success:
                print(f"Frame {i}: {frame.shape}")
            else:
                print(f"Frame {i}: FAILED")

    print("Camera test complete")
