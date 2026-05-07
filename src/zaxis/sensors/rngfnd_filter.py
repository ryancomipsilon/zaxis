# ===================== IMPORTS ======================

import time
import threading
from collections import deque
from pymavlink import mavutil

from zaxis.telemetry import MavlinkConnection

# =============================================================================

class RangefinderFilter:
    def __init__(
        self,
        connection: MavlinkConnection,
        sensor,
        max_change: float = 0.30,
        obstacle_height: float = 0.7,
        avg_window: int = 10,
        sensor_id: int = 1,
        sensor_type: int = None,
        orientation: int = None,
        hz: float = 50.0,
    ):
        self.connection = connection
        self.sensor = sensor
        self.max_change = max_change
        self.obstacle_height = obstacle_height
        self.hz = hz

        # Set defaults for sensor configuration
        if sensor_type is None:
            sensor_type = mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER
        if orientation is None:
            orientation = mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270

        self.sensor_cfg = {
            "min_distance": 0.01,
            "max_distance": 8.0,
            "sensor_type": sensor_type,
            "sensor_id": sensor_id,
            "orientation": orientation,
            "covariance": 0.01,
        }

        # Internal state
        self.window: deque = deque(maxlen=avg_window)
        self.over_obstacle: bool = False
        self.entry_raw: float | None = None
        
        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _publish(self, distance_m: float) -> None:
        """Publish distance sensor data via MAVLink."""
        self.connection.master.mav.distance_sensor_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            int(self.sensor_cfg["min_distance"] * 100),
            int(self.sensor_cfg["max_distance"] * 100),
            int(distance_m * 100),
            self.sensor_cfg["sensor_type"],
            self.sensor_cfg["sensor_id"],
            self.sensor_cfg["orientation"],
            int(self.sensor_cfg["covariance"] * 100),
        )

    def process(self, raw_distance: float) -> tuple[float, bool]:
        """
        Process a raw distance sensor reading through the filter.

        Args:
            raw_distance: Raw distance reading in meters

        Returns:
            Tuple of (filtered_distance, over_obstacle)
        """
        if raw_distance is None:
            return None, self.over_obstacle

        if not self.over_obstacle:
            self.window.append(raw_distance)

        # Calculate rolling average
        if len(self.window) > 0:
            avg = sum(self.window) / len(self.window)
        else:
            avg = raw_distance

        # Obstacle detection logic
        if not self.over_obstacle:
            if raw_distance < avg - self.max_change:
                self.over_obstacle = True
                self.entry_raw = raw_distance
        else:
            if raw_distance > self.entry_raw + self.max_change:
                self.over_obstacle = False
                self.entry_raw = None

        # Calculate filtered distance
        filtered_distance = raw_distance + self.obstacle_height if self.over_obstacle else raw_distance

        return filtered_distance, self.over_obstacle

    def publish(self, raw_distance: float) -> None:
        """
        Process and publish a raw distance reading.

        Args:
            raw_distance: Raw distance reading in meters
        """
        filtered_distance, _ = self.process(raw_distance)
        if filtered_distance is not None:
            self._publish(filtered_distance)

    def _loop(self) -> None:
        """Main loop that reads sensor and publishes filtered data."""
        period = 1.0 / self.hz
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            
            # Read from sensor and publish
            raw_distance = self.sensor.read()
            if raw_distance is not None:
                self.publish(raw_distance)
            
            # Maintain frequency
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def start(self) -> None:
        """Start the filter loop in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="rngfnd-filter",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the filter loop."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
