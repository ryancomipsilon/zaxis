# ===================== IMPORTS ======================

import threading
import time

from pymavlink import mavutil
from pymavlink.CSVReader import CSVMessage

from zaxis.telemetry.state import (
    AttitudeState,
    BatteryState,
    DistanceSensorState,
    GlobalPositionState,
    GPSFixType,
    GPSRawState,
    LocalPositionState,
    EstimatorType,
    TelemetryState,
)

# ==================== MAVLink fixed-point scaling factors ====================

_LAT_LON_SCALE  = 1e7   # degE7 -> degrees      (multiplier 1e-7)
_ALT_SCALE      = 1e3   # meters -> mm          (multiplier 1e-3)
_CURRENT_SCALE  = 1e2   # 10mA -> amps          (multiplier 1e-2)
_VOLTAGE_SCALE  = 1e3   # mV -> volts           (multiplier 1e-3)
_COG_SCALE      = 1e2   # cdeg -> degrees       (multiplier 1e-2)
_DIST_SCALE     = 1e2   # cm -> meters          (multiplier 1e-2)
_DIST_COV_SCALE = 1e4   # cm^2 -> meters^2      (multiplier 1e-4)

_DEG_TO_RAD     = 3.141592653589793 / 180.0
_S_TO_MS        = 1e3   # s -> ms
_US_TO_MS       = 1e3   # µs -> ms

# =============================================================================

class MavlinkConnection:
    _registry = {}

    def __init__(self, name: str = None):
        self.master = None
        self.connected = threading.Event()
        self.state = TelemetryState()
        self._listeners = []

        if name:
            self._registry[name] = self

    # ==================== API HELPERS ====================

    # TODO: implement lock protection, handle exceptions 

    def connect(self, device: str, baudrate: int) -> None:
        self.master = mavutil.mavlink_connection(
            device,
            baud=baudrate,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,
        )
    
        while True:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True)
            if msg is None:
                continue
            if msg.type not in (
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            ):
                self.master.target_system = msg.get_srcSystem()
                self.master.target_component = msg.get_srcComponent()
                break

        self._request_streams()
        self.connected.set()
        threading.Thread(target=self._rx_loop, daemon=True).start()

    def on(self, msg_type: str, callback) -> None:
        """Register a callback for a specific MAVLink message type."""
        self._listeners.append((msg_type, callback))

    def off(self, msg_type: str, callback) -> None:
        """Unregister a callback for a specific MAVLink message type."""
        self._listeners.remove((msg_type, callback))

    # =====================================================

    def _notify_listeners(self, msg) -> None:
        """Notify all registered listeners of a MAVLink message."""
        msg_type = msg.get_type()
        for listener_type, callback in self._listeners:
            if listener_type == msg_type:
                callback(msg)

    def _request_message_interval(self, message_id: int, interval_us: int) -> None:
        """Request a specific MAVLink message at the given interval (µs). 0 = disable, -1 = default."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,              # confirmation
            message_id,     # param1: message ID
            interval_us,    # param2: interval in µs
            0, 0, 0, 0, 0,  # params 3-7 unused
        )

    def _request_streams(self) -> None:
        hz_50 =  20_000   # µs — 50 Hz  (attitude, local/global position)
        hz_25 =  40_000   # µs — 25 Hz  (distance sensor)
        hz_10 = 100_000   # µs — 10 Hz  (GPS raw — limited by GPS hardware)
        hz_2  = 500_000   # µs —  2 Hz  (battery — slow-changing)

        self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,     hz_50)
        self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,      hz_50)
        self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION,     hz_50)
        self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,             hz_10)
        self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,         hz_25)
        self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,          hz_2)

    def _rx_loop(self) -> None:
        while self.connected.is_set():
            msg: CSVMessage = self.master.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue

            msg_type = msg.get_type()
            rx_ts = int(time.monotonic() * _S_TO_MS)

            # GLOBAL_POSITION_INT_COV — EKF-fused position with covariance.
            if msg_type == "GLOBAL_POSITION_INT_COV":
                self.state.update_global_position(GlobalPositionState(
                    fc_timestamp_ms=msg.time_usec // _US_TO_MS,
                    rx_timestamp_ms=rx_ts,
                    lat=msg.lat / _LAT_LON_SCALE,
                    lon=msg.lon / _LAT_LON_SCALE,
                    alt_msl=msg.alt / _ALT_SCALE,
                    alt_rel=msg.relative_alt / _ALT_SCALE,
                    vx=msg.vx,
                    vy=msg.vy,
                    vz=msg.vz,
                    cov=tuple(msg.covariance),
                    estimator_type=EstimatorType(msg.estimator_type),
                ))

            # GLOBAL_POSITION_INT — standard EKF-fused position (no covariance).
            elif msg_type == "GLOBAL_POSITION_INT":
                self.state.update_global_position(GlobalPositionState(
                    fc_timestamp_ms=msg.time_boot_ms,
                    rx_timestamp_ms=rx_ts,
                    lat=msg.lat / _LAT_LON_SCALE,
                    lon=msg.lon / _LAT_LON_SCALE,
                    alt_msl=msg.alt / _ALT_SCALE,
                    alt_rel=msg.relative_alt / _ALT_SCALE,
                    vx=msg.vx / 100.0,   # cm/s -> m/s
                    vy=msg.vy / 100.0,
                    vz=msg.vz / 100.0,
                    cov=None,
                    estimator_type=None,
                ))

            # GPS_RAW_INT — raw receiver data, stored separately from EKF position.
            elif msg_type == "GPS_RAW_INT":
                self.state.update_gps_raw(GPSRawState(
                    fc_timestamp_ms=msg.time_usec // _US_TO_MS,
                    rx_timestamp_ms=rx_ts,
                    lat=msg.lat / _LAT_LON_SCALE,
                    lon=msg.lon / _LAT_LON_SCALE,
                    alt_msl=msg.alt / _ALT_SCALE,
                    vel_n=None,
                    vel_e=None,
                    vel_d=None,
                    cog=msg.cog / _COG_SCALE * _DEG_TO_RAD,
                    eph=msg.eph,
                    epv=msg.epv,
                    fix_type=GPSFixType(msg.fix_type),
                ))

            # ATTITUDE_QUATERNION_COV
            elif msg_type == "ATTITUDE_QUATERNION_COV":
                self.state.update_attitude(AttitudeState(
                    fc_timestamp_ms=msg.time_usec // _US_TO_MS,
                    rx_timestamp_ms=rx_ts,
                    q=(msg.q[0], msg.q[1], msg.q[2], msg.q[3]),
                    rollspeed=msg.rollspeed,
                    pitchspeed=msg.pitchspeed,
                    yawspeed=msg.yawspeed,
                    cov=(msg.covariance[0], msg.covariance[4], msg.covariance[8]),
                ))

            # ATTITUDE_QUATERNION — standard attitude (no covariance).
            elif msg_type == "ATTITUDE_QUATERNION":
                self.state.update_attitude(AttitudeState(
                    fc_timestamp_ms=msg.time_boot_ms,
                    rx_timestamp_ms=rx_ts,
                    q=(msg.q1, msg.q2, msg.q3, msg.q4),
                    rollspeed=msg.rollspeed,
                    pitchspeed=msg.pitchspeed,
                    yawspeed=msg.yawspeed,
                    cov=None,
                ))

            # LOCAL_POSITION_NED_COV — local position with covariance.
            elif msg_type == "LOCAL_POSITION_NED_COV":
                self.state.update_local_position(LocalPositionState(
                    fc_timestamp_ms=msg.time_usec // _US_TO_MS,
                    rx_timestamp_ms=rx_ts,
                    x=msg.x,
                    y=msg.y,
                    z=msg.z,
                    vx=msg.vx,
                    vy=msg.vy,
                    vz=msg.vz,
                    ax=msg.ax,
                    ay=msg.ay,
                    az=msg.az,
                    cov=tuple(msg.covariance),
                    estimator_type=EstimatorType(msg.estimator_type),
                ))

            # LOCAL_POSITION_NED — standard local position (no covariance).
            elif msg_type == "LOCAL_POSITION_NED":
                self.state.update_local_position(LocalPositionState(
                    fc_timestamp_ms=msg.time_boot_ms,
                    rx_timestamp_ms=rx_ts,
                    x=msg.x,
                    y=msg.y,
                    z=msg.z,
                    vx=msg.vx,
                    vy=msg.vy,
                    vz=msg.vz,
                    ax=None,
                    ay=None,
                    az=None,
                    cov=None,
                    estimator_type=None,
                ))

            # BATTERY_STATUS
            elif msg_type == "BATTERY_STATUS":
                self.state.update_battery(BatteryState(
                    rx_timestamp_ms=rx_ts,
                    voltage=msg.voltages[0] / _VOLTAGE_SCALE if msg.voltages else None,
                    current=msg.current_battery / _CURRENT_SCALE if msg.current_battery != -1 else None,
                    remaining=msg.battery_remaining if msg.battery_remaining != -1 else None,
                ))

            # DISTANCE_SENSOR
            elif msg_type == "DISTANCE_SENSOR":
                self.state.update_distance_sensor(DistanceSensorState(
                    fc_timestamp_ms=msg.time_boot_ms,
                    rx_timestamp_ms=rx_ts,
                    current_distance=msg.current_distance / _DIST_SCALE,
                    min_distance=msg.min_distance / _DIST_SCALE,
                    max_distance=msg.max_distance / _DIST_SCALE,
                    sensor_type=msg.type,
                    sensor_id=msg.id,
                    orientation=msg.orientation,
                    covariance=msg.covariance / _DIST_COV_SCALE if msg.covariance != 255 else None,
                ))

            self._notify_listeners(msg)


    @classmethod
    def get(cls, name: str) -> "MavlinkConnection":
        if name not in cls._registry:
            raise KeyError(f"No connection named '{name}'")
        return cls._registry[name]