import threading
import time

from pymavlink import mavutil

from zaxis.telemetry.state import (
    AttitudeState,
    BatteryState,
    DistanceSensorState,
    GlobalPositionState,
    GPSFixType,
    GPSRawState,
    LocalPositionState,
    EstimatorType,
    OdometryState,
    OpticalFlowState,
    State,
)

# MAVLink fixed-point scaling factors (per protocol spec)
_LAT_LON_SCALE  = 1e7   # degE7 -> degrees     (multiplier 1e-7)
_ALT_SCALE      = 1e3   # mm -> meters          (multiplier 1e-3)
_CURRENT_SCALE  = 1e2   # 10mA -> amps          (multiplier 1e-2)
_VOLTAGE_SCALE  = 1e3   # mV -> volts           (multiplier 1e-3)
_COG_SCALE      = 1e2   # cdeg -> degrees       (multiplier 1e-2)
_DIST_SCALE     = 1e2   # cm -> meters          (multiplier 1e-2)
_DIST_COV_SCALE = 1e4   # cm^2 -> meters^2      (multiplier 1e-4)

_DEG_TO_RAD     = 3.141592653589793 / 180.0
_S_TO_MS        = 1e3   # s -> ms
_US_TO_MS       = 1e3   # µs -> ms


class Mavlink:
    def __init__(self, state: State):
        self.master = None
        self.connected = threading.Event()
        self.state = state

    def connect(self, device: str, baudrate: int) -> None:
        self.master = mavutil.mavlink_connection(
            device,
            baud=baudrate,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,
        )
        self.master.wait_heartbeat()
        self.connected.set()

        # Start reception thread
        threading.Thread(target=self._rx_loop, daemon=True).start()

    def _rx_loop(self) -> None:
        while self.connected.is_set():
            msg = self.master.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue

            msg_type = msg.get_type()
            rx_ts = int(time.monotonic() * _S_TO_MS)

            # GLOBAL_POSITION_INT_COV — EKF-fused position, preferred over GPS_RAW_INT.
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

            # LOCAL_POSITION_NED_COV
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

            # ODOMETRY
            elif msg_type == "ODOMETRY":
                self.state.update_odometry(OdometryState(
                    fc_timestamp_ms=msg.time_usec // _US_TO_MS,
                    rx_timestamp_ms=rx_ts,
                    x=msg.x,
                    y=msg.y,
                    z=msg.z,
                    q=(msg.q[0], msg.q[1], msg.q[2], msg.q[3]),
                    vx=msg.vx,
                    vy=msg.vy,
                    vz=msg.vz,
                    rollspeed=msg.rollspeed,
                    pitchspeed=msg.pitchspeed,
                    yawspeed=msg.yawspeed,
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

            # OPTICAL_FLOW
            elif msg_type == "OPTICAL_FLOW":
                self.state.update_optical_flow(OpticalFlowState(
                    fc_timestamp_ms=msg.time_usec // _US_TO_MS,
                    rx_timestamp_ms=rx_ts,
                    flow_x=msg.flow_comp_m_x,
                    flow_y=msg.flow_comp_m_y,
                    quality=msg.quality,
                    ground_distance=msg.ground_distance,
                    sensor_id=msg.sensor_id,
                ))