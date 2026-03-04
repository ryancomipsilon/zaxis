import threading
from dataclasses import dataclass

from enum import IntEnum


class GPSFixType(IntEnum):
    NO_GPS = 0
    NO_FIX = 1
    FIX_2D = 2
    FIX_3D = 3
    DGPS = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6
    STATIC = 7
    PPP = 8

class EstimatorType(IntEnum):
    UNKNOWN = 0
    NAIVE = 1
    VISION = 2
    VIO = 3
    GPS = 4
    GPS_INS = 5
    MOCAP = 6
    LIDAR = 7
    AUTOPILOT = 8


# GPS_RAW_INT
@dataclass(frozen=True)
class GPSRawState:
    fc_timestamp_ms: int | None = None   # ms
    rx_timestamp_ms: int | None = None   # ms


    lat: float | None = None          # degrees
    lon: float | None = None          # degrees
    alt_msl: float | None = None      # meters

    vel_n: float | None = None        # m/s
    vel_e: float | None = None        # m/s
    vel_d: float | None = None        # m/s

    cog: float | None = None          # radians
    eph: float | None = None          # dimensionless (HDOP)
    epv: float | None = None          # dimensionless (VDOP)
    fix_type: GPSFixType | None = None


# GLOBAL_POSITION_INT_COV
@dataclass(frozen=True)
class GlobalPositionState:
    fc_timestamp_ms: int | None = None   # ms
    rx_timestamp_ms: int | None = None   # ms

    lat: float | None = None             # degrees
    lon: float | None = None             # degrees
    alt_msl: float | None = None         # meters
    alt_rel: float | None = None         # meters

    vx: float | None = None              # m/s
    vy: float | None = None              # m/s
    vz: float | None = None              # m/s

    cov: tuple[float, ...] | None = None  # 36 elements, row-major, meters^2/(m/s)^2

    estimator_type: EstimatorType | None = None


# LOCAL_POSITION_NED_COV
@dataclass(frozen=True)
class LocalPositionState:
    fc_timestamp_ms: int | None = None   # ms
    rx_timestamp_ms: int | None = None   # ms


    x: float | None = None               # meters
    y: float | None = None               # meters
    z: float | None = None               # meters

    vx: float | None = None              # m/s
    vy: float | None = None              # m/s
    vz: float | None = None              # m/s

    ax: float | None = None              # m/s^2
    ay: float | None = None              # m/s^2
    az: float | None = None              # m/s^2

    cov: tuple[float, ...] | None = None
    estimator_type: EstimatorType | None = None


# ATTITUDE_QUATERNION_COV
@dataclass(frozen=True)
class AttitudeState:
    fc_timestamp_ms: int | None = None           # ms
    rx_timestamp_ms: int | None = None           # ms

    q: tuple[float, float, float, float] | None = None  # unit quaternion
    rollspeed: float | None = None                      # radians/s
    pitchspeed: float | None = None                     # radians/s
    yawspeed: float | None = None                       # radians/s
    cov: tuple[float, float, float] | None = None       # (radians)^2


# ODOMETRY
@dataclass(frozen=True)
class OdometryState:
    fc_timestamp_ms: int | None = None   # ms
    rx_timestamp_ms: int | None = None   # ms
    
    x: float | None = None               # meters
    y: float | None = None               # meters
    z: float | None = None               # meters
    
    q: tuple[float, float, float, float] | None = None  # unit quaternion (w, x, y, z)
    
    vx: float | None = None              # m/s
    vy: float | None = None              # m/s
    vz: float | None = None              # m/s
    
    rollspeed: float | None = None       # radians/s
    pitchspeed: float | None = None      # radians/s
    yawspeed: float | None = None        # radians/s


# BATTERY_STATUS
@dataclass(frozen=True)
class BatteryState:
    rx_timestamp_ms: int | None = None  # ms

    voltage: float | None = None     # volts
    current: float | None = None     # amps
    remaining: int | None = None     # percent


# DISTANCE_SENSOR
@dataclass(frozen=True)
class DistanceSensorState:
    fc_timestamp_ms: int | None = None      # ms 
    rx_timestamp_ms: int | None = None      # ms

    current_distance: float | None = None   # meters
    min_distance: float | None = None       # meters
    max_distance: float | None = None       # meters
    sensor_type: int | None = None          # lidar, sonar, etc.
    sensor_id: int | None = None
    orientation: int | None = None          # MAV_SENSOR_ORIENTATION enum
    covariance: float | None = None         # meters^2


# OPTICAL_FLOW
@dataclass(frozen=True)
class OpticalFlowState:
    fc_timestamp_ms: int | None = None       # ms
    rx_timestamp_ms: int | None = None       # ms

    flow_x: float | None = None              # m/s
    flow_y: float | None = None              # m/s
    quality: int | None = None               # 0-255, optical flow quality
    ground_distance: float | None = None     # meters, negative if unknown
    sensor_id: int | None = None             # optional, sensor identifier

class State:
    def __init__(self):
        self._lock = threading.Lock()
        self.global_position: GlobalPositionState | None = None
        self.gps_raw: GPSRawState | None = None
        self.local_position: LocalPositionState | None = None
        self.attitude: AttitudeState | None = None
        self.odometry: OdometryState | None = None
        self.battery: BatteryState | None = None
        self.distance_sensor: DistanceSensorState | None = None
        self.optical_flow: OpticalFlowState | None = None

    def update_global_position(self, value: GlobalPositionState) -> None:
        with self._lock:
            self.global_position = value

    def update_gps_raw(self, value: GPSRawState) -> None:
        with self._lock:
            self.gps_raw = value

    def update_local_position(self, value: LocalPositionState) -> None:
        with self._lock:
            self.local_position = value

    def update_attitude(self, value: AttitudeState) -> None:
        with self._lock:
            self.attitude = value

    def update_odometry(self, value: OdometryState) -> None:
        with self._lock:
            self.odometry = value

    def update_battery(self, value: BatteryState) -> None:
        with self._lock:
            self.battery = value

    def update_distance_sensor(self, value: DistanceSensorState) -> None:
        with self._lock:
            self.distance_sensor = value

    def update_optical_flow(self, value: OpticalFlowState) -> None:
        with self._lock:
            self.optical_flow = value