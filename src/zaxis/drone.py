from zaxis.telemetry import MavlinkConnection
from zaxis.telemetry.state import (
    AttitudeState,
    BatteryState,
    DistanceSensorState,
    GlobalPositionState,
    GPSRawState,
    LocalPositionState,
    OdometryState,
    OpticalFlowState,
)


class Drone:
    def __init__(self, telemetry: MavlinkConnection):
        self.telemetry = telemetry

    @property
    def global_position(self) -> GlobalPositionState | None:
        return self.telemetry.state.global_position

    @property
    def gps_raw(self) -> GPSRawState | None:
        return self.telemetry.state.gps_raw

    @property
    def local_position(self) -> LocalPositionState | None:
        return self.telemetry.state.local_position

    @property
    def attitude(self) -> AttitudeState | None:
        return self.telemetry.state.attitude

    @property
    def odometry(self) -> OdometryState | None:
        return self.telemetry.state.odometry

    @property
    def battery(self) -> BatteryState | None:
        return self.telemetry.state.battery

    @property
    def distance_sensor(self) -> DistanceSensorState | None:
        return self.telemetry.state.distance_sensor

    @property
    def optical_flow(self) -> OpticalFlowState | None:
        return self.telemetry.state.optical_flow