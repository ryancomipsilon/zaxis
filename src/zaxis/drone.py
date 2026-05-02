from zaxis.telemetry import MavlinkConnection
from zaxis.telemetry.state import (
    AttitudeState,
    BatteryState,
    DistanceSensorState,
    GlobalPositionState,
    GPSRawState,
    LocalPositionState,
)

from zaxis.control import FlightControls, FlightMode, Origin
from zaxis.runtime import RuntimeLoop


class Drone(FlightControls):
    def __init__(self, telemetry: MavlinkConnection):
        self.runtime = RuntimeLoop()
        super().__init__(telemetry, self.runtime)

    @property
    def global_position(self) -> GlobalPositionState | None:
        return self.telemetry.global_position

    @property
    def gps_raw(self) -> GPSRawState | None:
        return self.telemetry.gps_raw

    @property
    def local_position(self) -> LocalPositionState | None:
        return self.telemetry.local_position

    @property
    def attitude(self) -> AttitudeState | None:
        return self.telemetry.attitude

    @property
    def battery(self) -> BatteryState | None:
        return self.telemetry.battery

    @property
    def distance_sensor(self) -> DistanceSensorState | None:
        return self.telemetry.distance_sensor

    @property
    def FlightMode(self):
        return FlightMode

    @property
    def Origin(self):
        return Origin
