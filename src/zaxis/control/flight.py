# ===================== IMPORTS ======================

from pymavlink import mavutil
from pymavlink.CSVReader import CSVMessage

from zaxis.telemetry import MavlinkConnection, TelemetryState
from zaxis.runtime import RuntimeCommand, RuntimeLoop, CommandHandle

import math
from typing import Callable, cast, Tuple
from enum import Enum
import time
import threading

from zaxis.telemetry.state import DistanceSensorState

from collections import deque
import numpy as np

# =================== MAVLink fixed-point scaling factors ====================

from zaxis.telemetry.mavlink import _LAT_LON_SCALE, _ALT_SCALE
_METERS_PER_DEG_LAT = 111320.0               # fairly constant globally
_METERS_PER_DEG_LON_AT_EQUATOR = 111320.0    # shrinks with cos(lat) towards poles

# ==================== MAVLink Position Target Type Masks ====================

POSITION_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

POSITION_VELOCITY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

VELOCITY_ONLY_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


# =============================================================================

class Origin:
    def __init__(self, x: float, y: float, z: float, yaw: float):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw

class GlobalOrigin:
    def __init__(self, lat: float, lon: float, alt: float, yaw: float):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.yaw = yaw

class FlightMode(str, Enum):
    STABILIZE = "STABILIZE"
    GUIDED = "GUIDED"
    LAND = "LAND"
    RTL = "RTL"
    LOITER = "LOITER"
    AUTO = "AUTO"

class FlightControls:
    def __init__(self, connection: MavlinkConnection, runtime: RuntimeLoop):
        self.connection = connection
        self.telemetry = connection.state
        self.runtime = runtime
        self.primary_origin: Origin | None = None


    # TODO: MAV_CMD_RUN_PREARM_CHECKS

    # ==================== API HELPERS ====================
    
    def set_mode(self, mode: FlightMode) -> CommandHandle:
        master = self.connection.master
        mode_mapping = master.mode_mapping()
        mode_id = mode_mapping[mode.value]

        command = RuntimeCommand(
            name="flight_set_mode",
            callback=lambda: master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id, 0, 0, 0, 0, 0),
            hz=1
        )

        def on_heartbeat(msg: CSVMessage) -> None:
            if command.handle.done():
                self.connection.off("HEARTBEAT", on_heartbeat)
                return
            
            custom_mode_enabled = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED

            if custom_mode_enabled and msg.custom_mode == mode_id:
                self.runtime.unregister("flight_set_mode")
                self.connection.off("HEARTBEAT", on_heartbeat)

        if "flight_set_mode" in self.runtime.registered_commands:
            self.runtime.unregister("flight_set_mode", success=False)
        self.runtime.register(command)

        try:
            self.connection.off("HEARTBEAT", on_heartbeat)
        except Exception:
            pass
        finally:
            self.connection.on("HEARTBEAT", on_heartbeat)

        return command.handle

    def arm(self) -> CommandHandle:
        """Arm the drone."""
        master = self.connection.master

        if "flight_disarm" in self.runtime.registered_commands:
            self.runtime.unregister("flight_disarm", success=False)

        command = RuntimeCommand(
            name="flight_arm",
            callback=lambda: master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                1,  # 1 = Arm, 0 = Disarm
                0, 0, 0, 0, 0, 0
            ),
            hz=1
        )

        def on_heartbeat(msg: CSVMessage):
            if command.handle.done(): # Command finished unexpectedly
                self.connection.off("HEARTBEAT", on_heartbeat)
                return

            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

            if armed:
                self.runtime.unregister("flight_arm")
                self.connection.off("HEARTBEAT", on_heartbeat)

        try:
            self.connection.off("HEARTBEAT", on_heartbeat)
        except Exception:
            pass
        finally:
            self.connection.on("HEARTBEAT", on_heartbeat)

        if "flight_arm" in self.runtime.registered_commands:
            self.runtime.unregister("flight_arm", success=False)
        self.runtime.register(command)

        return command.handle

    def disarm(self) -> CommandHandle:
        """Disarm the drone."""
        master = self.connection.master

        if "flight_arm" in self.runtime.registered_commands:
            self.runtime.unregister("flight_arm", success=False)

        command = RuntimeCommand(
            name="flight_disarm",
            callback=lambda: master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                0,  # 1 = Arm, 0 = Disarm
                0, 0, 0, 0, 0, 0
            ),
            hz=1
        )

        def on_heartbeat(msg: CSVMessage):
            if command.handle.done(): # Command finished unexpectedly
                self.connection.off("HEARTBEAT", on_heartbeat)
                return

            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

            if not armed:
                self.runtime.unregister("flight_disarm")
                self.connection.off("HEARTBEAT", on_heartbeat)

        try:
            self.connection.off("HEARTBEAT", on_heartbeat)
        except Exception:
            pass
        finally:
            self.connection.on("HEARTBEAT", on_heartbeat)

        if "flight_arm" in self.runtime.registered_commands:
            self.runtime.unregister("flight_disarm", success=False)
        self.runtime.register(command)

        return command.handle

    def takeoff(self, altitude: float, tolerance: float = 0.1) -> CommandHandle:
        master = self.connection.master

        target_z: float | None = None

        def send_takeoff() -> None:
            nonlocal target_z

            local = self.telemetry.local_position
            if local is None or local.z is None:
                return

            if self.primary_origin is None:
                self.primary_origin = self.capture_origin()
                self.set_home()

            if target_z is None:
                target_z = local.z - altitude

            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0,
                0, 0,
                altitude
            )

        command = RuntimeCommand(
            name="flight_takeoff",
            callback=send_takeoff,
            hz=1,
            on_stop=None
        )

        def on_ack(msg: CSVMessage) -> None:
            if command.handle.done():
                self.connection.off("COMMAND_ACK", on_ack)
                return

            if msg.command != mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                return
            
            if msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return

            if target_z is None:
                return
            
            self.connection.off("COMMAND_ACK", on_ack)

            def monitor_altitude() -> None:
                while not command.handle.done() and abs(self.telemetry.local_position.z - target_z) > tolerance:
                    print(f"z: {round(self.telemetry.local_position.z - target_z, 2)}")
                    time.sleep(1 / 5)
                self.runtime.unregister("flight_takeoff")

            threading.Thread(target=monitor_altitude, daemon=True).start()

        try:
            self.connection.off("COMMAND_ACK", on_ack)
        except Exception:
            pass
        finally:
            self.connection.on("COMMAND_ACK", on_ack)

        self.runtime.register(command)
        return command.handle

    def land(self) -> CommandHandle:
        """Command the drone to land. Not stoppable once started."""
        self.set_mode(FlightMode.LAND)
        handle = CommandHandle()

        def on_heartbeat(msg: CSVMessage) -> None:
            if handle.done():
                self.connection.off("HEARTBEAT", on_heartbeat)
                return
            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if not armed:
                self.connection.off("HEARTBEAT", on_heartbeat)
                handle._finish()

        try:
            self.connection.off("HEARTBEAT", on_heartbeat)
        except Exception:
            pass
        finally:
            self.connection.on("HEARTBEAT", on_heartbeat)

        return handle
    def rngfnd_filter(
            self,
            sensor,
            max_change: float = 0.30,
            obstacle_height: float = 0.7,
            avg_window: int = 10,
            hz: float = 20.0,
        ) -> CommandHandle:
            master = self.connection.master
            sensor_cfg = {
                "min_distance": 0.01,
                "max_distance": 8.0,
                "sensor_type": mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
                "sensor_id": 1,
                "orientation": mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,
                "covariance": 0.01,
            }

            window: deque = deque(maxlen=avg_window)
            over_obstacle: list = [False]
            entry_raw: list = [None]

            def _publish(distance_m: float, raw: float) -> None:
                master.mav.distance_sensor_send(
                    int(time.time() * 1000) & 0xFFFFFFFF,
                    int(sensor_cfg["min_distance"] * 100),
                    int(sensor_cfg["max_distance"] * 100),
                    int(distance_m * 100),
                    sensor_cfg["sensor_type"],
                    sensor_cfg["sensor_id"],
                    sensor_cfg["orientation"],
                    int(sensor_cfg["covariance"] * 100)
                )

            def tick() -> None:
                raw = sensor.read()
                if raw is None:
                    return

                if not over_obstacle[0]:
                    window.append(raw)

                avg = sum(window) / len(window)

                if not over_obstacle[0]:
                    if raw < avg - max_change:
                        over_obstacle[0] = True
                        entry_raw[0] = raw
                else:
                    if raw > entry_raw[0] + max_change:
                        over_obstacle[0] = False
                        entry_raw[0] = None

                _publish(raw + obstacle_height if over_obstacle[0] else raw, raw)

            command = RuntimeCommand(
                name="rngfnd_filter",
                callback=tick,
                hz=hz,
            )
            if "rngfnd_filter" in self.runtime.registered_commands:
                self.runtime.unregister("rngfnd_filter", success=False)
            self.runtime.register(command)
            return command.handle

        
    def capture_origin(self, global_origin: bool = False) -> Origin | GlobalOrigin:
        attitude = self.telemetry.attitude
        w, xq, yq, zq = attitude.q
        yaw = math.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))

        if global_origin:
            gpos = self.telemetry.global_position
            if gpos is None:
                raise RuntimeError("No global position available")
            return GlobalOrigin(
                lat=gpos.lat,
                lon=gpos.lon,
                alt=gpos.alt_rel,
                yaw=yaw
            )
        
        local = self.telemetry.local_position
        return Origin(
            x=local.x,
            y=local.y,
            z=local.z,
            yaw=yaw
        )

    def set_velocity_ned(self, vx: float, vy: float, vz: float) -> None:
        master = self.connection.master
        master.mav.set_position_target_local_ned_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_ONLY_MASK,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )
 
    def set_velocity_body(self, vx: float, vy: float, vz: float) -> None:
        master = self.connection.master
        master.mav.set_position_target_local_ned_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            VELOCITY_ONLY_MASK,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    def goto_local(
        self,
        x: float,
        y: float,
        z: float,
        origin: Origin | None = None,
        face_wp: bool = False,
        radius: float = 0.05
        
    ) -> CommandHandle:
        """Go to a local position along axes FRD defined by the given origin."""

        if origin is None:
            if self.primary_origin is None:
                raise RuntimeError("No origin defined")
            origin = self.primary_origin

        x, y, z = self._resolve_ned(x, y, z, origin)

        command = RuntimeCommand(
            name="flight_goto",
            callback=lambda: self._send_position_target_local_ned(
                x, y, z,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                lambda: self._set_param("WP_YAW_BEHAVIOR", int(face_wp))
            ),
            hz=20,
            on_stop=self._send_position_hold
        )

        def monitor_position() -> None:
            while not command.handle.done():
                local = self.telemetry.local_position
                if local is not None:
                    dist = math.sqrt((local.x - x)**2 + (local.y - y)**2 + (local.z - z)**2)
                    print(f"x: {round(local.x - x, 3)}, y: {round(local.y - y, 3)}, z: {round(local.z - z, 3)}")
                    if dist <= radius:
                        break
                time.sleep(1 / 10)
            self.runtime.unregister("flight_goto")
            

        threading.Thread(target=monitor_position, daemon=True).start()
        self.runtime.register(command)

        return command.handle


    def goto_offset(
            self,
            x: float,
            y: float,
            z: float,
            ned: bool = False,
            face_wp: bool = False,
            radius: float = 0.05
        ):
        local = self.telemetry.local_position
        if local is None:
            raise RuntimeError("No local position available")

        if ned:
            target_x = local.x + x
            target_y = local.y + y
            target_z = local.z + z
        else:
            origin = self.capture_origin()
            target_x, target_y, target_z = self._resolve_ned(x, y, z, origin)
            target_z = z + origin.z

        command = RuntimeCommand(
            name="flight_goto",
            callback=lambda: self._send_position_target_local_ned(
                target_x, target_y, target_z, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                lambda: self._set_param("WP_YAW_BEHAVIOR", int(face_wp))
            ),
            hz=20,
            on_stop=self._send_position_hold
        )

        def monitor_position() -> None:
            while not command.handle.done():
                local = self.telemetry.local_position
                if local is not None:
                    dist = math.sqrt((local.x - target_x)**2 + (local.y - target_y)**2 + (local.z - target_z)**2)
                    if dist <= 0.05:
                        break
                time.sleep(1 / 10)
            self.runtime.unregister("flight_goto")
        

        threading.Thread(target=monitor_position, daemon=True).start()
        self.runtime.register(command)

        return command.handle

    def goto_global(
        self,
        x: float,
        y: float,
        z: float,
        origin: GlobalOrigin | None = None,
        face_wp: bool = False,
        radius: float = 0.05
    ) -> CommandHandle:
        """Go to a global position. If an origin is provided, x/y/z are FRD meter
        offsets relative to that origin; otherwise x/y/z are lat/lon/alt directly."""

        if origin is not None:
            cos_y = math.cos(origin.yaw)
            sin_y = math.sin(origin.yaw)

            fwd = x * cos_y - y * sin_y
            rgt = x * sin_y + y * cos_y

            target_lat = origin.lat + fwd / _METERS_PER_DEG_LAT
            target_lon = origin.lon + rgt / (
                _METERS_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(target_lat))
            )
            target_alt = origin.alt - z
        else:
            target_lat = x
            target_lon = y
            target_alt = -z

        command = RuntimeCommand(
            name="flight_goto",
            callback=lambda: self._send_position_target_global_int(
                int(target_lat * _LAT_LON_SCALE), int(target_lon * _LAT_LON_SCALE), target_alt,
                lambda: self._set_param("WP_YAW_BEHAVIOR", int(face_wp))
            ),
            hz=20,
            on_stop=self._send_position_hold
        )

        def monitor_position() -> None:
            while not command.handle.done():
                gpos = self.telemetry.global_position
                if gpos is not None:
                    dlat = (gpos.lat - target_lat) * _METERS_PER_DEG_LAT
                    dlon = (gpos.lon - target_lon) * _METERS_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(target_lat))
                    dalt = gpos.alt_rel - target_alt
                    dist = math.sqrt(dlat**2 + dlon**2 + dalt**2)
                    if dist <= radius:
                        break
                time.sleep(1 / 10)
            self.runtime.unregister("flight_goto")

        threading.Thread(target=monitor_position, daemon=True).start()
        self.runtime.register(command)

        return command.handle


    def rtl(self) -> CommandHandle:
        self._set_param("WP_YAW_BEHAVIOR", 2)
        self.set_mode(FlightMode.RTL)
        handle = CommandHandle() 

        def on_heartbeat(msg: CSVMessage) -> None:
            if handle.done():
                self.connection.off("HEARTBEAT", on_heartbeat)
                return
            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if not armed:
                self.connection.off("HEARTBEAT", on_heartbeat)
                handle._finish()  

        self.connection.on("HEARTBEAT", on_heartbeat)
        return handle

    def set_home(self) -> None:
        """Set the current position as the home position."""
        master = self.connection.master
        
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME,
            0,
            1,  # Use current location
            0, 0, 0, 0, 0, 0
        )

    def set_ekf_origin(self, lat: float = 0.0, lon: float = 0.0, alt: float = 0.0) -> None:
        """Set EKF origin."""
        master = self.connection.master

        master.mav.set_gps_global_origin_send(
            master.target_system,
            int(lat * _LAT_LON_SCALE),
            int(lon * _LAT_LON_SCALE),
            int(alt * _ALT_SCALE)
        )

    def reanchor(
        self,
        detection_center_px: Tuple[float, float],
        camera_center_px: Tuple[float, float],
        target_frd: Tuple[float, float, float],
        focal_length: Tuple[float, float],
        camera_offset_frd: Tuple[float, float] = (0.0, 0.0), 
    ) -> Tuple[float, float]:
        """
        Reanchors the drone origin based on a detected object in image space.

        Args:
            detection_center_px: (px, py) pixel coordinates of the detection center
            camera_center_px: (cx, cy) pixel coordinates of the image center
            target_frd: (forward, right, down) target position relative to origin (meters)
            focal_length: (fx, fy) focal length in pixels

        Returns:
            (delta_n, delta_e): correction applied to origin in NED frame (meters)
        """

        local = self.telemetry.local_position
        if local is None:
            raise RuntimeError("No local position available")

        if self.primary_origin is None:
            raise RuntimeError("No primary origin defined.")
        
        target_n, target_e, target_z = self._resolve_ned(
            target_frd[0], target_frd[1], target_frd[2], self.primary_origin
        )

        depth = abs(local.z - target_z)

        fx, fy = focal_length
        px, py = detection_center_px
        cx, cy = camera_center_px

        x = (px - cx) * (depth / fx)
        y = (py - cy) * (depth / fy)

        body_x = -y
        body_y = x

        attitude = self.telemetry.attitude
        w, xq, yq, zq = attitude.q
        yaw = math.atan2(
            2.0 * (w * zq + xq * yq),
            1.0 - 2.0 * (yq * yq + zq * zq)
        )

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        cam_fwd, cam_right = camera_offset_frd
        cam_offset_n = cos_y * cam_fwd - sin_y * cam_right
        cam_offset_e = sin_y * cam_fwd + cos_y * cam_right

        offset_n = cos_y * body_x - sin_y * body_y
        offset_e = sin_y * body_x + cos_y * body_y

        actual_n = target_n - offset_n - cam_offset_n
        actual_e = target_e - offset_e - cam_offset_e

        delta_n = actual_n - local.x
        delta_e = actual_e - local.y

        # Update origin
        self.primary_origin = Origin(
            x=self.primary_origin.x + delta_n,
            y=self.primary_origin.y + delta_e,
            z=self.primary_origin.z,
            yaw=self.primary_origin.yaw
        )

        return delta_n, delta_e

    # =====================================================

    def _set_param(self, param_name, value) -> CommandHandle:
        master = self.connection.master
        if isinstance(value, int):
            param_type = mavutil.mavlink.MAV_PARAM_TYPE_INT32
        else:
            param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32

        command = RuntimeCommand(
            name=f"set_param_{param_name}",
            callback=lambda: master.mav.param_set_send(
                master.target_system,
                master.target_component,
                param_name.encode('utf-8'),
            value,
            param_type
            ),
            hz=1/5
        )


        def on_param_set(msg: CSVMessage):
            param_id = cast(bytes, msg.param_id).rstrip("\x00")

            if param_id == param_name and math.isclose(msg.param_value, value, abs_tol=1e-6):
                self.connection.off("PARAM_VALUE", on_param_set)
                self.runtime.unregister(f"set_param_{param_name}")
        
        try:
            self.connection.off("PARAM_VALUE", on_param_set)
        except Exception:
            pass
        finally:
            self.connection.on("PARAM_VALUE", on_param_set)

        self.runtime.register(command)
        return command.handle

    def _resolve_ned(self, x, y, z, origin: Origin):
        cos_y = math.cos(origin.yaw)
        sin_y = math.sin(origin.yaw)

        real_x = origin.x + x * cos_y - y * sin_y
        real_y = origin.y + x * sin_y + y * cos_y
        real_z = z  

        return real_x, real_y, real_z

    def _send_position_target_local_ned(
        self,
        x: float,
        y: float,
        z: float,
        frame,
        waiter: Callable[[], CommandHandle]
    ) -> None:
        """Send a single SET_POSITION_TARGET_LOCAL_NED with position."""
        master = self.connection.master
        try:
            waiter().wait(timeout=1.0)
        except TimeoutError:
            pass

        master.mav.set_position_target_local_ned_send(
            0,
            master.target_system,
            master.target_component,
            frame,
            POSITION_ONLY_MASK,  
            x, y, z,
            0, 0, 0,
            0, 0, 0,
            0,
            0
        )

    def _send_position_target_global_int(
        self,
        lat: float,
        lon: float,
        z: float,
        waiter: Callable[[], CommandHandle]
    ) -> None:
        """Send a single SET_POSITION_TARGET_GLOBAL_INT with coordinates."""
        master = self.connection.master
        try:
            waiter().wait(timeout=1.0)
        except TimeoutError:
            pass

        master.mav.set_position_target_global_int_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            POSITION_ONLY_MASK,  
            lat, lon, z,
            0, 0, 0,
            0, 0, 0,
            0,
            0
        )

    def _send_position_hold(self) -> None:
        """Hold current position by targeting it immediately."""
        local = self.telemetry.local_position
        if local is None:
            return
        master = self.connection.master
        master.mav.set_position_target_local_ned_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            POSITION_ONLY_MASK,
            local.x, local.y, local.z,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

        # TODO: Implement precision landing    