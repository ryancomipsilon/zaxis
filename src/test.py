from zaxis.drone import Drone
from zaxis.telemetry import MavlinkConnection
from zaxis.sensors.benewake.tfluna import TFLuna
import time
from zaxis.control.flight import FlightControls
from zaxis.runtime import CommandHandle, RuntimeLoop

con = MavlinkConnection()
con.connect("/dev/ttyTHS1", 921600)
drone = Drone(con)

drone.set_mode(drone.FlightMode.LOITER).wait(callback=lambda: print("Mode being set"), frequency=0.01)

tfluna = TFLuna("/dev/ttyUSB0")
drone.rngfnd_filter(sensor=tfluna).wait()