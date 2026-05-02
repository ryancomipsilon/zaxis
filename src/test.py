from zaxis.drone import Drone
from zaxis.telemetry import MavlinkConnection
import time
con = MavlinkConnection()
con.connect("udpin:0.0.0.0:14550", 921600)
drone = Drone(con)

drone.set_mode(drone.FlightMode.GUIDED).wait()
drone.arm().wait()
t0 = time.time()
drone.takeoff(1).wait(callback=lambda: print(drone.telemetry.local_position))
print("Takeoff ended in ", time.time() - t0, " seconds")
drone.goto_local(-6,-2,-3).wait()
drone.goto_local(0,2.5,-3).wait()
drone.goto_local(5,-2.5,-3).wait()

print("Land sent")
drone.land().wait()
print("Landed")