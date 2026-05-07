import serial
import time

from ..base import DistanceSensor


class TFLuna(DistanceSensor):
    HEADER = b'\x59\x59'
    FRAME_SIZE = 9

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.0,  # non-blocking
    ):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.buffer = bytearray()

    def _read_serial(self):
        data = self.ser.read(self.ser.in_waiting or 1)
        if data:
            self.buffer.extend(data)

    def _parse_frame(self) -> float | None:
        while len(self.buffer) >= self.FRAME_SIZE:
            # look for header
            for i in range(len(self.buffer) - self.FRAME_SIZE + 1):
                if self.buffer[i:i+2] == self.HEADER:
                    frame = self.buffer[i:i+self.FRAME_SIZE]

                    # remove everything up to the end of the frame
                    del self.buffer[:i+self.FRAME_SIZE]

                    # checksum
                    if (sum(frame[:8]) & 0xFF) != frame[8]:
                        return None

                    dist_cm = frame[2] | (frame[3] << 8)
                    strength = frame[4] | (frame[5] << 8)

                    if strength < 100 or strength == 65535:
                        return None

                    return dist_cm / 100.0

            # if no header is found, discard 1 byte (desynchronized)
            self.buffer.pop(0)

        return None
    
    def read(self) -> float | None:
        self._read_serial()

        latest = None

        while True:
            val = self._parse_frame()
            if val is None:
                break
            latest = val

        return latest

    def close(self):
        self.ser.close()