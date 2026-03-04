"""zaxis - Autonomous Drone Control SDK."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zaxis")
except PackageNotFoundError:
    __version__ = "unknown"