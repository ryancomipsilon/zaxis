class DistanceSensor:
    """
    Base interface for distance sensors.
    """

    def read(self) -> float | None:
        """
        Returns the distance in meters or None if invalid.
        """
        raise NotImplementedError

    def close(self) -> None:
        pass