# ===================== IMPORTS ======================
import threading
import time
# =====================================================
_UNSET = object()
# =====================================================
class CommandHandle:
    def __init__(self, default_callback=None, default_frequency=30):
        self._done = threading.Event()
        self._status = "PENDING"
        self._default_callback = default_callback
        self._default_frequency = default_frequency
        self._on_stop: callable | None = None  # set by RuntimeLoop on register

    def _finish(self, success=True):
        self._status = "COMPLETED" if success else "FAILED"
        self._done.set()

    def stop(self) -> None:
        """Cancel this command externally and trigger its cleanup."""
        if self._done.is_set():
            return
        if self._on_stop:
            self._on_stop()
        self._finish(success=False)

    def wait(self, timeout=None, callback=_UNSET, frequency=_UNSET):
        cb = callback if callback is not _UNSET else self._default_callback
        freq = frequency if frequency is not _UNSET else self._default_frequency
        if cb is None:
            finished = self._done.wait(timeout)
            if not finished:
                self.stop()
            return self.success()
        period = 1.0 / freq
        deadline = time.monotonic() + timeout if timeout else None
        while not self._done.is_set():
            if deadline and time.monotonic() >= deadline:
                self.stop()
            cb()
            remaining = None
            if deadline:
                remaining = min(period, deadline - time.monotonic())
                remaining = max(remaining, 0)
            else:
                remaining = period
            self._done.wait(remaining)
        return self.success()

    def success(self):
        return self._status == "COMPLETED"

    def done(self):
        return self._done.is_set()