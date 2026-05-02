# ===================== IMPORTS ======================
import threading
import time
from zaxis.runtime.handle import CommandHandle
# =====================================================
class RuntimeCommand:
    def __init__(self, name, callback, hz, on_stop=None):
        self.name = name
        self.callback = callback
        self.hz = hz
        self.on_stop = on_stop  # cleanup
        self.handle: CommandHandle = CommandHandle()
        self._stop = threading.Event()

class RuntimeLoop:
    def __init__(self):
        self._commands = {}
        self._threads = {}
        self._lock = threading.RLock()

    # ==================== API HELPERS ====================

    def register(self, command):
        if command.hz <= 0:
            raise ValueError(f"Frequency must be > 0, got {command.hz}")
        with self._lock:
            if command.name in self._commands:
                self.unregister(command.name, success=False)
                
            def _on_stop():
                command._stop.set()
                if command.on_stop:
                    command.on_stop()

            command.handle._on_stop = _on_stop

            self._commands[command.name] = command
            t = threading.Thread(
                target=self._run_command,
                args=(command,),
                name=f"runtime-{command.name}",
                daemon=True,
            )
            self._threads[command.name] = t 
            t.start()

    def unregister(self, name, success=True):
        with self._lock:
            command = self._commands.pop(name, None)
            if command is None:
                return
            command._stop.set()
            command.handle._finish(success=success)
            t = self._threads.pop(name, None)
        if t:
            t.join(timeout=2.0)

    def stop(self):
        with self._lock:
            names = list(self._commands.keys())
            for name in names:
                self.unregister(name)

    @property
    def registered_commands(self):
        with self._lock:
            return list(self._commands.keys())

    # =====================================================

    def _run_command(self, command):
        period = 1.0 / command.hz
        while not command._stop.is_set():
            t0 = time.monotonic()
            command.callback()
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                command._stop.wait(timeout=sleep_time)