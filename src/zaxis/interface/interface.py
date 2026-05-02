import socket
import subprocess
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from zaxis.telemetry import MavlinkConnection

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_free_port() -> int:
    """Bind to port 0 and let the OS pick a free one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
    
def _get_local_ip() -> str:
    """Get the real outbound network IP, not the loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80)) 
        return s.getsockname()[0]

def _spawn_with_restart(cmd: list[str], name: str) -> None:
    """Run a subprocess, restarting it automatically if it dies."""
    def _run() -> None:
        while True:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                proc.wait()
                print(f"[z-axis] {name} exited, restarting...")
            except Exception as e:
                print(f"[z-axis] {name} failed to start: {e}, retrying in 2s...")
            time.sleep(2)

    thread = threading.Thread(target=_run, daemon=True, name=name)
    thread.start()

# ── Interface ─────────────────────────────────────────────────────────────────

_HTML_PATH = Path(__file__).parent / "index.html"

class MissionInterface:
    def __init__(self, connection: MavlinkConnection) -> None:
        self.connection = connection
        self.state = connection.state
        self._port: int | None = None
        self._ttyd_port: int | None = None
        self._vscode_port: int | None = None
        self._app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/")
        def serve_ui() -> FileResponse:
            return FileResponse(_HTML_PATH, media_type="text/html")

        @app.get("/config")
        def config() -> JSONResponse:
            return JSONResponse({
                "ttyd_port":   self._ttyd_port,
                "vscode_port": self._vscode_port,
            })

        return app

    def _start_ttyd(self) -> None:
        self._ttyd_port = _find_free_port()
        _spawn_with_restart(
            ["sudo", "ttyd", "-p", str(self._ttyd_port), "bash"],
            name="ttyd"
        )

    def _start_vscode(self) -> None:
        self._vscode_port = _find_free_port()
        home = str(Path.home())
        _spawn_with_restart(
            ["code-server", "--bind-addr", f"0.0.0.0:{self._vscode_port}", "--auth", "none", home],
            name="code-server"
        )

    def start(self) -> None:
        """Start ttyd, code-server, and the interface server."""
        self._start_ttyd()
        self._start_vscode()

        self._port = _find_free_port()
        config = uvicorn.Config(
            app=self._app,
            host="0.0.0.0",
            port=self._port,
            log_level="error"
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True, name="uvicorn")
        thread.start()

        ip = _get_local_ip()
        print(f"[z-axis] Interface running on http://{ip}:{self._port}")