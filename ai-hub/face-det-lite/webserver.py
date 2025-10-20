import asyncio
import signal
import threading, io, base64, json
from pathlib import Path
from typing import Set, Optional, Iterable, Union
import socket
from aiohttp import web, WSMsgType

class ThreadedAiohttpServer:
    """
    Simple aiohttp HTTP + WebSocket server that runs in its own thread.
    - Static files from `static_dir` at "/"
    - WebSocket endpoint at "/ws"
    - Call `broadcast(text_or_bytes)` from any thread to push to all clients
    - Call `stop()` to shut down (unbinds ports); also handles Ctrl+C if you wire it up
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        static_dir: Union[str, Path] = "./public",
        show_index: bool = True,
    ):
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir).resolve()
        self.show_index = show_index

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        self._started_evt = threading.Event()
        self._stopped_evt = threading.Event()

        # Track live websockets
        self._clients: Set[web.WebSocketResponse] = set()
        self._clients_lock = threading.Lock()

    # ------------- Public API -------------

    def start(self) -> None:
        """Start the server in a new thread and block until it's accepting connections."""
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run_loop, name="aiohttp-server", daemon=False)
        self._thread.start()
        # Wait until the site is up or an error happened
        self._started_evt.wait()

    def stop(self) -> None:
        """Stop the server and join the thread."""
        if not self._loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            fut.result(timeout=10)
        except Exception:
            pass  # best-effort shutdown
        # Ask the loop to stop
        self._loop.call_soon_threadsafe(self._loop.stop)
        # Join the thread
        if self._thread:
            self._thread.join(timeout=10)
        self._stopped_evt.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the server thread to finish (optional)."""
        if self._thread:
            self._thread.join(timeout=timeout)

    def broadcast(self, data: Union[str, bytes]) -> None:
        """
        Thread-safe broadcast to all connected WebSocket clients.
        Accepts str (text) or bytes (binary).
        """
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_coro(data), self._loop)

    def broadcast_img(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        self.broadcast(json.dumps({
            'type': 'image',
            'dataUrl': "data:image/jpeg;base64," + img_base64,
        }))

    # ------------- Internal: thread / loop -------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_async())
            self._started_evt.set()
            self._loop.run_forever()
        finally:
            # Ensure cleanup if the loop exits unexpectedly
            try:
                self._loop.run_until_complete(self._shutdown())
            except Exception:
                pass
            finally:
                self._loop.close()

    async def _start_async(self) -> None:
        # Create the app and routes
        self._app = web.Application()

        # WebSocket handler first (route order matters)
        self._app.router.add_get("/ws", self._ws_handler)

        # Static file serving at root
        # - If a file doesn't exist, aiohttp returns 404 automatically.
        if not self.static_dir.exists():
            # Create empty directory to avoid errors if missing
            self.static_dir.mkdir(parents=True, exist_ok=True)
        self._app.router.add_static("/", str(self.static_dir), show_index=self.show_index)

        # Runner + Site
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

    async def _shutdown(self) -> None:
        # Close all websockets first
        with self._clients_lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.close(code=1001, message=b"Server shutdown")
            except Exception:
                pass

        # Tear down the site/runner (unbind port)
        if self._site:
            try:
                await self._site.stop()
            except Exception:
                pass
        if self._runner:
            try:
                await self._runner.cleanup()  # closes the server socket
            except Exception:
                pass

    # ------------- Internal: websocket -------------

    async def _ws_handler(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)

        with self._clients_lock:
            self._clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Echo back or handle messages as needed
                    await ws.send_str(f"echo: {msg.data}")
                elif msg.type == WSMsgType.BINARY:
                    await ws.send_bytes(msg.data)  # echo
                elif msg.type == WSMsgType.ERROR:
                    # Loggable: ws.exception()
                    break
        finally:
            with self._clients_lock:
                self._clients.discard(ws)

        return ws

    async def _broadcast_coro(self, data: Union[str, bytes]) -> None:
        # Send to all currently open sockets
        with self._clients_lock:
            clients: Iterable[web.WebSocketResponse] = list(self._clients)

        to_remove = []
        for ws in clients:
            if ws.closed:
                to_remove.append(ws)
                continue
            try:
                if isinstance(data, (bytes, bytearray)):
                    await ws.send_bytes(data)
                else:
                    await ws.send_str(str(data))
            except Exception:
                # If send fails, drop client
                to_remove.append(ws)

        if to_remove:
            with self._clients_lock:
                for ws in to_remove:
                    self._clients.discard(ws)

def get_ip_addr():
    # Try to find the machine’s non-loopback IP address
    # (either from hostname lookup or from a dummy UDP connection)

    # 1. Attempt to get the first non-127.* address from hostname lookup
    hostname_ips = socket.gethostbyname_ex(socket.gethostname())[2]
    non_loopback_ips = [ip for ip in hostname_ips if not ip.startswith("127.")]
    first_hostname_ip = non_loopback_ips[:1]  # might be empty
    if (len(first_hostname_ip) > 0):
        return first_hostname_ip[0]

    # 2. Alternative: create a UDP socket to 8.8.8.8 (Google DNS) just to learn our outbound IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 53))
        ip_via_socket = s.getsockname()[0]
        return ip_via_socket
    except Exception as e:
        return '127.0.0.1'
    finally:
        s.close()
