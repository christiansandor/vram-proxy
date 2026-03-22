"""
docker_manager.py — container lifecycle for vram-proxy
=======================================================
Handles start, stop, and health-polling for services that declare a
`container:` block in config.yaml.

All Docker operations use the CLI via subprocess — no SDK dependency, and
it works as long as the Docker socket is mounted into the proxy container.

Public API (used by proxy.py)
------------------------------
  ensure_started(service)           → start + poll health, then return
  stop_container(service)           → synchronous docker stop
  touch(service)                    → record "last used now" for TTL tracking
  start_ttl_watchdog(config, lock)  → launch background idle-stopper thread

Config shape (optional block on any service)
--------------------------------------------
  container:
    name: whisper                 # existing docker container name (required)
    health_path: /health          # GET path polled after start (default: /health)
    start_timeout: 60             # seconds to wait for healthy (default: 60)
    stop_timeout: 30              # seconds for graceful docker stop (default: 30)
    ttl: 300                      # auto-stop after N seconds idle (omit = never)
    poll_interval: 2              # health poll cadence in seconds (default: 2)
"""

import http.client
import subprocess
import threading
import time
from urllib.parse import urlparse

# ── Per-service last-used timestamps for TTL tracking ───────────────────────
# service name → float (epoch of last completed request)
_last_used: dict[str, float] = {}
_last_used_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _container_cfg(service: dict) -> dict | None:
    return service.get('container')


def _run(args: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run a docker CLI command; return (returncode, combined output)."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, f"timed out after {timeout}s"
    except Exception as exc:
        return 1, str(exc)


def _is_running(container_name: str) -> bool:
    code, out = _run(
        ['docker', 'inspect', '--format', '{{.State.Running}}', container_name]
    )
    return code == 0 and out.strip().lower() == 'true'


def _poll_healthy(base_url: str, health_path: str, start_timeout: int, poll_interval: float) -> bool:
    """
    Poll GET <base_url><health_path> until HTTP 200 or timeout expires.
    Returns True if healthy within the deadline, False otherwise.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    use_https = parsed.scheme == 'https'

    deadline = time.time() + start_timeout

    while time.time() < deadline:
        try:
            conn = (
                http.client.HTTPSConnection(host, port, timeout=5)
                if use_https
                else http.client.HTTPConnection(host, port, timeout=5)
            )
            conn.request('GET', health_path)
            resp = conn.getresponse()
            conn.close()
            if resp.status == 200:
                return True
        except Exception:
            pass

        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(min(poll_interval, remaining))

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def has_container(service: dict) -> bool:
    """True if this service has a `container:` block in config."""
    return 'container' in service


def touch(service: dict) -> None:
    """Record that this service was just used, resetting its TTL countdown."""
    with _last_used_lock:
        _last_used[service['name']] = time.time()


def ensure_started(service: dict) -> None:
    """
    Start the service's container if it isn't running, then poll the health
    endpoint until it responds 200 or start_timeout is reached.

    On timeout we log a warning and return without raising — the proxy will
    forward the request anyway and let the upstream error propagate naturally.

    No-op for services without a `container:` block.
    """
    cfg = _container_cfg(service)
    if cfg is None:
        return

    name          = cfg['name']
    health_path   = cfg.get('health_path',   '/health')
    start_timeout = int(cfg.get('start_timeout', 60))
    poll_interval = float(cfg.get('poll_interval', 2))

    if _is_running(name):
        print(f"  [docker] {name!r}: already running", flush=True)
        return

    print(f"  [docker] Starting {name!r} ...", flush=True)
    code, out = _run(['docker', 'start', name])
    if code != 0:
        print(f"  [docker] ✗ docker start {name!r} failed: {out}", flush=True)
        print(f"  [docker]   Forwarding anyway — upstream will error if unavailable", flush=True)
        return

    print(
        f"  [docker] Waiting for {health_path} to return 200 "
        f"(timeout={start_timeout}s) ...",
        flush=True,
    )
    healthy = _poll_healthy(service['baseUrl'], health_path, start_timeout, poll_interval)

    if healthy:
        print(f"  [docker] ✓ {name!r} is healthy", flush=True)
    else:
        print(
            f"  [docker] ✗ {name!r} did not become healthy within {start_timeout}s "
            f"— forwarding anyway",
            flush=True,
        )


def stop_container(service: dict) -> None:
    """
    Stop a container-backed service synchronously.
    No-op if the container isn't running or has no `container:` block.
    """
    cfg = _container_cfg(service)
    if cfg is None:
        return

    name         = cfg['name']
    stop_timeout = int(cfg.get('stop_timeout', 30))

    if not _is_running(name):
        print(f"  [docker] {name!r}: already stopped", flush=True)
        return

    print(f"  [docker] Stopping {name!r} ...", flush=True)
    code, out = _run(
        ['docker', 'stop', '--time', str(stop_timeout), name],
        timeout=stop_timeout + 15,
    )
    if code == 0:
        print(f"  [docker] ✓ {name!r} stopped", flush=True)
    else:
        print(f"  [docker] ✗ docker stop {name!r} failed: {out}", flush=True)


# ---------------------------------------------------------------------------
# TTL watchdog
# ---------------------------------------------------------------------------

_WATCHDOG_INTERVAL = 10  # seconds between idle checks


def start_ttl_watchdog(config: dict, request_lock: threading.Lock) -> None:
    """
    Launch a daemon thread that stops idle containers whose TTL has expired.

    The thread acquires request_lock non-blocking before acting — if a
    request is currently in flight it skips the cycle entirely, so containers
    are never stopped mid-request.
    """
    # Check at least one service has a TTL before bothering to start the thread.
    if not any(
        has_container(s) and s['container'].get('ttl')
        for s in config.get('services', [])
    ):
        return

    print(f"  [ttl] Watchdog started (reads live config each cycle)", flush=True)

    def _watchdog():
        while True:
            time.sleep(_WATCHDOG_INTERVAL)

            # Don't interrupt an active request.
            acquired = request_lock.acquire(blocking=False)
            if not acquired:
                continue

            try:
                now = time.time()
                # Re-read services from config each cycle so changes to
                # config.yaml (TTL values, added/removed services) take
                # effect without a restart.
                services_with_ttl = [
                    svc for svc in config.get('services', [])
                    if has_container(svc) and svc['container'].get('ttl')
                ]
                for svc in services_with_ttl:
                    ttl  = int(svc['container']['ttl'])
                    name = svc['name']

                    with _last_used_lock:
                        last = _last_used.get(name)

                    # Never been used — leave it alone.
                    if last is None:
                        continue

                    idle_for = now - last
                    if idle_for < ttl:
                        continue

                    print(
                        f"  [ttl] {name!r} idle for {idle_for:.0f}s "
                        f"(ttl={ttl}s) — stopping",
                        flush=True,
                    )
                    stop_container(svc)

                    # Clear timestamp so it won't fire again until next use.
                    with _last_used_lock:
                        _last_used.pop(name, None)

            finally:
                request_lock.release()

    threading.Thread(target=_watchdog, daemon=True, name='ttl-watchdog').start()
