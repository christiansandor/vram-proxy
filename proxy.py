"""
vram-proxy — single-machine OpenAI-compatible proxy with VRAM management
========================================================================
All AI requests are serialized through a global lock.  Before forwarding
to a target service, every *other* service is unloaded synchronously:

  - Services with a `container:` block → docker stop  (kills process + VRAM)
  - Services without                   → plugin.unload() (API call to free VRAM)

After unloading others, if the target has a `container:` block it is started
and health-polled before the request is forwarded.

GET /v1/models aggregates model lists from all services and is never blocked
by the lock — it always responds immediately.
"""

import http.client
import importlib
import json
import os
import pkgutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import yaml

import docker_manager as dm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = 'config.yaml'

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Config watcher
# ---------------------------------------------------------------------------

def _print_services(cfg: dict) -> None:
    for svc in cfg.get('services', []):
        c = svc.get('container')
        if c:
            ttl_str = f", ttl={c['ttl']}s" if c.get('ttl') else ', no ttl'
            extra = f"  [container: {c['name']}{ttl_str}]"
        else:
            extra = ''
        prefix = svc.get('prefix', '')
        prefix_str = f'/{prefix}' if prefix else '(no prefix)'
        print(
            f"  [{svc.get('type', '?')}]  {svc['name']}  →  {svc['baseUrl']}"
            f"  prefix={prefix_str}{extra}",
            flush=True,
        )
        for route in svc.get('routes', []):
            display = f"/{prefix}{route}" if prefix else route
            print(f"          {display}", flush=True)


def _start_config_watcher() -> None:
    """
    Background thread that polls config.yaml for changes every 2 seconds.
    On change, reloads the file and updates the live `config` dict in-place
    so all running request handlers see the new values immediately.

    The update is done under the request lock so config never changes
    mid-request.
    """
    def _watch():
        last_mtime = os.path.getmtime(CONFIG_PATH)
        while True:
            time.sleep(2)
            try:
                mtime = os.path.getmtime(CONFIG_PATH)
                if mtime == last_mtime:
                    continue
                last_mtime = mtime

                with open(CONFIG_PATH) as f:
                    new_cfg = yaml.safe_load(f)

                # Acquire the request lock so we don't swap config mid-request.
                with _request_lock:
                    config.clear()
                    config.update(new_cfg)

                print(f"\n↺ config.yaml reloaded — active services:", flush=True)
                _print_services(config)
                print('', flush=True)

            except Exception as exc:
                print(f"  [config] Reload failed: {exc}", flush=True)

    threading.Thread(target=_watch, daemon=True, name='config-watcher').start()


# ---------------------------------------------------------------------------
# Global serialising lock
#
# Only one AI request runs at a time.  Concurrent callers queue here.
# The TTL watchdog also acquires this lock (non-blocking) before stopping
# idle containers, so it never interrupts an active request.
# ---------------------------------------------------------------------------

_request_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Plugin registry  (type_name -> ServicePlugin instance)
# ---------------------------------------------------------------------------

_plugins: dict = {}


def _load_plugins() -> None:
    import plugins as pkg
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f'plugins.{name}')
        plugin = getattr(module, 'PLUGIN', None)
        if plugin is not None:
            _plugins[plugin.type_name] = plugin
            print(f"  ↳ plugin loaded: {plugin.type_name}  (plugins/{name}.py)", flush=True)


def _get_plugin(service: dict):
    t = service.get('type', '<unset>')
    plugin = _plugins.get(t)
    if plugin is None:
        raise RuntimeError(f"No plugin registered for service type '{t}'")
    return plugin


# ---------------------------------------------------------------------------
# Prefix helpers
# ---------------------------------------------------------------------------

def _svc_prefix(service: dict) -> str:
    """
    Return the configured prefix for a service, normalised to a plain string
    with no leading/trailing slashes.  Empty string means no prefix.
    """
    return (service.get('prefix') or '').strip('/')


def _parse_models_path(path: str) -> tuple[str, bool]:
    """
    Detect whether `path` is a models endpoint and extract its prefix.

    Recognised patterns:
      /v1/models          → prefix='',       is_models=True
      /models             → prefix='',       is_models=True
      /openai/v1/models   → prefix='openai', is_models=True
      /comfyui/models     → prefix='comfyui',is_models=True
      /anything/else      → prefix='',       is_models=False

    Returns (prefix, is_models).
    """
    # Strip leading slash, split into segments
    parts = path.lstrip('/').split('/')

    # /v1/models  or  /models
    if parts == ['v1', 'models'] or parts == ['models']:
        return '', True

    # /<prefix>/v1/models  or  /<prefix>/models
    if len(parts) >= 2 and parts[-1] == 'models':
        if len(parts) >= 3 and parts[-2] == 'v1':
            return parts[0], True   # e.g. ['openai', 'v1', 'models']
        if len(parts) == 2:
            return parts[0], True   # e.g. ['comfyui', 'models']

    return '', False


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):

    def do_GET(self):    self.handle_request()
    def do_POST(self):   self.handle_request()
    def do_PUT(self):    self.handle_request()
    def do_DELETE(self): self.handle_request()

    def log_message(self, fmt, *args):
        pass  # We print our own logs

    # ── Entry point ─────────────────────────────────────────────────────────

    def handle_request(self) -> None:
        self.close_connection = True
        print(f"→ {self.command} {self.path}", flush=True)

        base_path = self.path.split('?')[0].rstrip('/')

        # Health check — always 200, never touches the lock.
        if self.command == 'GET' and base_path == '/health':
            body = json.dumps({'status': 'ok'}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Models endpoint: /<prefix>/v1/models, /<prefix>/models, or
        # /v1/models / /models for prefix-less services.
        if self.command == 'GET':
            prefix, is_models = _parse_models_path(base_path)
            if is_models:
                self._serve_models(prefix)
                return

        # Read body before acquiring the lock so a slow upload doesn't
        # block other requests from even queuing.
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else None

        match = self._find_service(self.path)
        if match is None:
            print(f"✗ No service configured for {self.path}", flush=True)
            self.send_error(404, "No service configured for this route")
            return

        target, stripped_path = match

        try:
            _get_plugin(target)  # Validate the plugin exists before queuing
        except RuntimeError as exc:
            self.send_error(500, str(exc))
            return

        print(f"  Waiting for lock...", flush=True)
        with _request_lock:
            print(
                f"  → Lock acquired, routing to "
                f"[{target.get('type','?')}] {target['name']}",
                flush=True,
            )
            self._unload_others(target)
            dm.ensure_started(target)
            self._forward(target, body, stripped_path)
            dm.touch(target)  # Reset TTL idle timer after request completes

        print(f"✓ Done {self.command} {self.path}", flush=True)

    # ── VRAM management ─────────────────────────────────────────────────────

    def _unload_others(self, target: dict) -> None:
        """
        Synchronously free VRAM on every service except the target.

        Container-backed services get a docker stop — the cleanest possible
        eviction since it terminates the process entirely.  Plugin-backed
        services get their plugin's unload() API call.
        """
        for svc in config.get('services', []):
            if svc is target:
                continue

            if dm.has_container(svc):
                dm.stop_container(svc)
            else:
                try:
                    _get_plugin(svc).unload(svc)
                except RuntimeError:
                    pass  # No plugin — skip silently
                except Exception as exc:
                    print(f"  ✗ Unload error for {svc['name']}: {exc}", flush=True)

    # ── Model aggregation (lock-free) ────────────────────────────────────────

    def _serve_models(self, prefix: str) -> None:
        all_models: list = []
        seen: set = set()

        for svc in config.get('services', []):
            if _svc_prefix(svc) != prefix:
                continue
            try:
                models = _get_plugin(svc).get_models(svc)
                added = 0
                for m in models:
                    mid = m.get('id', '')
                    if mid and mid not in seen:
                        seen.add(mid)
                        all_models.append(m)
                        added += 1
                print(f"  [models] {svc['name']}: {added} model(s)", flush=True)
            except Exception as exc:
                print(f"  [models] {svc['name']} error: {exc}", flush=True)

        body = json.dumps({'object': 'list', 'data': all_models}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        prefix_label = f'/{prefix}' if prefix else ''
        print(f"✓ {prefix_label}/v1/models → {len(all_models)} model(s)", flush=True)

    # ── Routing ──────────────────────────────────────────────────────────────

    def _find_service(self, path: str) -> tuple[dict, str] | None:
        """
        Return (service, stripped_path) for the first service whose
        prefix+route matches the request path, or None.

        stripped_path has the service prefix removed so the upstream
        receives the path it natively expects.
        """
        for svc in config.get('services', []):
            p = _svc_prefix(svc)
            prefix_seg = f'/{p}' if p else ''
            for route in svc.get('routes', []):
                full_route = prefix_seg + route
                if path == full_route or path.startswith(full_route + '/') or path.startswith(full_route + '?'):
                    stripped = path[len(prefix_seg):] if prefix_seg else path
                    return svc, stripped
        return None

    # ── Forwarding ───────────────────────────────────────────────────────────

    def _forward(self, service: dict, body: bytes | None, path: str | None = None) -> None:
        """
        Proxy the request to the target service and stream the response back.
        SSE (text/event-stream) is forwarded line-by-line; everything else
        in 8 KiB chunks.  The lock is held for the full duration of streaming.

        `path` is the prefix-stripped path to send upstream; defaults to
        self.path (used when no prefix is configured).
        """
        forward_path = path if path is not None else self.path
        parsed = urlparse(service['baseUrl'])
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        use_https = parsed.scheme == 'https'

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ('host', 'content-length', 'transfer-encoding')
        }
        if body is not None:
            headers['Content-Length'] = str(len(body))

        conn = None
        try:
            conn = (
                http.client.HTTPSConnection(host, port, timeout=600)
                if use_https
                else http.client.HTTPConnection(host, port, timeout=600)
            )
            conn.request(self.command, forward_path, body=body, headers=headers)
            response = conn.getresponse()

            self.send_response(response.status)
            for k, v in response.getheaders():
                if k.lower() not in ('transfer-encoding',):
                    self.send_header(k, v)
            self.send_header('Connection', 'close')
            self.send_header('X-Vram-Proxy-Service', service['name'])
            self.end_headers()

            content_type = response.getheader('content-type', '')
            try:
                if 'text/event-stream' in content_type:
                    for line in response:
                        self.wfile.write(line)
                        self.wfile.flush()
                        if b'[DONE]' in line:
                            break
                else:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except BrokenPipeError:
                pass
            except Exception as exc:
                print(f"  ✗ Streaming error: {exc}", flush=True)
            finally:
                try:
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

        except Exception as exc:
            print(f"  ✗ Forward to {service['name']} failed: {exc}", flush=True)
            try:
                self.send_error(502, f"Upstream error: {exc}")
            except Exception:
                pass
        finally:
            if conn:
                conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print("vram-proxy starting up...\n", flush=True)

    print("Loading plugins:", flush=True)
    _load_plugins()

    print("\nStarting config watcher:", flush=True)
    _start_config_watcher()

    print("\nStarting TTL watchdog:", flush=True)
    dm.start_ttl_watchdog(config, _request_lock)

    print("\nConfigured services:", flush=True)
    _print_services(config)

    httpd = ThreadingHTTPServer(
        ('', 8080),
        lambda req, addr, srv: ProxyHandler(req, addr, srv),
    )
    print("\nProxy listening on :8080\n", flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    run()
