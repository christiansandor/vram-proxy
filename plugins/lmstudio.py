"""
Plugin: LM Studio
=================
type: lmstudio

LM Studio exposes two relevant API surfaces:

  GET  /v1/models              → all *downloaded* models (OpenAI format)
                                 used for model discovery / /v1/models aggregation
  GET  /api/v1/models          → currently *loaded* models (LM Studio native)
                                 used to know what to unload
  POST /api/v1/models/unload   → unload a model by identifier
                                 body: { "identifier": "<model-id>" }

Unload strategy
---------------
1. Query /api/v1/models to see what is currently loaded in VRAM.
2. For each loaded model, POST to /api/v1/models/unload.
   LM Studio processes this synchronously, so the call blocks until the
   model has been fully evicted from GPU memory.
3. If nothing is loaded, skip — no unnecessary round-trips.

Config example
--------------
  services:
    - name: LM Studio
      type: lmstudio
      baseUrl: http://host.docker.internal:1234
      routes:
        - /v1/chat/completions
        - /v1/completions
        - /v1/embeddings
"""

import http.client
import json
from urllib.parse import urlparse

from plugins import ServicePlugin


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _http(base_url: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """
    Perform a single HTTP request and return (status_code, parsed_json).
    Returns (0, {}) on connection failure.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    use_https = parsed.scheme == 'https'

    body_bytes = json.dumps(body).encode() if body is not None else None
    headers = {}
    if body_bytes:
        headers['Content-Type'] = 'application/json'
        headers['Content-Length'] = str(len(body_bytes))

    conn = None
    try:
        conn = (
            http.client.HTTPSConnection(host, port, timeout=30)
            if use_https
            else http.client.HTTPConnection(host, port, timeout=30)
        )
        conn.request(method, path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        try:
            resp_body = json.loads(resp.read())
        except Exception:
            resp_body = {}
        return resp.status, resp_body
    except Exception as exc:
        print(f"  [lmstudio] HTTP error ({method} {path}): {exc}", flush=True)
        return 0, {}
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class LMStudioPlugin(ServicePlugin):
    type_name = 'lmstudio'

    # ── Model discovery ──────────────────────────────────────────────────────

    def get_models(self, service: dict) -> list:
        """
        Return all *downloaded* models via the standard OpenAI /v1/models
        endpoint.  This includes models that are not currently loaded.
        """
        status, body = _http(service['baseUrl'], 'GET', '/v1/models')
        if status == 200:
            return body.get('data', [])
        if status != 0:
            print(f"  [lmstudio] get_models: HTTP {status}", flush=True)
        return []

    # ── VRAM management ──────────────────────────────────────────────────────

    def unload(self, service: dict) -> None:
        """
        Unload every model currently resident in LM Studio's VRAM.
        Blocks until LM Studio confirms each unload.
        """
        status, body = _http(service['baseUrl'], 'GET', '/api/v1/models')
        if status == 0:
            # Service unreachable — nothing to unload.
            return
        if status != 200:
            print(f"  [lmstudio] unload: /api/v1/models returned HTTP {status}", flush=True)
            return

        loaded = body.get('data', [])
        if not loaded:
            print(f"  [lmstudio] {service['name']}: nothing loaded, skipping", flush=True)
            return

        for model in loaded:
            identifier = model.get('id', '').strip()
            if not identifier:
                continue

            print(f"  [lmstudio] Unloading {identifier!r} from {service['name']} ...", flush=True)
            status, _ = _http(
                service['baseUrl'],
                'POST',
                '/api/v1/models/unload',
                {'identifier': identifier},
            )
            if status == 200:
                print(f"  [lmstudio] ✓ Unloaded {identifier!r}", flush=True)
            else:
                print(f"  [lmstudio] ✗ Unload returned HTTP {status} for {identifier!r}", flush=True)


PLUGIN = LMStudioPlugin()
