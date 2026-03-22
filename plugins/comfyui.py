"""
Plugin: ComfyUI
===============
type: comfyui

ComfyUI does not expose an OpenAI-compatible model list, so models are
declared directly in config.yaml under a `models` key for this service.

Unload strategy
---------------
ComfyUI exposes a POST /free endpoint (added in recent versions) that
evicts loaded checkpoints and frees GPU memory:

    POST /free
    { "unload_models": true, "free_memory": true }

This call blocks until ComfyUI confirms the memory has been released.

Config example
--------------
  services:
    - name: ComfyUI
      type: comfyui
      baseUrl: http://host.docker.internal:8188
      models:
        - id: stable-diffusion-xl
          description: SDXL 1.0
        - id: flux-dev
          description: Flux.1 dev
      routes:
        - /prompt
        - /queue
        - /history
        - /upload
        - /view
        - /object_info

Notes
-----
- If your ComfyUI version predates the /free endpoint, unload() will log a
  warning and continue.  In that case, consider restarting the container
  between services instead.
- Routes above are ComfyUI's native API paths.  If you sit another proxy in
  front (e.g. to namespace under /comfyui/), adjust accordingly.
"""

import http.client
import json
from urllib.parse import urlparse

from plugins import ServicePlugin


def _http_post_json(base_url: str, path: str, body: dict) -> tuple[int, bytes]:
    """POST JSON, return (status, raw_body_bytes). Returns (0, b'') on error."""
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or 80

    body_bytes = json.dumps(body).encode()
    headers = {
        'Content-Type': 'application/json',
        'Content-Length': str(len(body_bytes)),
    }
    conn = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        conn.request('POST', path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    except Exception as exc:
        print(f"  [comfyui] HTTP error (POST {path}): {exc}", flush=True)
        return 0, b''
    finally:
        if conn:
            conn.close()


class ComfyUIPlugin(ServicePlugin):
    type_name = 'comfyui'

    # ── Model discovery ──────────────────────────────────────────────────────

    def get_models(self, service: dict) -> list:
        """
        ComfyUI has no OpenAI-format model list endpoint.
        Return whatever is declared under `models:` in config.yaml.
        """
        raw = service.get('models', [])
        models = []
        for m in raw:
            mid = m.get('id', '').strip()
            if not mid:
                continue
            models.append({
                'id': mid,
                'object': 'model',
                'owned_by': service.get('name', 'comfyui'),
                # Pass through any extra keys declared in config (description, etc.)
                **{k: v for k, v in m.items() if k != 'id'},
            })
        return models

    # ── VRAM management ──────────────────────────────────────────────────────

    def unload(self, service: dict) -> None:
        """
        Call ComfyUI's /free endpoint to evict loaded checkpoints from VRAM.
        """
        print(f"  [comfyui] Freeing GPU memory on {service['name']} ...", flush=True)
        status, _ = _http_post_json(
            service['baseUrl'],
            '/free',
            {'unload_models': True, 'free_memory': True},
        )
        if status == 200:
            print(f"  [comfyui] ✓ GPU memory freed", flush=True)
        elif status == 0:
            # Service is not running — nothing to unload.
            pass
        else:
            print(
                f"  [comfyui] ✗ /free returned HTTP {status} "
                f"(is your ComfyUI version recent enough?)",
                flush=True,
            )


PLUGIN = ComfyUIPlugin()
