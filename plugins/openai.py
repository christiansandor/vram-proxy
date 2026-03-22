"""
Plugin: Generic OpenAI-compatible service
=========================================
type: openai

Use this for any service that:
  - Exposes a standard GET /v1/models endpoint (for model discovery)
  - Has no unload / free-memory mechanism

Typical candidates: openedai-whisper, Kokoro TTS, LocalAI, Ollama, etc.

Unload behaviour
----------------
unload() is a deliberate no-op.  These services generally hold little or no
persistent VRAM between requests (audio models, small embeddings, etc.), so
evicting them before routing to an LLM is usually unnecessary.

If you need real unload support for a specific service, create a dedicated
plugin (e.g. plugins/ollama.py with type_name = 'ollama') and implement
the appropriate unload call there.

Config example
--------------
  services:
    - name: Whisper
      type: openai
      baseUrl: http://host.docker.internal:8000
      routes:
        - /v1/audio/transcriptions
        - /v1/audio/translations

    - name: Kokoro TTS
      type: openai
      baseUrl: http://host.docker.internal:8880
      routes:
        - /v1/audio/speech
"""

import http.client
import json
from urllib.parse import urlparse

from plugins import ServicePlugin


def _http_get_json(base_url: str, path: str) -> tuple[int, dict]:
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    use_https = parsed.scheme == 'https'

    conn = None
    try:
        conn = (
            http.client.HTTPSConnection(host, port, timeout=10)
            if use_https
            else http.client.HTTPConnection(host, port, timeout=10)
        )
        conn.request('GET', path)
        resp = conn.getresponse()
        try:
            body = json.loads(resp.read())
        except Exception:
            body = {}
        return resp.status, body
    except Exception as exc:
        print(f"  [openai] HTTP error (GET {path}): {exc}", flush=True)
        return 0, {}
    finally:
        if conn:
            conn.close()


class OpenAICompatiblePlugin(ServicePlugin):
    type_name = 'openai'

    def get_models(self, service: dict) -> list:
        status, body = _http_get_json(service['baseUrl'], '/v1/models')
        if status == 200:
            return body.get('data', [])
        if status != 0:
            print(f"  [openai] get_models: HTTP {status} from {service['name']}", flush=True)
        return []

    def unload(self, service: dict) -> None:
        # No unload mechanism — this is intentional, not an oversight.
        print(
            f"  [openai] {service['name']}: no unload endpoint, skipping",
            flush=True,
        )


PLUGIN = OpenAICompatiblePlugin()
