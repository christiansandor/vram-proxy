# vram-proxy

A lightweight OpenAI-compatible proxy for single-machine AI setups with multiple GPU-bound services.

## The problem it solves

Running several AI backends on one machine (LM Studio, ComfyUI, Whisper, TTS...) means they fight over the same VRAM pool.  vram-proxy serialises all requests through a global lock and **unloads every other service before forwarding each request**, so only one backend holds GPU memory at a time.

Some services (like openedai-whisper) load their model at boot and offer no unload API — the only way to free their VRAM is to stop the container.  vram-proxy handles this too, starting and stopping containers on demand and auto-stopping idle ones via a configurable TTL.

## How it works

```
Client → vram-proxy :5234
             │
             ├─ GET /v1/models  ── lock-free ──────────────────────────────────▶ aggregated list
             │
             └─ Any other request
                 1. Read request body
                 2. Find matching service by route prefix
                 3. Acquire global lock  (queue concurrent requests here)
                 4. Unload every OTHER service:
                      container services → docker stop  (kills process + VRAM)
                      plugin services    → plugin.unload()  (API call)
                 5. Start target container if needed → poll health endpoint
                 6. Forward request → stream response
                 7. Release lock  (record last-used timestamp for TTL)
```

The lock is held for the full duration of streaming, so a long LLM generation queues up image generation requests — exactly what you want on a single GPU.

## Quick start

```bash
cp config.yaml.example config.yaml
# Edit config.yaml to match your services and ports
docker compose up --build
```

The proxy listens on **port 5234** (configurable in docker-compose.yaml).

## Configuration

```yaml
services:
  # Service without container management (LM Studio stays running always)
  - name: LM Studio
    type: lmstudio
    baseUrl: http://host.docker.internal:1234
    routes:
      - /v1/chat/completions
      - /v1/embeddings

  # Service WITH container management
  - name: Whisper
    type: openai
    baseUrl: http://host.docker.internal:8000
    container:
      name: openedai-whisper    # name of an existing docker container
      health_path: /health      # polled after start until 200 (default: /health)
      start_timeout: 60         # seconds before giving up on health (default: 60)
      stop_timeout: 15          # seconds for graceful docker stop (default: 30)
      poll_interval: 2          # seconds between health polls (default: 2)
      ttl: 300                  # auto-stop after 5 min idle (omit = never)
    routes:
      - /v1/audio/transcriptions
```

### Container management behaviour

| Scenario | What happens |
|----------|-------------|
| Request arrives, container stopped | `docker start` → poll health → forward |
| Request arrives, container already running | Forward immediately (no restart) |
| Another service needs VRAM | `docker stop` on this container (synchronous) |
| TTL expires, no active request | `docker stop` automatically |
| TTL expires during active request | Skipped — fires on next idle cycle |
| Health poll times out | Log warning, forward anyway (upstream may 502) |

### Routes

Routing uses **prefix matching** — the first service whose `routes` list contains a prefix of the request path wins.  Put more specific routes before broader ones if they could overlap.

Do **not** add `/v1/models` to any service's routes — the proxy owns that endpoint.

## Service types & plugins

| type | `get_models` source | `unload` mechanism |
|------|--------------------|--------------------|
| `lmstudio` | `GET /v1/models` | `GET /api/v1/models` → `POST /api/v1/models/unload` per model |
| `comfyui` | Declared in `config.yaml` under `models:` | `POST /free {"unload_models":true,"free_memory":true}` |
| `openai` | `GET /v1/models` | No-op (log only) — pair with `container:` for full unload |

## Adding a custom service type

Create `plugins/myservice.py`:

```python
from plugins import ServicePlugin

class MyServicePlugin(ServicePlugin):
    type_name = 'myservice'   # matches `type:` in config.yaml

    def get_models(self, service: dict) -> list:
        # Return [{"id": "...", "object": "model", "owned_by": "..."}, ...]
        return []

    def unload(self, service: dict) -> None:
        # Call your service's unload/free endpoint here.
        # Must BLOCK until VRAM is released.
        # No-op is fine if you're using container: stop instead.
        pass

PLUGIN = MyServicePlugin()
```

The proxy discovers plugins automatically — no registration needed.

## Response headers

Every proxied response includes:

| Header | Value |
|--------|-------|
| `X-Vram-Proxy-Service` | Name of the service that handled the request |

## Notes

- **Docker socket**: the docker-compose.yaml mounts `/var/run/docker.sock` into the container. Remove it if you don't use any `container:` blocks.
- **`host.docker.internal`**: works automatically on Windows/Mac Docker Desktop. On Linux it requires the `extra_hosts` entry in docker-compose.yaml (already included).
- **Queueing**: requests queue at the lock. A 60-second generation will make a new TTS request wait 60 seconds before unloading begins — intentional.
- **`/v1/models`**: always lock-free, so it responds instantly even mid-generation.
- **ComfyUI `/free`**: requires a reasonably recent ComfyUI version. A non-200 response logs a warning and continues.
- **No health checks for routing**: the proxy forwards unconditionally and lets upstream errors propagate as 502s.
