"""
audit.py — structured audit logging for vram-proxy
====================================================
Writes one JSON line per request to audit_logs/audit.log.
Files rotate daily; `keep_days` old files are retained (default: 30).

Log record fields
-----------------
  ts          ISO-8601 UTC timestamp
  ip          Client IP address
  method      HTTP method (GET, POST, …)
  path        Raw request path (including query string)
  service     Matched service name, or "-" for unrouted / auth-failed requests
  status      HTTP status code returned to the client
  duration_ms Wall-clock milliseconds from first byte received to response sent
  auth        "ok" | "fail" | "exempt"
"""

import json
import logging
import logging.handlers
import os
import time

_AUDIT_DIR = 'audit_logs'
_LOG_FILE  = os.path.join(_AUDIT_DIR, 'audit.log')

_logger: logging.Logger | None = None


def setup(keep_days: int = 30) -> None:
    """
    Initialise the audit logger.  Safe to call multiple times — subsequent
    calls are no-ops.
    """
    global _logger
    if _logger is not None:
        return

    os.makedirs(_AUDIT_DIR, exist_ok=True)

    handler = logging.handlers.TimedRotatingFileHandler(
        _LOG_FILE,
        when='midnight',
        interval=1,
        backupCount=keep_days,
        utc=True,
        encoding='utf-8',
    )
    # One raw JSON line per record — no extra formatting.
    handler.setFormatter(logging.Formatter('%(message)s'))

    _logger = logging.getLogger('vram_proxy.audit')
    _logger.setLevel(logging.INFO)
    _logger.propagate = False   # Don't bubble up to the root logger
    _logger.addHandler(handler)

    print(
        f"  [audit] Logging to {_LOG_FILE!r} "
        f"(daily rotation, keep_days={keep_days})",
        flush=True,
    )


def record(
    *,
    ip: str,
    method: str,
    path: str,
    service: str,
    status: int,
    start_time: float,
    auth: str,
) -> None:
    """
    Write one audit record.  `start_time` should be the value of
    ``time.monotonic()`` captured at the beginning of the request.
    """
    if _logger is None:
        return

    entry = {
        'ts':          time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'ip':          ip,
        'method':      method,
        'path':        path,
        'service':     service,
        'status':      status,
        'duration_ms': round((time.monotonic() - start_time) * 1000),
        'auth':        auth,
    }
    _logger.info(json.dumps(entry, separators=(',', ':')))
