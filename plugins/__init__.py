"""
Plugin interface for vram-proxy.

Each plugin is a .py file in this directory that:
  1. Subclasses ServicePlugin
  2. Sets  type_name = 'my-type'  on the class
  3. Exposes  PLUGIN = MyPlugin()  at module level

The proxy discovers plugins automatically via pkgutil.  On each AI request:
  - unload()     is called (synchronously) for every service that is NOT
                 the request target, so they release VRAM before we proceed.
  - get_models() is called on every service to assemble the /v1/models list.

Plugins run in-process; they should not raise from get_models() or unload()
(catch and log instead) so one broken service doesn't kill others.
"""


class ServicePlugin:
    """Base class for all vram-proxy service type plugins."""

    # Override in subclass: must match the 'type' value in config.yaml
    type_name: str = None

    def get_models(self, service: dict) -> list:
        """
        Return a list of OpenAI-format model objects for this service.

        Each item should at minimum have:
            { "id": "model-name", "object": "model", "owned_by": "..." }

        Return [] (never raise) if the service is unreachable or doesn't
        expose a model list.
        """
        raise NotImplementedError

    def unload(self, service: dict) -> None:
        """
        Synchronously unload all active models / free VRAM on this service.

        Must BLOCK until the service has released its GPU memory — the proxy
        will not forward the next request until this returns.

        Should be a no-op (with an optional log line) when nothing is loaded.
        Should not raise; log and return on failure.
        """
        raise NotImplementedError
