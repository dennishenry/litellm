import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal

from fastapi import HTTPException, Request
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.types.mcp_server.mcp_server_manager import MCPServer

TOKEN_BUCKET: Final = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local rate = tonumber(ARGV[1]) / 60
local burst = tonumber(ARGV[2])
local old = redis.call('HMGET', KEYS[1], 'tokens', 'updated')
local tokens = math.min(burst, tonumber(old[1] or burst) + math.max(0, now - tonumber(old[2] or now)) * rate)
local retry = 0
if tokens >= 1 then tokens = tokens - 1
else retry = math.max(1, math.ceil((1 - tokens) / rate)) end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated', now)
redis.call('EXPIRE', KEYS[1], math.max(1, math.ceil(burst / rate)))
return retry
"""
_RETRY: Final = TypeAdapter(int)
_LOGGER: Final = verbose_proxy_logger.getChild("oauth")
_LOGGER.setLevel(logging.INFO)
_CLIENT_VERSION: Final = re.compile(
    r"\b(OpenWebUI|open-webui|Claude|claude-code|Cursor|codex|python-httpx)/([0-9]{1,4}(?:\.[0-9]{1,4}){1,3})(?:\b|$)"
)


@dataclass(frozen=True, slots=True)
class OAuthTokenLimits:
    rpm: int = 120
    burst: int = 30

    def __post_init__(self) -> None:
        if self.rpm <= 0 or self.burst <= 0:
            raise ValueError("OAuth token limits must be positive")


@lru_cache(maxsize=64)
def _operational_event(event: Literal["MCP_OAUTH_THROTTLED", "MCP_OAUTH_LIMITER_FAILURE"], minute: int) -> None:
    _LOGGER.warning("%s", event)


def challenge_diagnostic(request: Request, origin: Literal["gateway", "upstream", "missing_credential"]) -> None:
    if random.random() >= 0.01:
        return
    version: Final = _CLIENT_VERSION.search(request.headers.get("user-agent", "")[:256])
    route: Final = getattr(request.scope.get("route"), "path", "/mcp")
    _LOGGER.info(
        "MCP_OAUTH_CHALLENGE origin=%s route=%s method=%s credential_present=%s client_version=%s",
        origin,
        route,
        request.method if request.method in {"GET", "POST", "DELETE", "OPTIONS"} else "other",
        bool(request.headers.get("authorization") or request.headers.get("x-litellm-api-key")),
        version.group(0) if version else "unknown",
    )


async def token_retry_after(
    cache: RedisCache | None, server_id: str, client_id: str, client_address: str, limits: OAuthTokenLimits
) -> int:
    if cache is None:
        _operational_event("MCP_OAUTH_LIMITER_FAILURE", int(time.time() // 60))
        return 0
    key: Final = (
        "mcp:oauth:token:"
        + hashlib.sha256(json.dumps((server_id, client_id, client_address), separators=(",", ":")).encode()).hexdigest()
    )
    try:
        result: Final = await cache.async_register_script(TOKEN_BUCKET)(keys=(key,), args=(limits.rpm, limits.burst))
        return _RETRY.validate_python(result, strict=True)
    except Exception:  # noqa: BLE001  # limiter failures preserve existing OAuth behavior and alert without exception text
        _operational_event("MCP_OAUTH_LIMITER_FAILURE", int(time.time() // 60))
        return 0


async def enforce_token_limit(request: Request, server: MCPServer, client_id: str) -> None:
    if os.getenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_ENABLED", "false").lower() != "true":
        return
    if not (server.is_oauth_delegate or server.delegate_auth_to_upstream):
        return
    configured: Final = frozenset(
        name.strip().casefold()
        for name in os.getenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_SERVERS", "zoom,slack,atlassian").split(",")
        if name.strip()
    )
    if not configured.intersection(
        name.casefold() for name in (server.server_id, server.server_name, server.alias, server.name) if name
    ):
        return
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # initialized during proxy startup
        general_settings,
        redis_usage_cache,
    )

    limits: Final = OAuthTokenLimits(
        rpm=int(os.getenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_RPM", "120")),
        burst=int(os.getenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_BURST", "30")),
    )
    # The legacy leftmost-XFF path permits client-supplied prefixes. Require a
    # configured hop count for forwarded limiter keys; otherwise use the peer.
    address: Final = (
        IPAddressUtils.get_mcp_client_ip(request)
        if general_settings.get("mcp_xff_num_trusted_hops") is not None
        else (request.client.host if request.client else None)
    )
    retry: Final = await token_retry_after(
        redis_usage_cache, server.server_id, server.client_id or client_id, address or "unknown", limits
    )
    if retry:
        _operational_event("MCP_OAUTH_THROTTLED", int(time.time() // 60))
        raise HTTPException(429, "OAuth token request rate exceeded", headers={"Retry-After": str(retry)})
