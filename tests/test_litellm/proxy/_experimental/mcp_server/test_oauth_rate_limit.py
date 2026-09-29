import asyncio
import os
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server import oauth_rate_limit as limiter
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.mark.asyncio
async def test_two_replicas_share_burst_and_recover_after_cooldown():
    port = os.getenv("TEST_REDIS_PORT")
    if port is None:
        pytest.skip("Set TEST_REDIS_PORT to a disposable Redis or Valkey")
    namespace = "oauth-test-" + uuid4().hex
    replicas = tuple(RedisCache(host="127.0.0.1", port=int(port), namespace=namespace) for _ in range(2))
    limits = limiter.OAuthTokenLimits(rpm=120, burst=30)
    results = await asyncio.gather(
        *(limiter.token_retry_after(replicas[i % 2], "slack", "client", "192.0.2.1", limits) for i in range(60))
    )
    assert results.count(0) == 30
    assert set(results) == {0, 1}
    for server, client, address in (
        ("zoom", "client", "192.0.2.1"),
        ("slack", "other", "192.0.2.1"),
        ("slack", "client", "192.0.2.2"),
    ):
        assert await limiter.token_retry_after(replicas[1], server, client, address, limits) == 0
    keys = list(replicas[0].redis_client.scan_iter(match=namespace + "*"))
    assert len(keys) == 4
    assert all(0 < replicas[0].redis_client.ttl(key) <= 15 for key in keys)
    await asyncio.sleep(1.05)
    assert await limiter.token_retry_after(replicas[1], "slack", "client", "192.0.2.1", limits) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_exists", [False, True])
async def test_store_failure_preserves_oauth_and_alerts_without_credentials(cache_exists, caplog):
    cache = MagicMock(spec=RedisCache) if cache_exists else None
    if cache is not None:
        cache.async_register_script.return_value = AsyncMock(side_effect=OSError("redis://private-secret"))
    limiter._operational_event.cache_clear()
    assert await limiter.token_retry_after(cache, "slack", "client", "192.0.2.1", limiter.OAuthTokenLimits()) == 0
    assert "MCP_OAUTH_LIMITER_FAILURE" in caplog.text
    assert "private-secret" not in caplog.text


@pytest.mark.asyncio
async def test_exhausted_bucket_rejects_before_upstream_discovery(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server import discoverable_endpoints as endpoints

    monkeypatch.setenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_ENABLED", "true")
    cache = MagicMock(spec=RedisCache)
    cache.async_register_script.return_value = AsyncMock(return_value=2)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache)
    discovery = AsyncMock(side_effect=AssertionError("must not contact upstream"))
    monkeypatch.setattr(endpoints, "_server_with_oauth_endpoints", discovery)
    request = Request(
        {"type": "http", "method": "POST", "path": "/slack/token", "headers": [], "client": ("192.0.2.1", 1234)}
    )
    server = MCPServer(
        name="test", transport="http", server_id="slack", server_name="slack", auth_type=MCPAuth.oauth_delegate
    )
    with pytest.raises(HTTPException) as error:
        await endpoints.exchange_token_with_server(request, server, "refresh_token", None, None, "client", None, None)
    assert error.value.status_code == 429
    assert error.value.headers == {"Retry-After": "2"}
    discovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_delegated_and_unconfigured_servers_are_not_throttled(monkeypatch):
    monkeypatch.setenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_ENABLED", "true")
    check = AsyncMock(side_effect=AssertionError("not a delegated target"))
    monkeypatch.setattr(limiter, "token_retry_after", check)
    request = Request({"type": "http", "method": "POST", "path": "/token", "headers": []})
    await limiter.enforce_token_limit(
        request, MCPServer(name="test", transport="http", server_id="slack", auth_type=MCPAuth.oauth2), "client"
    )
    await limiter.enforce_token_limit(
        request, MCPServer(name="test", transport="http", server_id="other", auth_type=MCPAuth.oauth_delegate), "client"
    )
    check.assert_not_awaited()


def test_challenge_diagnostics_never_record_credentials_or_raw_paths(monkeypatch, caplog):
    monkeypatch.setattr(limiter.random, "random", lambda: 0)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/private-secret",
            "headers": [
                (b"authorization", b"Bearer private-secret"),
                (b"user-agent", b"OpenWebUI/1.2.3 private-secret"),
            ],
        }
    )
    limiter.challenge_diagnostic(request, "upstream")
    assert (
        "origin=upstream route=/mcp method=POST credential_present=True client_version=OpenWebUI/1.2.3" in caplog.text
    )
    assert "private-secret" not in caplog.text


@pytest.mark.asyncio
async def test_untrusted_forwarded_address_does_not_split_buckets(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_ENABLED", "true")
    monkeypatch.setattr(
        proxy_server, "general_settings", {"use_x_forwarded_for": True, "mcp_trusted_proxy_ranges": ["127.0.0.0/8"]}
    )
    check = AsyncMock(return_value=0)
    monkeypatch.setattr(limiter, "token_retry_after", check)
    server = MCPServer(name="test", transport="http", server_id="zoom", auth_type=MCPAuth.oauth_delegate)
    for spoof in (b"192.0.2.2", b"192.0.2.3"):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/zoom/token",
                "headers": [(b"x-forwarded-for", spoof)],
                "client": ("192.0.2.1", 1234),
            }
        )
        await limiter.enforce_token_limit(request, server, "client")
    assert check.await_args_list[0].args[3] == check.await_args_list[1].args[3] == "192.0.2.1"


@pytest.mark.asyncio
async def test_forwarded_address_requires_hop_count_and_uses_actual_oauth_client(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_MCP_OAUTH_TOKEN_LIMIT_ENABLED", "true")
    settings = {"use_x_forwarded_for": True, "mcp_trusted_proxy_ranges": ["127.0.0.0/8"]}
    monkeypatch.setattr(proxy_server, "general_settings", settings)
    check = AsyncMock(return_value=0)
    monkeypatch.setattr(limiter, "token_retry_after", check)
    server = MCPServer(
        name="Zoom",
        transport="http",
        server_id="server-uuid",
        auth_type=MCPAuth.oauth_delegate,
        client_id="registered-client",
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/zoom/token",
            "headers": [(b"x-forwarded-for", b"192.0.2.99, 192.0.2.1")],
            "client": ("127.0.0.1", 1234),
        }
    )
    await limiter.enforce_token_limit(request, server, "arbitrary-client")
    assert check.await_args.args[2:4] == ("registered-client", "127.0.0.1")
    monkeypatch.setattr(proxy_server, "general_settings", {**settings, "mcp_xff_num_trusted_hops": 1})
    await limiter.enforce_token_limit(request, server, "arbitrary-client")
    assert check.await_args.args[2:4] == ("registered-client", "192.0.2.1")
