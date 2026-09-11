import pytest
from aiohttp import web

from xinference_client import AsyncRESTfulClient


@pytest.mark.asyncio
async def test_list_models_and_explicit_close(monkeypatch):
    # Authentication discovery is synchronous; isolate it from the async server.
    monkeypatch.setattr(
        AsyncRESTfulClient, "_check_cluster_authenticated", lambda self: None
    )
    model = {"id": "test-model", "model_type": "LLM"}

    async def list_models(request):
        return web.json_response({"data": [model]})

    app = web.Application()
    app.router.add_get("/v1/models", list_models)
    runner = web.AppRunner(app)
    await runner.setup()
    client = None
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        client = AsyncRESTfulClient(f"http://127.0.0.1:{port}")
        session = client.session
        assert await client.list_models() == {"test-model": model}
        await client.close()
        assert session.closed
        assert client.session is None
        await client.close()
    finally:
        if client is not None:
            await client.close()
        await runner.cleanup()
