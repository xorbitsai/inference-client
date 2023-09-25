import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio

from ...handler.model_handler import RESTfulChatModelHandle, RESTfulEmbeddingModelHandle
from ..restful_client import RESTfulClient


@pytest_asyncio.fixture
async def setup():
    import xoscar as xo
    from xinference.deploy.supervisor import start_supervisor_components
    from xinference.deploy.utils import create_worker_actor_pool
    from xinference.deploy.worker import start_worker_components

    pool = await create_worker_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}"
    )
    print(f"Pool running on localhost:{pool.external_address}")

    endpoint = await start_supervisor_components(
        pool.external_address, "127.0.0.1", xo.utils.get_next_port()
    )
    await start_worker_components(
        address=pool.external_address,
        supervisor_address=pool.external_address,
        main_pool=pool,
    )

    # wait for the api.
    time.sleep(3)
    async with pool:
        yield endpoint, pool.external_address


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
def test_RESTful_client(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="orca", model_size_in_billions=3, quantization="q4_0"
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulChatModelHandle)

    with pytest.raises(RuntimeError):
        model = client.get_model(model_uid="test")
        assert isinstance(model, RESTfulChatModelHandle)

    with pytest.raises(RuntimeError):
        completion = model.generate({"max_tokens": 64})

    completion = model.generate("Once upon a time, there was a very old computer")
    assert "text" in completion["choices"][0]

    completion = model.generate(
        "Once upon a time, there was a very old computer", {"max_tokens": 64}
    )
    assert "text" in completion["choices"][0]

    streaming_response = model.generate(
        "Once upon a time, there was a very old computer",
        {"max_tokens": 64, "stream": True},
    )

    for chunk in streaming_response:
        assert "text" in chunk["choices"][0]

    with pytest.raises(RuntimeError):
        completion = model.chat({"max_tokens": 64})

    completion = model.chat("What is the capital of France?")
    assert "content" in completion["choices"][0]["message"]

    streaming_response = model.chat(
        prompt="What is the capital of France?", generate_config={"stream": True}
    )

    for chunk in streaming_response:
        assert "content" or "role" in chunk["choices"][0]["delta"]

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="tiny-llama",
        model_size_in_billions=1,
        model_format="ggufv2",
        quantization="Q2_K",
    )
    assert len(client.list_models()) == 1

    # Test concurrent chat is OK.
    def _check(stream=False):
        model = client.get_model(model_uid=model_uid)
        completion = model.generate(
            "AI is going to", generate_config={"stream": stream, "max_tokens": 5}
        )
        if stream:
            for chunk in completion:
                assert "text" in chunk["choices"][0]
                assert len(chunk["choices"][0]["text"]) > 0
        else:
            assert "text" in completion["choices"][0]
            assert len(completion["choices"][0]["text"]) > 0

    for stream in [True, False]:
        results = []
        with ThreadPoolExecutor() as executor:
            for _ in range(3):
                r = executor.submit(_check, stream=stream)
                results.append(r)
        for r in results:
            r.result()

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    with pytest.raises(RuntimeError):
        client.terminate_model(model_uid=model_uid)

    model_uid2 = client.launch_model(
        model_name="orca",
        model_size_in_billions=3,
        quantization="q4_0",
    )

    model2 = client.get_model(model_uid=model_uid2)

    embedding_res = model2.create_embedding("The food was delicious and the waiter...")
    assert "embedding" in embedding_res["data"][0]

    client.terminate_model(model_uid=model_uid2)
    assert len(client.list_models()) == 0


def test_RESTful_client_for_embedding(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(model_name="gte-base", model_type="embedding")
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulEmbeddingModelHandle)

    completion = model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 768

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0


def test_RESTful_client_custom_model(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)

    model_regs = client.list_model_registrations(model_type="LLM")
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

    model = """{
  "version": 1,
  "context_length":2048,
  "model_name": "custom_model",
  "model_lang": [
    "en", "zh"
  ],
  "model_ability": [
    "embed",
    "chat"
  ],
  "model_specs": [
    {
      "model_format": "pytorch",
      "model_size_in_billions": 7,
      "quantizations": [
        "4-bit",
        "8-bit",
        "none"
      ],
      "model_id": "ziqingyang/chinese-alpaca-2-7b"
    }
  ],
  "prompt_style": {
    "style_name": "ADD_COLON_SINGLE",
    "system_prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    "roles": [
      "Instruction",
      "Response"
    ],
    "intra_message_sep": "\\n\\n### "
  }
}"""
    client.register_model(model_type="LLM", model=model, persist=False)

    data = client.get_model_registration(model_type="LLM", model_name="custom_model")
    assert "custom_model" in data["model_name"]

    new_model_regs = client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is not None

    client.unregister_model(model_type="LLM", model_name="custom_model")
    new_model_regs = client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs)
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is None
