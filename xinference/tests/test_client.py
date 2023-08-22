# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from ..client import ChatModelHandle, Client, RESTfulChatModelHandle, RESTfulClient


def test_client(setup):
    endpoint, _ = setup
    client = Client(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="orca", model_size_in_billions=3, quantization="q4_0"
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, ChatModelHandle)

    completion = model.chat("write a poem.")
    assert "content" in completion["choices"][0]["message"]

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="orca",
        model_size_in_billions=3,
        quantization="q4_0",
    )

    model = client.get_model(model_uid=model_uid)

    embedding_res = model.create_embedding("The food was delicious and the waiter...")
    assert "embedding" in embedding_res["data"][0]

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0


def test_client_custom_model(setup):
    endpoint, _ = setup
    client = Client(endpoint)

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
