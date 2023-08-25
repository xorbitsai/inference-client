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

import configparser
import logging
import os
import sys
from typing import List, Optional

import click
from xoscar.utils import get_next_port

from .. import __version__
from ..client import (
    RESTfulChatglmCppChatModelHandle,
    RESTfulChatModelHandle,
    RESTfulClient,
    RESTfulGenerateModelHandle,
)
from ..constants import (
    XINFERENCE_DEFAULT_ENDPOINT_PORT,
    XINFERENCE_DEFAULT_LOCAL_HOST,
    XINFERENCE_ENV_ENDPOINT,
)
from ..types import ChatCompletionMessage

try:
    # provide elaborate line editing and history features.
    # https://docs.python.org/3/library/functions.html#input
    import readline  # noqa: F401
except ImportError:
    pass


def get_config_string(log_level: str) -> str:
    return f"""[loggers]
keys=root

[handlers]
keys=stream_handler

[formatters]
keys=formatter

[logger_root]
level={log_level.upper()}
handlers=stream_handler

[handler_stream_handler]
class=StreamHandler
formatter=formatter
level={log_level.upper()}
args=(sys.stderr,)

[formatter_formatter]
format=%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s
"""


def get_endpoint(endpoint: Optional[str]) -> str:
    # user didn't specify the endpoint.
    if endpoint is None:
        if XINFERENCE_ENV_ENDPOINT in os.environ:
            return os.environ[XINFERENCE_ENV_ENDPOINT]
        else:
            default_endpoint = f"http://{XINFERENCE_DEFAULT_LOCAL_HOST}:{XINFERENCE_DEFAULT_ENDPOINT_PORT}"
            return default_endpoint
    else:
        return endpoint


@click.group(invoke_without_command=True, name="xinference-client")
@click.pass_context
@click.version_option(__version__, "--version", "-v")
@click.option("--log-level", default="INFO", type=str)
@click.option("--host", "-H", default=XINFERENCE_DEFAULT_LOCAL_HOST, type=str)
@click.option("--port", "-p", default=XINFERENCE_DEFAULT_ENDPOINT_PORT, type=int)
def cli(
    ctx,
    log_level: str,
    host: str,
    port: str,
):
    if ctx.invoked_subcommand is None:
        from .server import main

        logging_conf = configparser.RawConfigParser()
        logger_config_string = get_config_string(log_level)
        logging_conf.read_string(logger_config_string)
        logging.config.fileConfig(logging_conf)  # type: ignore

        address = f"{host}:{get_next_port()}"

        main(address=address, host=host, port=port, logging_conf=logging_conf)


@cli.command("register")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-type", "-t", default="LLM", type=str)
@click.option("--file", "-f", type=str)
@click.option("--persist", "-p", is_flag=True)
def register_model(
    endpoint: Optional[str],
    model_type: str,
    file: str,
    persist: bool,
):
    endpoint = get_endpoint(endpoint)
    with open(file) as fd:
        model = fd.read()

    client = RESTfulClient(base_url=endpoint)
    client.register_model(
        model_type=model_type,
        model=model,
        persist=persist,
    )


@cli.command("unregister")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-type", "-t", default="LLM", type=str)
@click.option("--model-name", "-n", type=str)
def unregister_model(
    endpoint: Optional[str],
    model_type: str,
    model_name: str,
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    client.unregister_model(
        model_type=model_type,
        model_name=model_name,
    )


@cli.command("registrations")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-type", "-t", default="LLM", type=str)
def list_model_registrations(
    endpoint: Optional[str],
    model_type: str,
):
    from tabulate import tabulate

    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    registrations = client.list_model_registrations(model_type=model_type)

    table = []
    for registration in registrations:
        model_name = registration["model_name"]
        model_family = client.get_model_registration(model_type, model_name)
        table.append(
            [
                model_type,
                model_family["model_name"],
                model_family["model_lang"],
                model_family["model_ability"],
                registration["is_builtin"],
            ]
        )
    print(
        tabulate(table, headers=["Type", "Name", "Language", "Ability", "Is-built-in"]),
        file=sys.stderr,
    )


@cli.command("launch")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-name", "-n", type=str)
@click.option("--size-in-billions", "-s", default=None, type=int)
@click.option("--model-format", "-f", default=None, type=str)
@click.option("--quantization", "-q", default=None, type=str)
def model_launch(
    endpoint: Optional[str],
    model_name: str,
    size_in_billions: int,
    model_format: str,
    quantization: str,
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    model_uid = client.launch_model(
        model_name=model_name,
        model_size_in_billions=size_in_billions,
        model_format=model_format,
        quantization=quantization,
    )

    print(f"Model uid: {model_uid}", file=sys.stderr)


@cli.command("list")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
def model_list(endpoint: Optional[str]):
    from tabulate import tabulate

    endpoint = get_endpoint(endpoint)
    client = RESTfulClient(base_url=endpoint)

    table = []
    models = client.list_models()
    for model_uid, model_spec in models.items():
        table.append(
            [
                model_uid,
                model_spec["model_type"],
                model_spec["model_name"],
                model_spec["model_format"],
                model_spec["model_size_in_billions"],
                model_spec["quantization"],
            ]
        )
    print(
        tabulate(
            table,
            headers=[
                "UID",
                "Type",
                "Name",
                "Format",
                "Size (in billions)",
                "Quantization",
            ],
        ),
        file=sys.stderr,
    )


@cli.command("terminate")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-uid", type=str)
def model_terminate(
    endpoint: Optional[str],
    model_uid: str,
):
    endpoint = get_endpoint(endpoint)

    client = RESTfulClient(base_url=endpoint)
    client.terminate_model(model_uid=model_uid)


@cli.command("generate")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-uid", type=str)
@click.option("--max_tokens", default=256, type=int)
def model_generate(
    endpoint: Optional[str],
    model_uid: str,
    max_tokens: int,
):
    endpoint = get_endpoint(endpoint)

    restful_client = RESTfulClient(base_url=endpoint)
    restful_model = restful_client.get_model(model_uid=model_uid)
    if not isinstance(
        restful_model, (RESTfulChatModelHandle, RESTfulGenerateModelHandle)
    ):
        raise ValueError(f"model {model_uid} has no generate method")

    while True:
        prompt = input("User: ")
        if prompt == "":
            break
        print(f"Assistant: {prompt}", end="", file=sys.stdout)
        response = restful_model.generate(
            prompt=prompt,
            generate_config={"max_tokens": max_tokens},
        )
        if not isinstance(response, dict):
            raise ValueError("generate result is not valid")
        print(f"{response['choices'][0]['text']}\n", file=sys.stdout)


@cli.command("chat")
@click.option(
    "--endpoint",
    "-e",
    type=str,
)
@click.option("--model-uid", type=str)
@click.option("--max_tokens", default=256, type=int)
def model_chat(
    endpoint: Optional[str],
    model_uid: str,
    max_tokens: int,
):
    # TODO: chat model roles may not be user and assistant.
    endpoint = get_endpoint(endpoint)
    chat_history: "List[ChatCompletionMessage]" = []

    restful_client = RESTfulClient(base_url=endpoint)
    restful_model = restful_client.get_model(model_uid=model_uid)
    if not isinstance(
        restful_model, (RESTfulChatModelHandle, RESTfulChatglmCppChatModelHandle)
    ):
        raise ValueError(f"model {model_uid} has no chat method")

    while True:
        prompt = input("User: ")
        if prompt == "":
            break
        chat_history.append(ChatCompletionMessage(role="user", content=prompt))
        print("Assistant: ", end="", file=sys.stdout)
        response = restful_model.chat(
            prompt=prompt,
            chat_history=chat_history,
            generate_config={"max_tokens": max_tokens},
        )
        if not isinstance(response, dict):
            raise ValueError("chat result is not valid")
        response_content = response["choices"][0]["message"]["content"]
        print(f"{response_content}\n", file=sys.stdout)
        chat_history.append(
            ChatCompletionMessage(role="assistant", content=response_content)
        )


if __name__ == "__main__":
    cli()
