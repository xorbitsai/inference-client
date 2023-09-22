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
from typing import Any, Dict, Iterator, List, Optional, Union

import requests

from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatglmCppGenerateConfig,
    Completion,
    CompletionChunk,
    Embedding,
    LlamaCppGenerateConfig,
    PytorchGenerateConfig,
)
from .utils import chat_streaming_response_iterator, streaming_response_iterator


class RESTfulModelHandle:
    """
    A sync model interface (for RESTful client) which provides type hints that makes it much easier to use xinference
    programmatically.
    """

    def __init__(self, model_uid: str, base_url: str):
        self._model_uid = model_uid
        self._base_url = base_url


class RESTfulEmbeddingModelHandle(RESTfulModelHandle):
    def create_embedding(self, input: Union[str, List[str]]) -> "Embedding":
        """
        Create an Embedding from user input via RESTful APIs.

        Parameters
        ----------
        input: Union[str, List[str]]
            Input text to embed, encoded as a string or array of tokens.
            To embed multiple inputs in a single request, pass an array of strings or array of token arrays.

        Returns
        -------
        Embedding
           The resulted Embedding vector that can be easily consumed by machine learning models and algorithms.

        Raises
        ------
        RuntimeError
            Report the failure of embeddings and provide the error message.

        """
        url = f"{self._base_url}/v1/embeddings"
        request_body = {"model": self._model_uid, "input": input}
        response = requests.post(url, json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to create the embeddings, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data


class RESTfulChatglmCppChatModelHandle(RESTfulEmbeddingModelHandle):
    def chat(
        self,
        prompt: str,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional["ChatglmCppGenerateConfig"] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        """
        Given a list of messages comprising a conversation, the ChatGLM model will return a response via RESTful APIs.

        Parameters
        ----------
        prompt: str
            The user's input.
        chat_history: Optional[List["ChatCompletionMessage"]]
            A list of messages comprising the conversation so far.
        generate_config: Optional["ChatglmCppGenerateConfig"]
            Additional configuration for ChatGLM chat generation.

        Returns
        -------
        Union["ChatCompletion", Iterator["ChatCompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["ChatCompletionChunk"].
            When stream is set to False, the function will return "ChatCompletion".

        Raises
        ------
        RuntimeError
            Report the failure to generate the chat from the server. Detailed information provided in error message.

        """

        url = f"{self._base_url}/v1/chat/completions"

        if chat_history is None:
            chat_history = []

        chat_history.append({"role": "user", "content": prompt})

        request_body: Dict[str, Any] = {
            "model": self._model_uid,
            "messages": chat_history,
        }

        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))
        response = requests.post(url, json=request_body, stream=stream)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate chat completion, detail: {response.json()['detail']}"
            )

        if stream:
            return chat_streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data


class RESTfulGenerateModelHandle(RESTfulEmbeddingModelHandle):
    def generate(
        self,
        prompt: str,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["Completion", Iterator["CompletionChunk"]]:
        """
        Creates a completion for the provided prompt and parameters via RESTful APIs.

        Parameters
        ----------
        prompt: str
            The user's message or user's input.
        generate_config: Optional[Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]]
            Additional configuration for the chat generation.
            "LlamaCppGenerateConfig" -> Configuration for ggml model
            "PytorchGenerateConfig" -> Configuration for pytorch model

        Returns
        -------
        Union["Completion", Iterator["CompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["CompletionChunk"].
            When stream is set to False, the function will return "Completion".

        Raises
        ------
        RuntimeError
            Fail to generate the completion from the server. Detailed information provided in error message.

        """

        url = f"{self._base_url}/v1/completions"

        request_body: Dict[str, Any] = {"model": self._model_uid, "prompt": prompt}
        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))

        response = requests.post(url, json=request_body, stream=stream)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate completion, detail: {response.json()['detail']}"
            )

        if stream:
            return streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data


class RESTfulChatModelHandle(RESTfulGenerateModelHandle):
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        """
        Given a list of messages comprising a conversation, the model will return a response via RESTful APIs.

        Parameters
        ----------
        prompt: str
            The user's input.
        system_prompt: Optional[str]
            The system context provide to Model prior to any chats.
        chat_history: Optional[List["ChatCompletionMessage"]]
            A list of messages comprising the conversation so far.
        generate_config: Optional[Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]]
            Additional configuration for the chat generation.
            "LlamaCppGenerateConfig" -> configuration for ggml model
            "PytorchGenerateConfig" -> configuration for pytorch model

        Returns
        -------
        Union["ChatCompletion", Iterator["ChatCompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["ChatCompletionChunk"].
            When stream is set to False, the function will return "ChatCompletion".

        Raises
        ------
        RuntimeError
            Report the failure to generate the chat from the server. Detailed information provided in error message.

        """

        url = f"{self._base_url}/v1/chat/completions"

        if chat_history is None:
            chat_history = []

        if chat_history and chat_history[0]["role"] == "system":
            if system_prompt is not None:
                chat_history[0]["content"] = system_prompt

        else:
            if system_prompt is not None:
                chat_history.insert(0, {"role": "system", "content": system_prompt})

        chat_history.append({"role": "user", "content": prompt})

        request_body: Dict[str, Any] = {
            "model": self._model_uid,
            "messages": chat_history,
        }
        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))
        response = requests.post(url, json=request_body, stream=stream)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate chat completion, detail: {response.json()['detail']}"
            )

        if stream:
            return chat_streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data
