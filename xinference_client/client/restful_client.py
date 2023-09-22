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
import uuid
from typing import Any, Dict, List, Optional, Union

import requests

from ..handler.model_handler import (
    RESTfulChatglmCppChatModelHandle,
    RESTfulChatModelHandle,
    RESTfulEmbeddingModelHandle,
    RESTfulGenerateModelHandle,
    RESTfulModelHandle,
)


class RESTfulClient:
    def __init__(self, base_url):
        self.base_url = base_url

    @classmethod
    def _gen_model_uid(cls) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve the model specifications from the Server.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The collection of model specifications with their names on the server.

        """

        url = f"{self.base_url}/v1/models"

        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def launch_model(
        self,
        model_name: str,
        model_type: str = "LLM",
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        **kwargs,
    ) -> str:
        """
        Launch the model based on the parameters on the server via RESTful APIs.

        Parameters
        ----------
        model_name: str
            The name of model.
        model_type: str
            type of model.
        model_size_in_billions: Optional[int]
            The size (in billions) of the model.
        model_format: Optional[str]
            The format of the model.
        quantization: Optional[str]
            The quantization of model.
        replica: Optional[int]
            The replica of model, default is 1.
        n_gpu: Optional[Union[int, str]],
            The number of GPUs used by the model, default is "auto".
            ``n_gpu=None`` means cpu only, ``n_gpu=auto`` lets the system automatically determine the best number of GPUs to use.
        **kwargs:
            Any other parameters been specified.

        Returns
        -------
        str
            The unique model_uid for the launched model.

        """

        url = f"{self.base_url}/v1/models"

        model_uid = self._gen_model_uid()

        payload = {
            "model_uid": model_uid,
            "model_name": model_name,
            "model_type": model_type,
            "model_size_in_billions": model_size_in_billions,
            "model_format": model_format,
            "quantization": quantization,
            "replica": replica,
            "n_gpu": n_gpu,
        }

        for key, value in kwargs.items():
            payload[str(key)] = value

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to launch model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        model_uid = response_data["model_uid"]
        return model_uid

    def terminate_model(self, model_uid: str):
        """
        Terminate the specific model running on the server.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model we want.

        Raises
        ------
        RuntimeError
            Report failure to get the wanted model with given model_uid. Provide details of failure through error message.

        """

        url = f"{self.base_url}/v1/models/{model_uid}"

        response = requests.delete(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to terminate model, detail: {response.json()['detail']}"
            )

    def _get_supervisor_internal_address(self):
        url = f"{self.base_url}/v1/address"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get supervisor internal address")
        response_data = response.json()
        return response_data

    def get_model(self, model_uid: str) -> RESTfulModelHandle:
        """
        Launch the model based on the parameters on the server via RESTful APIs.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model.

        Returns
        -------
        ModelHandle
            The corresponding Model Handler based on the Model specified in the uid:
            "RESTfulChatglmCppChatModelHandle" -> provide handle to ChatGLM Model
            "RESTfulGenerateModelHandle" -> provide handle to basic generate Model. e.g. Baichuan.
            "RESTfulChatModelHandle" -> provide handle to chat Model. e.g. Baichuan-chat.

        Raises
        ------
        RuntimeError
            Report failure to get the wanted model with given model_uid. Provide details of failure through error message.

        """

        url = f"{self.base_url}/v1/models/{model_uid}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get the model description, detail: {response.json()['detail']}"
            )
        desc = response.json()

        if desc["model_type"] == "LLM":
            if desc["model_format"] == "ggmlv3" and "chatglm" in desc["model_name"]:
                return RESTfulChatglmCppChatModelHandle(model_uid, self.base_url)
            elif "chat" in desc["model_ability"]:
                return RESTfulChatModelHandle(model_uid, self.base_url)
            elif "generate" in desc["model_ability"]:
                return RESTfulGenerateModelHandle(model_uid, self.base_url)
            else:
                raise ValueError(f"Unrecognized model ability: {desc['model_ability']}")
        elif desc["model_type"] == "embedding":
            return RESTfulEmbeddingModelHandle(model_uid, self.base_url)
        else:
            raise ValueError(f"Unknown model type:{desc['model_type']}")

    def describe_model(self, model_uid: str):
        """
        Get model information via RESTful APIs.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - "model_type": str
               the type of the model determined by its function, e.g. "LLM" (Large Language Model)
            - "model_name": str
               the name of the specific LLM model family
            - "model_lang": List[str]
               the languages supported by the LLM model
            - "model_ability": List[str]
               the ability or capabilities of the LLM model
            - "model_description": str
               a detailed description of the LLM model
            - "model_format": str
               the format specification of the LLM model
            - "model_size_in_billions": int
               the size of the LLM model in billions
            - "quantization": str
               the quantization applied to the model
            - "revision": str
               the revision number of the LLM model specification
            - "context_length": int
               the maximum text length the LLM model can accommodate (include all input & output)

        Raises
        ------
        RuntimeError
            Report failure to get the wanted model with given model_uid. Provide details of failure through error message.

        """

        url = f"{self.base_url}/v1/models/{model_uid}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get the model description, detail: {response.json()['detail']}"
            )
        return response.json()

    def register_model(self, model_type: str, model: str, persist: bool):
        """
        Register a custom model.

        Parameters
        ----------
        model_type: str
            The type of model.
        model: str
            The model definition. (refer to: https://inference.readthedocs.io/en/latest/models/custom.html)
        persist: bool


        Raises
        ------
        RuntimeError
            Report failure to register the custom model. Provide details of failure through error message.
        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}"
        request_body = {"model": model, "persist": persist}
        response = requests.post(url, json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to register model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def unregister_model(self, model_type: str, model_name: str):
        """
        Unregister a custom model.

        Parameters
        ----------
        model_type: str
            The type of model.
        model_name: str
            The name of the model

        Raises
        ------
        RuntimeError
            Report failure to unregister the custom model. Provide details of failure through error message.
        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}/{model_name}"
        response = requests.delete(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to register model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def list_model_registrations(self, model_type: str) -> List[Dict[str, Any]]:
        """
        List models registered on the server.

        Parameters
        ----------
        model_type: str
            The type of the model.

        Returns
        -------
        List[Dict[str, Any]]
            The collection of registered models on the server.

        Raises
        ------
        RuntimeError
            Report failure to list model registration. Provide details of failure through error message.

        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model registration, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def get_model_registration(
        self, model_type: str, model_name: str
    ) -> Dict[str, Any]:
        """
        Get the model with the model type and model name registered on the server.

        Parameters
        ----------
        model_type: str
            The type of the model.

        model_name: str
            The name of the model.
        Returns
        -------
        List[Dict[str, Any]]
            The collection of registered models on the server.
        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}/{model_name}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model registration, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data
