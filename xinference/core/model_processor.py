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

import asyncio
import platform
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from ..core import Model
from ..model.llm import LLMFamilyV1, LLMSpecV1
from .utils import log_async, log_sync

logger = getLogger(__name__)


class ModelProcessor:
    def __init__(self):
        self._model_uid_to_model: Dict[str, Model] = {}
        self._model_uid_to_model_spec: Dict[
            str, Tuple[LLMFamilyV1, LLMSpecV1, str]
        ] = {}

    def _check_model_is_valid(self, model_name):
        # baichuan-base and baichuan-chat depend on `cpm_kernels` module,
        # but `cpm_kernels` cannot run on Darwin system.
        if platform.system() == "Darwin":
            # TODO: there's no baichuan-base.
            if model_name in ["baichuan-base", "baichuan-chat"]:
                raise ValueError(f"{model_name} model can't run on Darwin system.")

    @staticmethod
    def _to_llm_description(
        llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
    ) -> Dict[str, Any]:
        return {
            "model_type": "LLM",
            "model_name": llm_family.model_name,
            "model_lang": llm_family.model_lang,
            "model_ability": llm_family.model_ability,
            "model_description": llm_family.model_description,
            "model_format": llm_spec.model_format,
            "model_size_in_billions": llm_spec.model_size_in_billions,
            "quantization": quantization,
            "revision": llm_spec.model_revision,
            "context_length": llm_family.context_length,
        }

    @log_sync(logger=logger)
    async def list_model_registrations(self, model_type: str) -> List[Dict[str, Any]]:
        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            ret = [
                {"model_name": f.model_name, "is_builtin": True}
                for f in BUILTIN_LLM_FAMILIES
            ]
            user_defined_llm_families = get_user_defined_llm_families()
            ret.extend(
                [
                    {"model_name": f.model_name, "is_builtin": False}
                    for f in user_defined_llm_families
                ]
            )

            return ret
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def get_model_registration(
        self, model_type: str, model_name: str
    ) -> Dict[str, Any]:
        if model_type == "LLM":
            from ..model.llm import BUILTIN_LLM_FAMILIES, get_user_defined_llm_families

            for f in BUILTIN_LLM_FAMILIES + get_user_defined_llm_families():
                if f.model_name == model_name:
                    return f

            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def register_model(self, model_type: str, model: str, persist: bool):
        # TODO: centralized model registrations
        if model_type == "LLM":
            from ..model.llm import LLMFamilyV1, register_llm

            llm_family = LLMFamilyV1.parse_raw(model)
            register_llm(llm_family, persist)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_sync(logger=logger)
    async def unregister_model(self, model_type: str, model_name: str):
        # TODO: centralized model registrations
        if model_type == "LLM":
            from ..model.llm import unregister_llm

            unregister_llm(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_async(logger=logger)
    async def launch_builtin_model(
        self,
        model_uid: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str],
        quantization: Optional[str],
        **kwargs,
    ) -> Model:
        assert model_uid not in self._model_uid_to_model
        self._check_model_is_valid(model_name)

        from ..model.llm import match_llm, match_llm_cls

        match_result = match_llm(
            model_name,
            model_format,
            model_size_in_billions,
            quantization,
            is_local_deployment=True,
        )
        if not match_result:
            raise ValueError(
                f"Model not found, name: {model_name}, format: {model_format},"
                f" size: {model_size_in_billions}, quantization: {quantization}"
            )
        llm_family, llm_spec, quantization = match_result
        assert quantization is not None

        from ..model.llm.llm_family import cache

        save_path = await asyncio.to_thread(cache, llm_family, llm_spec, quantization)

        llm_cls = match_llm_cls(llm_family, llm_spec)
        if not llm_cls:
            raise ValueError(
                f"Model not supported, name: {model_name}, format: {model_format},"
                f" size: {model_size_in_billions}, quantization: {quantization}"
            )

        model = llm_cls(
            model_uid, llm_family, llm_spec, quantization, save_path, kwargs
        )
        model_ref = Model(model=model)
        model_ref.load()
        self._model_uid_to_model[model_uid] = model_ref
        self._model_uid_to_model_spec[model_uid] = (llm_family, llm_spec, quantization)
        return model_ref

    @log_async(logger=logger)
    async def terminate_model(self, model_uid: str):
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        model_ref = self._model_uid_to_model[model_uid]

        del model_ref
        del self._model_uid_to_model[model_uid]
        del self._model_uid_to_model_spec[model_uid]

    @log_sync(logger=logger)
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        ret = {}
        for k, v in self._model_uid_to_model_spec.items():
            ret[k] = self._to_llm_description(v[0], v[1], v[2])
        return ret

    @log_sync(logger=logger)
    async def get_model(self, model_uid: str) -> Model:
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        return self._model_uid_to_model[model_uid]

    @log_sync(logger=logger)
    async def describe_model(self, model_uid: str) -> Dict[str, Any]:
        if model_uid not in self._model_uid_to_model:
            raise ValueError(f"Model not found in the model list, uid: {model_uid}")

        llm_family, llm_spec, quantization = self._model_uid_to_model_spec[model_uid]
        return self._to_llm_description(llm_family, llm_spec, quantization)
