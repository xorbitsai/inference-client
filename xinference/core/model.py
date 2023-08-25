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

from typing import TYPE_CHECKING, Iterator, List, Optional, Union

if TYPE_CHECKING:
    from ..model.llm.core import LLM
    from ..types import ChatCompletionChunk, CompletionChunk

import logging

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, model: "LLM"):
        super().__init__()
        self._model = model
        self._generator: Optional[Iterator] = None

    def load(self):
        self._model.load()

    async def generate(self, prompt: str, *args, **kwargs):
        if not hasattr(self._model, "generate"):
            raise AttributeError(f"Model {self._model.model_spec} is not for generate.")

        return getattr(self._model, "generate")(prompt, *args, **kwargs)

    async def chat(self, prompt: str, *args, **kwargs):
        if not hasattr(self._model, "chat"):
            raise AttributeError(f"Model {self._model.model_spec} is not for chat.")

        return getattr(self._model, "chat")(prompt, *args, **kwargs)

    async def create_embedding(self, input: Union[str, List[str]], *args, **kwargs):
        if not hasattr(self._model, "create_embedding"):
            raise AttributeError(
                f"Model {self._model.model_spec} is not for creating embedding."
            )

        return getattr(self._model, "create_embedding")(input, *args, **kwargs)

    async def next(self) -> Union["ChatCompletionChunk", "CompletionChunk"]:
        try:
            assert self._generator is not None
            return next(self._generator)
        except StopIteration:
            self._generator = None
            raise Exception("StopIteration")
