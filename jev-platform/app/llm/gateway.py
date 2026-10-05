"""
One door to every model call.

Why a gateway instead of passing ChatOpenAI around:
  * the JEV layer, executors and tools never import a provider SDK, so swapping
    OpenAI <-> Claude is one env var;
  * three roles (router / worker / judge) can point at different models, which is
    how you keep pre-routing and post-checking cheap;
  * tests replace this one class with a scripted fake and the whole graph runs
    offline.
"""
from __future__ import annotations

import logging
from typing import Literal, Sequence, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.config import Settings

log = logging.getLogger(__name__)

Role = Literal["router", "worker", "judge"]
T = TypeVar("T", bound=BaseModel)


class LLMGateway:
    def __init__(self, settings: Settings):
        self.s = settings
        self._cache: dict[str, BaseChatModel] = {}

    # ------------------------------------------------------------------ models
    def model_name(self, role: Role) -> str:
        p = self.s.llm_provider
        return getattr(self.s, f"{p}_{role}_model")

    def chat_model(self, role: Role) -> BaseChatModel:
        name = self.model_name(role)
        if name in self._cache:
            return self._cache[name]

        if self.s.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=name,
                temperature=0,
                api_key=self.s.openai_api_key,
                timeout=self.s.llm_timeout_s,
                max_retries=self.s.llm_max_retries,
            )
        else:
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model=name,
                temperature=0,
                api_key=self.s.anthropic_api_key,
                timeout=self.s.llm_timeout_s,
                max_retries=self.s.llm_max_retries,
                max_tokens=2048,
            )
        self._cache[name] = model
        return model

    # ------------------------------------------------------------------ calls
    async def complete(self, role: Role, system: str, user: str) -> str:
        msg = await self.chat_model(role).ainvoke([SystemMessage(system), HumanMessage(user)])
        return message_text(msg)

    async def structured(self, role: Role, system: str, user: str, schema: type[T]) -> T:
        model = self.chat_model(role).with_structured_output(schema)
        return await model.ainvoke([SystemMessage(system), HumanMessage(user)])

    async def invoke_with_tools(
        self, role: Role, messages: Sequence[BaseMessage], tools: Sequence[BaseTool]
    ) -> AIMessage:
        model = self.chat_model(role).bind_tools(list(tools))
        return await model.ainvoke(list(messages))


def message_text(msg: BaseMessage) -> str:
    """Claude can return a list of content blocks; flatten to plain text."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict))
