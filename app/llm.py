"""Async Azure OpenAI streaming client."""

from typing import AsyncGenerator

from openai import AsyncAzureOpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)

SYSTEM_PROMPT = (
    "You are a natural, concise conversational voice assistant. "
    "Limit answers to 1-2 spoken sentences."
)

_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


async def stream_llm_response(
    history: list[dict],
) -> AsyncGenerator[str, None]:
    """Stream text token deltas from Azure OpenAI."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    stream = await _client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
