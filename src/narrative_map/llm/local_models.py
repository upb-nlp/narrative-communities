"""
Local LLM and embedding helpers for GraphRAG pipelines.

Uses local vLLM (OpenAI-compatible) for completions and a local embedding API.
"""

import logging
import re
import aiohttp
import requests
import numpy as np

from narrative_map.config import EMBEDDING_API_URL, LLM_API_URL, LLM_MODEL_NAME

logger = logging.getLogger(__name__)


async def local_embed(texts):
    """
    Async embedding function using local embedding API.

    Args:
        texts: String or list of strings to embed

    Returns:
        List of embedding vectors
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        payload = {"input": texts}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                EMBEDDING_API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    return [item["embedding"] for item in result["data"]]
                else:
                    logger.error(f"Embedding API failed: {response.status}")
                    raise Exception(
                        f"Embedding API failed with status {response.status}"
                    )

    except Exception as e:
        logger.error(f"Embedding API error: {e}")
        raise


def local_embed_sync(texts):
    """
    Synchronous embedding function using local embedding API.
    For use with BERTopic and other non-async libraries.

    Args:
        texts: String or list of strings to embed

    Returns:
        numpy array of embedding vectors
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        response = requests.post(
            EMBEDDING_API_URL,
            headers={"Content-Type": "application/json"},
            json={"input": texts},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)
        else:
            error_msg = f"Embedding API failed with status {response.status_code}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    except requests.exceptions.RequestException as e:
        error_msg = f"Embedding API connection error: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def batched_embed(texts, batch_size=50):
    """
    Embed texts in batches to avoid timeout.
    This wrapper is used by BERTopic's embedding_model parameter.
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = local_embed_sync(batch)
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


async def local_llm_complete(
    prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    """
    Async LLM completion using the local vLLM Chat Completions API.

    Args:
        prompt: The main prompt text
        system_prompt: Optional system prompt
        history_messages: Optional list of previous messages
        **kwargs: Additional parameters (api_url, model_name, max_tokens, temperature)

    Returns:
        Generated text response
    """
    if history_messages is None:
        history_messages = []

    api_url = kwargs.get("api_url", LLM_API_URL)
    model_name = kwargs.get("model_name", LLM_MODEL_NAME)
    max_tokens = kwargs.get("max_tokens", 2048)
    # Lower temperature for more consistent structured output
    temperature = kwargs.get("temperature", 0.3)

    try:
        # Build messages array for chat completions API
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history messages
        for msg in history_messages:
            messages.append(msg)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"LLM API URL: {api_url}")
        logger.debug(f"LLM Model: {model_name}")
        logger.debug(f"Messages: {len(messages)} messages")

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["</s>", "<|end|>"],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    raw_content = result["choices"][0]["message"].get("content")
                    if raw_content is None:
                        raise ValueError(
                            "LLM API returned None for content. "
                            "Check vLLM logs and model output."
                        )
                    content = raw_content.strip()
                    # Check if the response looks like an error
                    if content.startswith("Error:") or "Error:" in content:
                        logger.error(f"LLM returned error response: {content[:200]}")
                        raise Exception(f"LLM API returned error: {content}")
                    return content
                else:
                    error_text = await response.text()
                    logger.error(
                        f"LLM API request failed with status {response.status}: {error_text[:500]}"
                    )
                    raise Exception(
                        f"LLM API request failed with status {response.status}: {error_text[:200]}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"LLM API connection error: {e}")
        raise Exception(f"LLM API connection error: {e}")
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        raise


def get_local_dspy_lm(model_name: str | None = None):
    """
    Get a dspy Language Model wrapper for the local LLM API.
    This allows using dspy with the local vLLM endpoint that follows OpenAI format.
    """
    import asyncio
    import dspy

    # Always use custom wrapper for better control and compatibility
    # The dspy.OpenAI class may have API changes between versions
    logger.info("Using custom DSPy LM wrapper for local vLLM API")

    # Create a custom LM wrapper
    class LocalDSPyLM(dspy.LM):
        """DSPy Language Model wrapper for local vLLM API"""

        def __init__(self):
            super().__init__(model="local")
            self.api_url = LLM_API_URL
            self.model_name = model_name or LLM_MODEL_NAME

        def __call__(self, prompt=None, messages=None, **kwargs):
            """Primary interface for dspy - handles prompt formatting"""
            # Handle both old and new DSPy calling conventions
            if messages is not None:
                return self.basic_request(messages, **kwargs)
            elif prompt is not None:
                return self.basic_request(prompt, **kwargs)
            else:
                raise ValueError("Either prompt or messages must be provided")

        def request(self, prompt=None, messages=None, **kwargs):
            """Request interface that handles dspy's prompt format"""
            if messages is not None:
                return self.basic_request(messages, **kwargs)
            return self.basic_request(prompt, **kwargs)

        def basic_request(self, prompt, **kwargs):
            """Basic request interface for dspy"""
            # dspy may pass prompts in different formats
            # Handle both string prompts and message arrays
            messages = None
            system_prompt = None
            user_prompt = ""

            if isinstance(prompt, str):
                # Simple string prompt
                user_prompt = prompt
            elif isinstance(prompt, list):
                # List of messages (dspy format)
                messages = prompt
                # Extract system and user prompts
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "system":
                            system_prompt = content
                        elif role == "user":
                            user_prompt = content
                    elif isinstance(msg, str):
                        # Sometimes messages are just strings
                        user_prompt = msg
            else:
                # Try to convert to string
                user_prompt = str(prompt)

            logger.debug(
                f"DSPy LM request - system_prompt: {system_prompt[:50] if system_prompt else None}, user_prompt: {user_prompt[:100] if user_prompt else None}"
            )

            # Use synchronous requests to avoid event loop conflicts
            # DSPy expects synchronous behavior, and we may be called from async context
            try:
                # Build messages for API call
                api_messages = []
                if system_prompt:
                    api_messages.append({"role": "system", "content": system_prompt})
                if user_prompt:
                    api_messages.append({"role": "user", "content": user_prompt})

                # Model context is typically 16k total; input + completion must fit. Use 8k for output
                # so input (~3k) + completion (~8k) stays under limit.
                payload = {
                    "model": self.model_name,
                    "messages": api_messages,
                    "max_tokens": kwargs.get("max_tokens", 8192),
                    "temperature": kwargs.get("temperature", 0.2),
                    "top_p": 0.9,
                    "stop": ["</s>", "<|end|>"],
                }

                # Use synchronous requests instead of async to avoid event loop issues
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=600,
                )

                if response.status_code == 200:
                    result = response.json()
                    msg = result["choices"][0]["message"]
                    raw_content = msg.get("content")
                    # vLLM reasoning models (Qwen3, DeepSeek-R1, etc.) put thinking in
                    # "reasoning" and final answer in "content". Sometimes content is
                    # None (e.g. max_tokens hit during reasoning). Use reasoning as
                    # fallback when content is empty.
                    if raw_content is None or (isinstance(raw_content, str) and not raw_content.strip()):
                        raw_content = msg.get("reasoning") or msg.get("reasoning_content")
                        if raw_content:
                            logger.warning(
                                "DSPy LM: content was None/empty, using reasoning field as fallback. "
                                "Consider increasing max_tokens so the model can finish."
                            )
                    if raw_content is None or (isinstance(raw_content, str) and not raw_content.strip()):
                        logger.error(
                            "DSPy LM returned None/empty for both content and reasoning. "
                            "Message keys: %s",
                            list(msg.keys()) if msg else None,
                        )
                        raise ValueError(
                            "LLM API returned None for content. "
                            "The model may have refused, timed out, or returned empty. "
                            "Try increasing max_tokens or reducing context length."
                        )
                    content = raw_content.strip() if isinstance(raw_content, str) else str(raw_content)
                    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
                    logger.debug(
                        f"DSPy LM response: {content[:100] if content else None}"
                    )
                    # Return in dspy expected format
                    return [{"text": content}]
                else:
                    error_text = response.text[:500]
                    logger.error(
                        f"DSPy LM API request failed with status {response.status_code}: {error_text}"
                    )
                    raise Exception(
                        f"DSPy LM API request failed with status {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                logger.error(f"DSPy LM connection error: {e}")
                raise Exception(f"DSPy LM connection error: {e}")
            except Exception as e:
                logger.error(f"DSPy LM request failed: {e}")
                raise

    return LocalDSPyLM()
