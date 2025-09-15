"""
inference.py
============
Inference functions for different model providers (OpenAI, vLLM, OpenRouter).
Provides unified interface for multimodal prompting across different backends.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import requests
from typing import Any, Dict, List
from PIL import Image

logger = logging.getLogger(__name__)


def _respond_with_gpt5(*, instructions: str, user_parts: List[Dict[str, Any]], model: str = "gpt-5", temperature: float = 1.0) -> str:
    """Send a multimodal prompt to GPT-5 via the Responses API and return text."""
    logger.debug(
        f"Sending Responses API request to {model} with {len(user_parts)} content parts and temperature={temperature}"
    )
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "The OpenAI SDK is required. Install with `pip install openai`."
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    logger.debug("Making OpenAI Responses API call")
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": user_parts,
            }
        ],
        temperature=temperature,
    )
    # Prefer helper property when available
    content = getattr(resp, "output_text", None)
    if content is None:
        try:
            chunks: List[str] = []
            for item in getattr(resp, "output", []) or []:  # type: ignore[attr-defined]
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        chunks.append(getattr(c, "text", ""))
            content = "".join(chunks)
        except Exception:
            content = ""
    content = (content or "").strip()
    logger.debug(
        f"Received Responses API content length: {len(content)} characters"
    )
    logger.debug(
        f"Raw model response: {content[:200]}..." if len(content) > 200 else f"Raw model response: {content}"
    )
    return content


def _respond_with_vllm(*, instructions: str, user_parts: List[Dict[str, Any]], model: str, base_url: str = "http://localhost:8000/v1", temperature: float = 1.0, api_key: str = "EMPTY") -> str:
    """Send a multimodal prompt to vLLM server via OpenAI-compatible API and return text."""
    logger.debug(
        f"Sending vLLM request to {base_url} with model {model} and {len(user_parts)} content parts and temperature={temperature}"
    )
    
    # Prepare messages for OpenAI format
    messages = [
        {"role": "system", "content": instructions}
    ]
    
    # Convert user_parts to OpenAI chat format
    user_content = []
    for part in user_parts:
        if part.get("type") == "input_text":
            user_content.append({
                "type": "text",
                "text": part.get("text", "")
            })
        elif part.get("type") == "input_image":
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": part.get("image_url", "")
                }
            })
    
    messages.append({
        "role": "user", 
        "content": user_content
    })
    
    # Make request to vLLM server
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        logger.debug(f"Received vLLM content length: {len(content)} characters")
        logger.debug(
            f"Raw model response: {content[:200]}..." if len(content) > 200 else f"Raw model response: {content}"
        )
        return content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"vLLM request failed: {e}")
        raise RuntimeError(f"Failed to get response from vLLM server: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected vLLM response format: {e}")
        raise RuntimeError(f"Unexpected response format from vLLM server: {e}")


def _respond_with_openrouter(*, instructions: str, user_parts: List[Dict[str, Any]], model: str, temperature: float = 1.0) -> str:
    """Send a multimodal prompt to OpenRouter API and return text."""
    logger.debug(
        f"Sending OpenRouter request with model {model} and {len(user_parts)} content parts and temperature={temperature}"
    )
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")
    
    # Prepare messages for OpenAI format
    messages = [
        {"role": "system", "content": instructions}
    ]
    
    # Convert user_parts to OpenAI chat format
    user_content = []
    for part in user_parts:
        if part.get("type") == "input_text":
            user_content.append({
                "type": "text",
                "text": part.get("text", "")
            })
        elif part.get("type") == "input_image":
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": part.get("image_url", "")
                }
            })
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    # Make request to OpenRouter
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/RLDiary/Set-RLGym",
        "X-Title": "Set-RLGym"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        logger.debug(f"Received OpenRouter content length: {len(content)} characters")
        logger.debug(
            f"Raw model response: {content[:200]}..." if len(content) > 200 else f"Raw model response: {content}"
        )
        return content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter request failed: {e}")
        raise RuntimeError(f"Failed to get response from OpenRouter: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected OpenRouter response format: {e}")
        raise RuntimeError(f"Unexpected response format from OpenRouter: {e}")