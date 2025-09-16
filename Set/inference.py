"""
inference.py
============
Inference functions for different model providers (OpenAI, vLLM, OpenRouter).
Provides unified interface for multimodal prompting across different backends.
"""

from __future__ import annotations
from openai import OpenAI
import base64
import io
import json
import logging
import os
import requests
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


def _respond_with_gpt5(
    *,
    instructions: str,
    user_parts: List[Dict[str, Any]],
    model: str = "gpt-5",
    temperature: float = 1.0,
    reasoning_effort: str = "medium",
    reasoning_summary: str = "detailed",
) -> Tuple[str, Optional[str]]:
    """Send a multimodal prompt to GPT-5 via the Responses API and return (content, reasoning).

    Returns a tuple of the final output text and an optional reasoning trace if
    the model provides one via the Responses API structured output.
    """
    logger.debug(
        f"Sending Responses API request to {model} with {len(user_parts)} content parts and temperature={temperature}"
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    logger.debug("Making OpenAI Responses API call")
    resp = client.chat.completions.create(
    model='Qwen2.5-VL-3B-Instruct/',
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image? Be specific."},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        }
    ],
    temperature=0.2,
    max_tokens=512,
)
    # Prefer helper property when available
    # Extract output text and optional reasoning trace
    content = getattr(resp, "output_text", None)
    reasoning: Optional[str] = None

    if content is None or content.strip() == "":
        try:
            text_chunks: List[str] = []
            reasoning_chunks: List[str] = []
            for item in getattr(resp, "output", []) or []:  # type: ignore[attr-defined]
                item_type = getattr(item, "type", None)
                if item_type == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            text_chunks.append(getattr(c, "text", ""))
                elif item_type == "reasoning":
                    summary_list = getattr(item, "summary", None)
                    if isinstance(summary_list, list):
                        for s in summary_list:
                            s_type = s.get("type") if isinstance(s, dict) else getattr(s, "type", None)
                            if s_type == "summary_text":
                                s_text = s.get("text") if isinstance(s, dict) else getattr(s, "text", "")
                                if s_text:
                                    reasoning_chunks.append(s_text)
            content = "".join(text_chunks)
            if reasoning_chunks:
                reasoning = "\n".join(reasoning_chunks).strip() or None
        except Exception:
            content = content or ""

    content = (content or "").strip()
    # Extract reasoning using the Responses API top-level reasoning object when available.
    # See sample format: resp.reasoning = { "effort": ..., "summary": ... }
    if reasoning is None:
        logger.debug("No reasoning returned from Responses API")
    else:
        logger.debug(f"Received reasoning: {reasoning}")

    logger.debug(f"Received Responses API content length: {len(content)} characters")
    if reasoning is not None:
        logger.debug(f"Received reasoning length: {len(reasoning)} characters")
    logger.debug(
        f"Raw model response: {content[:200]}..." if len(content) > 200 else f"Raw model response: {content}"
    )
    return content, reasoning


def _respond_with_vllm(
    *,
    instructions: str,
    user_parts: List[Dict[str, Any]],
    model: str,
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 1.0,
    api_key: str = "EMPTY",
) -> Tuple[str, Optional[str]]:
    """Send a multimodal prompt to vLLM Responses API and return (content, reasoning|None).

    Uses the OpenAI Responses-style endpoint exposed by vLLM at `{base_url}/responses`.
    The payload mirrors the one used with the OpenAI SDK in `_respond_with_gpt5`.
    """
    logger.debug(
        f"Sending vLLM request to {base_url} with model {model} and {len(user_parts)} content parts and temperature={temperature}"
    )
    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        resp = client.chat.completions.create(
        model=model,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                ]
            }
        ],
            
        ],
        temperature=temperature,
        )

        result = resp.output[0].content[0].text

        # Extract output text
        content: Optional[str] = result.get("output_text")
        reasoning: Optional[str] = None

        if not content:
            try:
                text_chunks: List[str] = []
                reasoning_chunks: List[str] = []
                for item in result.get("output", []) or []:
                    item_type = item.get("type")
                    if item_type == "message":
                        for c in item.get("content", []) or []:
                            if c.get("type") == "output_text":
                                text_chunks.append(c.get("text", ""))
                    elif item_type == "reasoning":
                        summary_list = item.get("summary")
                        if isinstance(summary_list, list):
                            for s in summary_list:
                                if isinstance(s, dict) and s.get("type") == "summary_text":
                                    s_text = s.get("text")
                                    if s_text:
                                        reasoning_chunks.append(s_text)
                content = "".join(text_chunks)
                if reasoning_chunks:
                    reasoning = "\n".join(reasoning_chunks).strip() or None
            except Exception:
                content = content or ""

        content = (content or "").strip()

        # Top-level reasoning object as fallback
        if reasoning is None:
            r = result.get("reasoning")
            if isinstance(r, dict):
                summary = r.get("summary")
                if isinstance(summary, str) and summary.strip():
                    reasoning = summary.strip()
                elif isinstance(summary, list):
                    # Some proxies might already unwrap summary into list entries
                    texts = [s.get("text", "") for s in summary if isinstance(s, dict) and s.get("type") == "summary_text"]
                    rt = "\n".join([t for t in texts if t]).strip()
                    reasoning = rt or None

        logger.debug(f"Received vLLM content length: {len(content)} characters")
        if reasoning:
            logger.debug(f"Received vLLM reasoning length: {len(reasoning)} characters")
        logger.debug(
            f"Raw model response: {content[:200]}..." if len(content) > 200 else f"Raw model response: {content}"
        )
        return content, reasoning

    except requests.exceptions.RequestException as e:
        logger.error(f"vLLM request failed: {e}")
        raise RuntimeError(f"Failed to get response from vLLM server: {e}")
    except Exception as e:
        logger.error(f"Unexpected vLLM response format: {e}")
        raise RuntimeError(f"Unexpected response format from vLLM server: {e}")


def _respond_with_openrouter(
    *,
    instructions: str,
    user_parts: List[Dict[str, Any]],
    model: str,
    temperature: float = 1.0,
) -> Tuple[str, Optional[str]]:
    """Send a multimodal prompt to OpenRouter API and return (content, reasoning|None)."""
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
        reasoning: Optional[str] = None  # OpenRouter OpenAI-compatible API typically does not include reasoning
        return content, reasoning
        
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter request failed: {e}")
        raise RuntimeError(f"Failed to get response from OpenRouter: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected OpenRouter response format: {e}")
        raise RuntimeError(f"Unexpected response format from OpenRouter: {e}")
