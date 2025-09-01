"""
agent.py
========
LLM-driven agent to play a single turn of the SET game using the image-only
environment provided by Set/env.py. The agent renders a system prompt from the
Jinja2 template in `Set/templates/system_prompt_template.j2`, sends the current
board image to an LLM (GPT-5), parses the model's action, and applies it to the
environment.

Notes
-----
- Requires the `openai` and `Jinja2` packages. Set `OPENAI_API_KEY` in env.
- Handles index conversion internally. The board image overlays 1-based
  labels; the model uses these directly and the code converts to 0-based internally.
- Only simulates a single turn: either select(i,j,k) or deal_three().
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from PIL import Image

# ========================
# LOGGING CONFIGURATION
# ========================
# Set ENABLE_LOGGING = True to enable logging, False to disable
ENABLE_LOGGING = True

def setup_logging():
    """Configure logging for the agent module."""
    if ENABLE_LOGGING:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('set_agent.log')
            ]
        )
    else:
        logging.disable(logging.CRITICAL)

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    # Prefer explicit import path for local execution
    from .env import (
        SetEnv,
        INITIAL_BOARD_SIZE,
        MAX_BOARD_SIZE,
        BOARD_INCREASE_STEP,
    )
except ImportError:  # direct execution fallback
    from env import (
        SetEnv,
        INITIAL_BOARD_SIZE,
        MAX_BOARD_SIZE,
        BOARD_INCREASE_STEP,
    )


# -----------------
# Prompt preparation
# -----------------

def _render_system_prompt(require_initial_set: bool) -> str:
    """Render the system prompt from the Jinja2 template with environment vars.

    Does not raise if Jinja2 is missing; instead provides a helpful error.
    """
    logger.debug(f"Rendering system prompt with require_initial_set={require_initial_set}")
    template_path = os.path.join(os.path.dirname(__file__), "templates", "system_prompt_template.j2")
    context = {
        "board_capacity": INITIAL_BOARD_SIZE,
        "grid_rows": 3,
        "max_board_size": MAX_BOARD_SIZE,
        "board_increase_step": BOARD_INCREASE_STEP,
        "strategy_hints": True,
        "require_initial_set": require_initial_set,
        # Sharpen the immediate objective for a single turn
        "objective": "find exactly one valid Set OR request three more cards (one action only)",
    }

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except Exception as e:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Jinja2 is required to render the system prompt. Install with `pip install Jinja2`."
        ) from e

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(template_path)),
        autoescape=select_autoescape(enabled_extensions=(".j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template(os.path.basename(template_path))
    rendered = tmpl.render(**context)

    # Add strict output format instruction to enable robust parsing
    rendered += (
        "\n\nSTRICT OUTPUT FORMAT:\n"
        "Respond with a single JSON object only, no prose.\n"
        "Schema (1-based indices as shown on cards):\n"
        "- For selecting a set: {\"action\": \"select\", \"indices\": [i, j, k]}\n"
        "- For dealing three: {\"action\": \"deal_three\"}\n"
        "Use the exact numbers shown on the card faces.\n"
    )
    logger.debug(f"System prompt rendered, length: {len(rendered)} characters")
    return rendered


# -----------------
# Image utilities
# -----------------

def _pil_to_png_base64(img: Image.Image) -> str:
    logger.debug(f"Converting PIL image to base64, size: {img.size}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    logger.debug(f"Base64 conversion complete, data length: {len(data)} characters")
    return f"data:image/png;base64,{data}"


# -----------------
# OpenAI client helpers
# -----------------

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


# -----------------
# Parsing utilities
# -----------------

Action = Union[Tuple[str, Tuple[int, int, int]], Tuple[str, None]]


def _parse_action(s: str) -> Action:
    """Parse the model's response into an action.

    Accepts strictly JSON. If JSON parsing fails, attempts to recover from
    common formats like `select(i, j, k)`.
    Returns:
      - ("select", (i,j,k))
      - ("deal_three", None)
    Raises ValueError on irrecoverable formats.
    """
    logger.debug(f"Parsing action from response: {s[:100]}..." if len(s) > 100 else f"Parsing action from response: {s}")
    s = s.strip()
    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            act = obj.get("action")
            if act == "select":
                idx = obj.get("indices")
                if (
                    isinstance(idx, list)
                    and len(idx) == 3
                    and all(isinstance(v, int) for v in idx)
                ):
                    action_result = ("select", (int(idx[0]), int(idx[1]), int(idx[2])))
                    logger.debug(f"Parsed JSON select action: {action_result}")
                    return action_result
            elif act == "deal_three":
                logger.debug("Parsed JSON deal_three action")
                return ("deal_three", None)
    except Exception:
        pass

    # Fallback: parse patterns like select(0, 4, 8)
    logger.debug("JSON parsing failed, trying regex fallback")
    m = re.search(r"select\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", s, re.I)
    if m:
        action_result = ("select", (int(m.group(1)), int(m.group(2)), int(m.group(3))))
        logger.debug(f"Parsed regex select action: {action_result}")
        return action_result

    if re.search(r"deal_three\s*\(\s*\)\s*", s, re.I) or s.lower().strip() == "deal_three":
        logger.debug("Parsed regex deal_three action")
        return ("deal_three", None)

    logger.error(f"Failed to parse action from: {s[:200]}")
    raise ValueError(f"Unrecognized action format: {s[:200]}")


# -----------------
# Public API
# -----------------

@dataclass
class TurnResult:
    """Outcome for a single agent turn."""
    action: str
    indices: Optional[Tuple[int, int, int]]
    env_message: Optional[str]
    initial_image: Image.Image
    result_image: Optional[Image.Image]


def simulate_single_turn(seed: Optional[int] = None, *, save_prefix: Optional[str] = None) -> TurnResult:
    """Run one SET turn with a GPT-5 agent.

    - Creates a new environment (optionally seeded)
    - Renders the system prompt and sends the initial board image to GPT-5
    - Parses the returned action and applies it to the environment
    - Optionally saves images with the given prefix
    """
    logger.info(f"Starting single turn simulation with seed={seed}, save_prefix={save_prefix}")
    env = SetEnv(seed=seed)
    logger.debug(f"Created SetEnv with seed={seed}")
    initial_img = env.get_board_image()
    logger.debug(f"Generated initial board image, size: {initial_img.size}")
    if save_prefix:
        filename = f"{save_prefix}_initial.png"
        initial_img.save(filename)
        logger.debug(f"Saved initial board image to {filename}")

    system_prompt = _render_system_prompt(require_initial_set=env.require_initial_set)
    img_data_url = _pil_to_png_base64(initial_img)

    user_parts: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Here is the current board. Respond with a single JSON object only, "
                "per the STRICT OUTPUT FORMAT (use 1-based numbers as shown)."
            ),
        },
        {
            "type": "input_image",
            "image_url": img_data_url,
        },
    ]

    reply_text = _respond_with_gpt5(instructions=system_prompt, user_parts=user_parts)
    logger.debug("Received response from GPT-5, parsing action")
    action, indices = _parse_action(reply_text)
    logger.info(f"Parsed action: {action}, indices: {indices}")

    if action == "select":
        assert indices is not None
        logger.info(f"Executing select action with 1-based indices: {indices}")
        # Convert 1-based indices to 0-based for environment
        zero_based_indices = (indices[0] - 1, indices[1] - 1, indices[2] - 1)
        logger.debug(f"Converted to 0-based indices: {zero_based_indices}")
        
        msg, img = env.select(zero_based_indices)
        logger.debug(f"Select action result: {msg}")
    elif action == "deal_three":
        logger.info("Executing deal_three action")
        msg, img = env.deal_three()
        logger.debug(f"Deal_three action result: {msg}")
        print('Dealing three cards')
    else:
        logger.error(f"Unexpected action received: {action}")
        raise RuntimeError(f"Unexpected action: {action}")

    if save_prefix and img is not None:
        filename = f"{save_prefix}_after.png"
        img.save(filename)
        logger.debug(f"Saved result board image to {filename}")

    result = TurnResult(
        action=action,
        indices=indices if action == "select" else None,
        env_message=msg,
        initial_image=initial_img,
        result_image=img,
    )
    logger.info(f"Turn simulation completed. Action: {action}, Message: {msg}")
    return result


if __name__ == "__main__":  # Simple CLI demo
    import argparse

    parser = argparse.ArgumentParser(description="Run a single SET turn via GPT-5 agent.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="board",
        help="Prefix for saved images (prefix_initial.png, prefix_after.png)",
    )
    args = parser.parse_args()

    result = simulate_single_turn(seed=args.seed, save_prefix=args.save_prefix)
    print(json.dumps({
        "action": result.action,
        "indices": result.indices,
        "env_message": result.env_message,
        "saved": {
            "initial": f"{args.save_prefix}_initial.png" if args.save_prefix else None,
            "after": f"{args.save_prefix}_after.png" if args.save_prefix else None,
        },
    }, indent=2))
