"""
run.py
======
Main entry point for running SET game with different inference backends.
Supports OpenAI GPT-5, vLLM local models, and OpenRouter models.
"""

import argparse
import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    # Prefer explicit import path for local execution
    from .agent import simulate_single_turn, TurnResult, setup_logging
    from .inference import _respond_with_gpt5, _respond_with_vllm, _respond_with_openrouter
except ImportError:  # direct execution fallback
    from agent import simulate_single_turn, TurnResult, setup_logging
    from inference import _respond_with_gpt5, _respond_with_vllm, _respond_with_openrouter


def create_inference_function(backend: str, **kwargs):
    """Create an inference function based on the backend choice."""
    if backend == "openai":
        model = kwargs.get("model", "gpt-5")
        temperature = kwargs.get("temperature", 1.0)
        return lambda instructions, user_parts: _respond_with_gpt5(
            instructions=instructions, 
            user_parts=user_parts, 
            model=model, 
            temperature=temperature
        )
    
    elif backend == "vllm":
        model = kwargs.get("model", "llama-3.2-90b-vision-instruct")
        base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        temperature = kwargs.get("temperature", 1.0)
        api_key = kwargs.get("api_key", "EMPTY")
        return lambda instructions, user_parts: _respond_with_vllm(
            instructions=instructions,
            user_parts=user_parts,
            model=model,
            base_url=base_url,
            temperature=temperature,
            api_key=api_key
        )
    
    elif backend == "openrouter":
        model = kwargs.get("model", "anthropic/claude-3.5-sonnet")
        temperature = kwargs.get("temperature", 1.0)
        return lambda instructions, user_parts: _respond_with_openrouter(
            instructions=instructions,
            user_parts=user_parts,
            model=model,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_game(
    backend: str = "openai",
    model: Optional[str] = None,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    save_prefix: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = False
) -> TurnResult:
    """Run a single turn of the SET game with specified configuration."""
    
    # Set up logging
    setup_logging()
    
    # Set default models per backend
    if model is None:
        if backend == "openai":
            model = "gpt-5"
        elif backend == "vllm":
            model = "llama-3.2-90b-vision-instruct"
        elif backend == "openrouter":
            model = "anthropic/claude-3.5-sonnet"
    
    # Create inference function with parameters
    inference_kwargs = {
        "model": model,
        "temperature": temperature
    }
    
    if backend == "vllm":
        inference_kwargs["base_url"] = base_url or "http://localhost:8000/v1"
        inference_kwargs["api_key"] = api_key or "EMPTY"
    
    inference_fn = create_inference_function(backend, **inference_kwargs)
    
    if verbose:
        print(f"Running SET game with:")
        print(f"  Backend: {backend}")
        print(f"  Model: {model}")
        print(f"  Temperature: {temperature}")
        if backend == "vllm":
            print(f"  Base URL: {inference_kwargs['base_url']}")
        print(f"  Seed: {seed}")
        print(f"  Save prefix: {save_prefix}")
        print()
    
    # Temporarily monkey patch the inference function in agent module
    import agent
    original_respond = agent._respond_with_gpt5
    agent._respond_with_gpt5 = inference_fn
    
    try:
        result = simulate_single_turn(seed=seed, save_prefix=save_prefix)
        return result
    finally:
        # Restore original function
        agent._respond_with_gpt5 = original_respond


def main():
    """Command-line interface for running SET game with different inference backends."""
    parser = argparse.ArgumentParser(
        description="Run SET game with different inference backends (OpenAI, vLLM, OpenRouter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with OpenAI GPT-5 (default)
  python run.py --backend openai --model gpt-5
  
  # Run with local vLLM server
  python run.py --backend vllm --model llama-3.2-90b-vision-instruct --base-url http://localhost:8000/v1
  
  # Run with OpenRouter
  python run.py --backend openrouter --model anthropic/claude-3.5-sonnet
  
  # Run with specific parameters
  python run.py --backend openai --temperature 0.7 --seed 42 --save-prefix game1 --verbose
        """
    )
    
    # Backend selection
    parser.add_argument(
        "--backend", 
        choices=["openai", "vllm", "openrouter"],
        default="openai",
        help="Inference backend to use (default: openai)"
    )
    
    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use. Defaults: openai=gpt-5, vllm=llama-3.2-90b-vision-instruct, openrouter=anthropic/claude-sonnet-4"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model sampling (default: 1.0)"
    )
    
    # vLLM specific parameters
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for vLLM server (default: http://localhost:8000/v1)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for vLLM server (default: EMPTY). For OpenAI/OpenRouter, set OPENAI_API_KEY/OPENROUTER_API_KEY env vars"
    )
    
    # Game parameters
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible games"
    )
    
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="board",
        help="Prefix for saved board images (default: board)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate environment variables based on backend
    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required for OpenAI backend")
        sys.exit(1)
    elif args.backend == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is required for OpenRouter backend")
        sys.exit(1)
    
    try:
        result = run_game(
            backend=args.backend,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
            save_prefix=args.save_prefix,
            base_url=args.base_url,
            api_key=args.api_key,
            verbose=args.verbose
        )
        
        # Print results
        output = {
            "backend": args.backend,
            "model": args.model or "default",
            "action": result.action,
            "indices": result.indices,
            "env_message": result.env_message,
            "saved_images": {
                "initial": f"{args.save_prefix}_initial.png" if args.save_prefix else None,
                "after": f"{args.save_prefix}_after.png" if args.save_prefix else None,
            }
        }
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()